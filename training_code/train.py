import os
import gc
import sys
import time
import random
import argparse
import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm
from colorama import Fore, Back, Style
from functools import partial
import warnings

import segmentation_models_pytorch as smp
from transformers import get_cosine_schedule_with_warmup

import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler

from utils import seed_everything, setup_logger, worker_init_fn, draw_sample, group_weight
from dataset import get_train_transforms, get_valid_transforms, HOADataset
from dist_utils import is_main_process, get_rank
from engine import train_one_epoch, valid_one_epoch
from loss import Loss
import random

import matplotlib.pyplot as plt
import torch.nn.functional as F

def save_training_mosaic(dataset, output_dir, n_samples=8):
    os.makedirs(output_dir, exist_ok=True)
    
    total_samples = min(n_samples * 2, len(dataset))
    indices = random.sample(range(len(dataset)), k=total_samples)
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, n_samples * 3))

    if n_samples == 1:
        axes = axes[None, :]  # Ensure 2D indexing if only one row

    for row in range(n_samples):
        for col in range(2):  # 2 pairs per row
            idx = indices[row * 2 + col]
            img, mask, _ = dataset.get_one_sample(idx)

            img_np = img.cpu().numpy().squeeze()
            img_np = img_np[0, :, :] if img_np.ndim == 3 else img_np

            mask_np = mask.cpu().numpy().squeeze()
            mask_np = mask_np[0, :, :] if mask_np.ndim == 3 else mask_np

            ax_img = axes[row, col * 2]
            ax_mask = axes[row, col * 2 + 1]

            im1 = ax_img.imshow(img_np, cmap='gray')
            ax_img.set_title(f"Image {idx}")
            ax_img.axis('off')
            fig.colorbar(im1, ax=ax_img, fraction=0.046, pad=0.04)

            im2 = ax_mask.imshow(mask_np, cmap='gray')
            ax_mask.set_title(f"Mask {idx}")
            ax_mask.axis('off')
            fig.colorbar(im2, ax=ax_mask, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mosaic_train_samples_random.png"))
    plt.close()


def save_visual_predictions(model, valid_loader, output_dir, epoch, loss_value=None, n_samples=4, min_fg_ratio=0.0001):
    import random
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans", "sans-serif"]

    model.eval()
    samples_saved = 0
    max_tries = 50

    all_batches = list(iter(valid_loader))
    random.shuffle(all_batches)

    with torch.no_grad():
        tries = 0
        for batch in all_batches:
            if samples_saved >= n_samples or tries >= max_tries:
                break

            images = batch["image"].cuda()
            masks = batch["mask"].cuda()
            preds = torch.sigmoid(model(images))
            preds_bin = (preds > 0.5).float()

            for i in range(images.shape[0]):
                pred_np = preds_bin[i].cpu().numpy().squeeze()
                mask_np = masks[i].cpu().numpy().squeeze()
                img_np = images[i].cpu().numpy().squeeze()

                if img_np.ndim == 3:
                    img_np = img_np[0]
                    mask_np = mask_np[0]
                    pred_np = pred_np[0]

                fg_ratio = np.count_nonzero(mask_np) / mask_np.size
                if fg_ratio < min_fg_ratio:
                    continue

                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                titles = ["Input", "Ground Truth", "Prediction"]
                images_to_show = [img_np, mask_np, pred_np]

                for ax, img, title in zip(axes, images_to_show, titles):
                    ax.imshow(img, cmap="gray")
                    ax.set_title(title)
                    ax.axis("off")

                fig.suptitle(f"Epoch {epoch} | Sample {samples_saved+1} | Loss: {loss_value:.4f}" if loss_value is not None else f"Epoch {epoch} | Sample {samples_saved+1}", fontsize=14)
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                save_path = os.path.join(output_dir, f"vis_epoch_{epoch}_sample_{samples_saved}.png")
                plt.savefig(save_path)
                plt.close()

                samples_saved += 1
                tries += 1
                if samples_saved >= n_samples:
                    break


def save_training_predictions(model, train_loader, output_dir, epoch, loss_value=None, n_samples=4):
    model.eval()
    samples_saved = 0
    loader_iter = iter(train_loader)

    with torch.no_grad():
        while samples_saved < n_samples:
            try:
                batch = next(loader_iter)
            except StopIteration:
                break

            image = batch["image"].cuda()
            mask = batch["mask"].cuda()

            pred = model(image)
            pred = torch.sigmoid(pred)
            pred = (pred > 0.5).float()

            for i in range(image.shape[0]):
                pred_np = pred[i].cpu().numpy().squeeze()
                mask_np = mask[i].cpu().numpy().squeeze()
                img_np = image[i].cpu().numpy().squeeze()

                if img_np.ndim == 3:
                    img_np = img_np[0]
                    mask_np = mask_np[0]
                    pred_np = pred_np[0]

                # Skip mostly empty ground truth masks
                if np.count_nonzero(mask_np) / mask_np.size < 0.0001:
                    continue

                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(img_np, cmap="gray")
                axes[0].set_title("Train Image")
                axes[1].imshow(mask_np, cmap="gray")
                axes[1].set_title("Train Ground Truth")
                axes[2].imshow(pred_np, cmap="gray")
                axes[2].set_title("Prediction")

                for ax in axes:
                    ax.axis("off")

                fig.suptitle(f"Epoch {epoch} | Train Sample {samples_saved+1} | Loss: {loss_value:.4f}" if loss_value else f"Epoch {epoch}", fontsize=12)
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                os.makedirs(output_dir, exist_ok=True)
                save_path = os.path.join(output_dir, f"train_vis_epoch_{epoch}_sample_{samples_saved}.png")
                plt.savefig(save_path)
                plt.close()

                samples_saved += 1
                if samples_saved >= n_samples:
                    break


def get_parser():
    parser = argparse.ArgumentParser(description="HOA Training")
    
    # basic
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--exp", type=str, default="hoa")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--num_workers", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    
    # data
    parser.add_argument("--memmap_dir", type=str, default="path/to/memmap")
    parser.add_argument("--image_size", type=int, default=1536)
    parser.add_argument(
        "--train_groups", 
        type=str, 
        default="kidney_1_dense|kidney_1_dense_xz|kidney_1_dense_zy|kidney_1_voi|kidney_1_voi_xz|kidney_1_voi_zy|kidney_2|kidney_2_xz|kidney_2_zy|kidney_3_sparse|kidney_3_xz|kidney_3_zy"
    )
    parser.add_argument("--valid_groups", type=str, default="kidney_3_dense")
    parser.add_argument("--normalize_dist_map", type=bool, default=False)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--rotate_slice", type=float, default=0.3)
    parser.add_argument("--rotate_slice_limit", type=float, default=30)
    
    # model
    parser.add_argument("--backbone", type=str, default="convnext_tiny")
    parser.add_argument("--upsample_method", type=str, default="nearest")
    parser.add_argument("--input_channels", type=int, default=3)
    parser.add_argument("--sync_bn", type=bool, default=True)
    parser.add_argument("--focal_coef", type=float, default=1.0)
    parser.add_argument("--dice_coef", type=float, default=1.0)
    parser.add_argument("--boundary_coef", type=float, default=0.01)
    parser.add_argument("--boundary_coef_max", type=float, default=0.01)
    parser.add_argument("--custom_loss_coef", type=float, default=1.0)
    parser.add_argument("--focal_alpha", type=float, default=0.25)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--pretrained_weights", type=str, default=None, help="Path to the pretrained .pth model weights")
    
    
    # training
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-2)
    parser.add_argument("--train_batch_size_per_device", type=int, default=1)
    parser.add_argument("--valid_batch_size_per_device", type=int, default=2)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    
    # ddp
    parser.add_argument("--dist_backend", type=str, default="nccl")
    parser.add_argument("--port", type=int, default=25555)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    
    return parser


def main_worker(local_rank, args):
    warnings.filterwarnings("ignore", category=FutureWarning)

    # local rank
    args.local_rank = local_rank
    args.rank = args.rank * args.ngpus_per_node + local_rank
    torch.cuda.set_device(args.local_rank)
    
    # Setup seed
    seed_everything(args.seed)
    
    # Setup logger
    setup_logger(
        args.output_dir,
        distributed_rank=args.local_rank,
        filename="train.log",
        mode="a"
    )
    
    # Init dist env
    dist.init_process_group(
        backend=args.dist_backend,
        init_method = f"tcp://127.0.0.1:{args.port}",
        world_size=args.world_size,
        rank=args.rank,
    )
    dist.barrier()
    
    # Log args
    if is_main_process():
        warnings.filterwarnings("ignore", category=FutureWarning)
        logger.info(args)

    print(f"[Rank {args.rank}] Using GPU {args.local_rank} / Total GPUs: {args.ngpus_per_node}")

    
    # Load dataset
    output_dist_maps = args.boundary_coef > 0.0
    train_set = HOADataset(
        args.memmap_dir,
        args.train_groups,
        channels=args.input_channels,
        transforms=get_train_transforms(args.image_size), 
        mixup=args.mixup, 
        output_dist_maps=output_dist_maps,
        normalize_dist_map=args.normalize_dist_map,
        rotate_slice_prob=args.rotate_slice,
        rotate_slice_angle_limit=args.rotate_slice_limit,
    )
    valid_set = HOADataset(
        args.memmap_dir,
        args.valid_groups,
        channels=args.input_channels,
        transforms=get_valid_transforms(args.image_size), 
        rotate_slice_prob=0.0,
    )   
    
    # Build Dataloader
    args.num_workers = int((args.num_workers + args.ngpus_per_node - 1) / args.ngpus_per_node)
    init_fn = partial(
        worker_init_fn,
        num_workers=args.num_workers,
        rank=args.rank,
        seed=args.seed
    )
    train_sampler = DistributedSampler(train_set, shuffle=True)
    valid_sampler = DistributedSampler(valid_set, shuffle=False)
    train_loader = DataLoader(
        train_set, 
        batch_size=args.train_batch_size_per_device, 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=True, 
        worker_init_fn=init_fn,
        sampler=train_sampler,
    )
    valid_loader = DataLoader(
        valid_set, 
        batch_size=args.valid_batch_size_per_device, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True, 
        sampler=valid_sampler,
    )
    
    
    #if is_main_process():
    #    logger.info("Saving training data mosaic to verify input...")
    #    save_training_mosaic(train_set, args.output_dir, n_samples=4)     
    
    logger.info(f"Train samples: {len(train_set)} | Valid samples: {len(valid_set)}")
    logger.info(f"Augmentations: \n{get_train_transforms(args.image_size)}")
        
    # Build Model
    in_channels = 3 if args.input_channels==1 else args.input_channels
    num_class = 1 if args.input_channels==1 else args.input_channels
    if "swin" in args.backbone:
        model = smp.Unet(
            encoder_name=args.backbone,
            encoder_args={"img_size": args.image_size},
            encoder_weights="imagenet",
            decoder_norm_type="LN",
            decoder_act_type="GeLU",
            decoder_use_checkpoint=True,
            in_channels=in_channels,
            classes=num_class,
            activation=None
        )
    else:
        model = smp.Unet(
            encoder_name=args.backbone,      # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            encoder_args={"in_channels": in_channels},
            decoder_norm_type="GN",
            decoder_act_type="GeLU",
            decoder_use_checkpoint=False,
            decoder_upsample_method=args.upsample_method,
            in_channels=in_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=num_class,        # model output channels (number of classes in your dataset)
            activation=None,
        )
    if args.sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logger.info(f"Using sync batchnorm")        
        
    model = nn.parallel.DistributedDataParallel(
        model.cuda(),
        device_ids=[args.local_rank],
    )
    
    if args.pretrained_weights is not None:
        map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
        state_dict = torch.load(args.pretrained_weights, map_location=map_location)
        
        # If model was trained with DataParallel/DistributedDataParallel (has 'module.' prefix)
        if list(state_dict.keys())[0].startswith("module."):
            model.load_state_dict(state_dict)
        else:
            model.module.load_state_dict(state_dict)
        
        logger.info(f"Loaded pretrained weights from {args.pretrained_weights}")    
    
    # Build loss_fn
    loss_fn = Loss(
        focal_coef=args.focal_coef, 
        dice_coef=args.dice_coef, 
        boundary_coef=args.boundary_coef, 
        custom_loss_coef=args.custom_loss_coef, 
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
    ).cuda()

    # Build optimizer & scheduler
    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    param_dict = group_weight([], model.module, args.lr)
    optimizer = torch.optim.AdamW(param_dict, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_training_steps=num_training_steps, num_warmup_steps=num_warmup_steps)
    scaler = torch.cuda.amp.GradScaler()
    
    # Check boundary loss coef
    if args.boundary_coef_max < args.boundary_coef:
        args.boundary_coef_max = args.boundary_coef
    logger.info(f"Boundary loss weight: {args.boundary_coef} -> {args.boundary_coef_max}")
    init_boundary_coef = args.boundary_coef
    boundary_coef_step = (args.boundary_coef_max-args.boundary_coef) / args.epochs
    
    # Training
    best_iou = -np.inf
    for epoch in range(1, args.epochs+1):
        
        # shuffle loader
        train_sampler.set_epoch(epoch)
        
        # update boundary loss coef
        boundary_coef = init_boundary_coef + (epoch-1)*boundary_coef_step
        loss_fn.boundary_coef = boundary_coef
        logger.info(f"Current boundary loss weight {args.boundary_coef:0.5f}")        
        
        # train
        train_one_epoch(model, optimizer, scheduler, train_loader, loss_fn, scaler, epoch, args)
        gc.collect()
        torch.cuda.empty_cache()
        
        # evaluate
        #iou = valid_one_epoch(model, valid_loader, loss_fn, epoch, args)
        iou, eval_loss = valid_one_epoch(model, valid_loader, loss_fn, epoch, args)

        gc.collect()
        torch.cuda.empty_cache()
        
        
        if is_main_process():
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"epoch_{epoch}.pth"))
            #save_visual_predictions(model, valid_loader, args.output_dir, epoch, loss_value=eval_loss)
            #save_training_predictions(model, train_loader, args.output_dir, epoch, loss_value=eval_loss)
            if iou > best_iou:
                logger.info(f"{Fore.GREEN}Validation IOU Improved ({best_iou} ---> {iou})")
                best_iou = iou
                torch.save(model.state_dict(), os.path.join(args.output_dir, f"{args.backbone}_{epoch}epoch_best.pth"))
                logger.info(f"Model saved at {os.path.join(args.output_dir, f'{args.backbone}_{epoch}epoch_best.pth')}{Style.RESET_ALL}")
                #save_visual_predictions(model, valid_loader, args.output_dir, epoch)
        
        #if epoch % 5 == 0 or epoch == args.epochs:
        #save_visual_predictions(model, valid_loader, args.output_dir, epoch)
            
        gc.collect()  
        torch.cuda.empty_cache()           


@logger.catch
def main():
    parser = get_parser()
    args = parser.parse_args()

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not cvd:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES is empty. Refusing to run to avoid using unallocated GPUs."
        )
    args.ngpus_per_node = len([x for x in cvd.split(",") if x.strip() != ""])
    args.world_size = args.ngpus_per_node  # single-node run
    # ---------------------------------------------------------------------------

    mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,))

    
if __name__ == "__main__":
    main()
    sys.exit(0)
