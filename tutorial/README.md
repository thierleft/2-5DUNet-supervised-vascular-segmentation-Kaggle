# 2.5D U-Net for Vascular Segmentation (HiP-CT) Tutorial

Step-by-step guide to run the Team 1 Kaggle model on the UCL CS HPC.

---

## 1. Prepare dataset and essential metadata after downloading TIFF slices

You can access HiP-CT datasets on your local machine through Globus (some details here https://github.com/HiPCTProject) or using `hoa-tools` (repository here https://github.com/HumanOrganAtlas/hoa-tools). A script in the preprocessing_helpers folder is included to download datasets as TIFF files through `hoa-tools`. 

To alleviate the impact of intensity histogram shifts, a normalization based on z-score is conducted in the model's data loader. This data loader requires the data shape as [Z,X,Y], mean and standard deviation (SD). A script in the preprocessing_helpers is included to compute means and SD is provided, and requires the user to have produced organ masks to compute intensity from that region of interest (ROI), ignoring background intensities. I would recommend using `organ-masker` for this (available in respository https://github.com/HiPCTProject/organ-masker). You then need to manually input these values in the `dataset.py` script (all normalization is handled in the code from there).

## 2. Send data on the UCL CS HPC

### Send all local folders and subfolders containing TIFF files on cluster using `rsync`

If you on a computer wired to the UCL network or using UCL VPN, you can directly `rsync` your files to the cluster servers. On your Linux or [WSL](https://github.com/microsoft/WSL) command promt, run locally:

```bash
rsync -avz /mnt/d/ucemlef/DATA_FOLDER ID@pryor.cs.ucl.ac.uk:/home/ID/storage/STORAGESPACE_NAME/
```

### Send Dropbox folders containing TIFF files on cluster using `wget`

If you have successfully uploaded all your datasets on Dropbox, you can download them directly to the cluster by copying the share link (and switching the end `dl=0` to `dl=1`, enabling direct download). In a terminal on the cluster, submit with `qsub` an .sh file with these command lines (full example to download the Kaggle kidney dataset in the submission_scripts folder):

```bash
wget -O DATASET_NAME.zip "https://www.dropbox.com/DROPBOX_SHARELINK&dl=1"
UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip -o -j DATASET_NAME.zip -d DATA_FOLDER/DATASET_NAME
rm DATASET_NAME.zip
```


## 3. Preprocess dataset for training or fine-tuning

We now want to convert the TIFF series for both images and labels to memory-mapped volumetric files for all subjects (e.g. Subject01.mmap and Subject01_mask.mmap). Launch with `qsub` your preprocessing .sh script in the submission_scripts folder (calling `prepare_dataset.py`). N.B. This part does not yet require GPU. We will convert your dataset from

```
/path/to/parent  <-- folder to be passed as input
├── Subject01
├── Subject01_labels
├── Subject02
├── Subject02_labels
├── ...
└── (other stuff is allowed, will be ignored if it doesn't match)
```

to the following structure, including only the memory-mapped files that will be loaded more  efficiently than TIF series later during training or fine-tuning.

```
/path/to/preprocessed data folder  <-- folder to be passed as output
├── Subject01.mmap
├── Subject01_mask.mmap
├── Subject02.mmap
├── Subject02_mask.mmap
└── ...
```

I like tracking the progress of my scripts (especially the training/fine-tuning) using the command line below on the output log files (live updates with `-f`).
```bash
tail -f  /home/ID/storage/STORAGESPACE_NAME/LOGS_FOLDER/prepare_data.oPROCESSID
```


## 4. Launch training or fine-tuning

To launch training/fine-tuning, simply edit your training .sh file (example in submission_scripts folder) and submit with `qsub` like above. Since the model is essentially 2D, we "slice through" the volumes along each axis for training and validation by specifying along which axis the data loader will sample the 1536 x 1536 x 3 pseudo-volumes used. This is done by adding the extension `_xz` and `_zy` at the end of your data ID like shown below (no specification means sampling along the default XY plane).


```bash
--train_groups "Subject01|Subject01_xz|Subject01_zy|Subject02|Subject02_xz|Subject02_zy"
```

Otherwise, you have a wide range of parameters you can tweak. The only one I have actually added is the `--pretrained_weights` to allow users to load weights from a previous training they might have (or for running fine-tuning on new organs/datasets after loading model weights trained on another organ/dataset for instance).
```bash
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
```


## 5. Running inference on new datasets

Inference does not require you to pre-convert your TIF series to memory-mapped files, this will be done by default as a first step.

