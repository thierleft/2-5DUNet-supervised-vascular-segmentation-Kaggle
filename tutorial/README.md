# 2.5D U-Net for Segmentation Tutorial

Step-by-step guide to run the Team 1 Kaggle model on the UCL CS HPC. The goal is to train/fine-tune this supervised 2.5D U-Net with ConvNeXt-tiny encoder for segmentation of vasculature or other structures in 3D HiP-CT volumes and then run inference for downstream analyses of these segmentation.

---

## 1. Prepare your Python environment for running the Kaggle Team 1 model

> **TL;DR** Obtain access to cluster and set your Python environment with specified requirements.

To get started, you first need to obtain access to the UCL computer science (CS) high performance cluster (HPC), ideally asking for additional project storage space (4TB or less) linked to your home folder and then request to be added under the "leelab" group to access the 8 H100 GPUs. Specific instructions are available following these two links https://liveuclac.sharepoint.com/sites/MSMHGroup/SitePages/High-Performance-Computing-clusters.aspx and https://hpc.cs.ucl.ac.uk/, or asking current lab members. To use conda efficiently, you will need to create an environment and move it to your project storage space. Always submit bash/conda commands within an `.sh` file with `qsub`  for job scheduling within the workload management system (SGE) of UCL CS HPC rather than typing commands directly in the command prompt which will be extremely slow to run. I have included multiple `.sh` files in the submission_scripts folders that we will use for every single step below. You will notice that we always need to source conda and CUDA for running Python scripts using GPU. 

After initializing a Python 3.9 environment with conda and having moved it in your storage space, you could run for instance the `envInstallTorch.sh` from the submissions_scripts folder and install all dependencies by submitting it as a job (run `qstat` to check if you have other jobs running):

```bash
qsub envInstallTorch.sh
```

You would now have a working conda environment that can be used for all training/fine-tuning/inference.

## 2. Prepare dataset and essential metadata after downloading TIFF slices

> **TL;DR** Need all datasets as TIFF slices with dataset dimensions, mean and standard deviation within organ mask before running training.

You can access HiP-CT datasets on your local machine through Globus (some details here https://github.com/HiPCTProject) or using `hoa-tools` (repository here https://github.com/HumanOrganAtlas/hoa-tools). A script in the preprocessing_helpers folder is included to download datasets as TIFF files through `hoa-tools`. 

To alleviate the impact of intensity histogram shifts, a normalization based on z-score is conducted in the model's data loader. This data loader requires the data shape as [Z,X,Y], mean and standard deviation (SD). A script in the preprocessing_helpers is included to compute means and SD, and requires the user to have produced whole organ masks to compute intensity histogram taken within the organ region of interest (ROI), ignoring background intensities. I would recommend using `organ-masker` for this (available in respository https://github.com/HiPCTProject/organ-masker). You then need to manually input these values in the `dataset.py` script (all normalization is handled in the code from there).

## 3. Send data on the UCL CS HPC

> **TL;DR** Choose your preferred way to transfer your TIFF datasets to the cluster.

### Send all local folders and subfolders containing TIFF files on cluster using `rsync`

If you're on a computer wired to the UCL network or with an active UCL VPN, you can directly `rsync` your files to the cluster servers. On your Linux or [WSL](https://github.com/microsoft/WSL) command promt, run locally:

```bash
rsync -avz /mnt/d/ucemlef/DATA_FOLDER ID@pryor.cs.ucl.ac.uk:/home/ID/storage/STORAGESPACE_NAME/
```

You will be promtped to input your UCL CS HPC password and then the transfer will begin. 


### Send Dropbox folders containing TIFF files on cluster using `wget`

If you have previously successfully uploaded all your datasets on Dropbox, you can download them directly to the cluster by leveraging the share link option (and switching the end `dl=0` to `dl=1`, enabling direct download). In a terminal on the cluster, submit with `qsub` an `.sh` file with these command lines (full example to download the Kaggle kidney dataset in the submission_scripts folder):

```bash
wget -O DATASET_NAME.zip "https://www.dropbox.com/DROPBOX_SHARELINK&dl=1"
UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip -o -j DATASET_NAME.zip -d DATA_FOLDER/DATASET_NAME
rm DATASET_NAME.zip
```

It will attempt zipping all files to a zipped folder, download it, and then unzip it locally, so this option may crash given a lot of TIFF images in a dataset. Do double-check that it worked and repeat if it didn't.

## 4. Preprocess dataset for training or fine-tuning

> **TL;DR** From input TIFF slices, produce memory-mapped files used to train/fine-tune.

If you want to fine-tune or train the network on your own datasets, carry on reading here, but if you simply want to run inference, skip to **Section 6** below. We now want to convert the TIFF series for both images and labels to memory-mapped volumetric files for all subjects (*e.g.* `Subject01.mmap` and `Subject01_mask.mmap`). Launch with `qsub` your preprocessing `.sh` script in the submission_scripts folder (calling `prepare_dataset.py`). *N.B. This part does not yet require GPU.* 

The Python call should look simply like:
```bash
python prepare_data.py 
    -s TRAININGDATA_FOLDER 
    -o PREPROCESSEDDATA_FOLDER
```

We will convert your dataset from

```
/path/to/TRAININGDATA_FOLDER  <-- folder to be passed as input
├── Subject01
├── Subject01_labels
├── Subject02
├── Subject02_labels
├── ...
└── (other stuff is allowed, will be ignored if it doesn't match)
```

to the following structure below, including only the memory-mapped files that will be loaded more efficiently than TIFF series later during training or fine-tuning.

```
/path/to/PREPROCESSEDDATA_FOLDER  <-- folder to be passed as output
├── Subject01.mmap
├── Subject01_mask.mmap
├── Subject02.mmap
├── Subject02_mask.mmap
└── ...
```

I like tracking the progress of my scripts (especially the training/fine-tuning) using the command line below on the output log files in the folder you specified with the `-o` flag in your `.sh` script (live updates with `-f`).
```bash
tail -f  /home/ID/storage/STORAGESPACE_NAME/LOGS_FOLDER/prepare_data.oPROCESSID
```

## 5. Launch training or fine-tuning

> **TL;DR** Train/fine-tune on all orthogonal axes and produce model weights (.pth) to be loaded later for inference.

To launch training/fine-tuning, simply edit your training `.sh` file (example in submission_scripts folder) and submit with `qsub` like above. Since the model is essentially 2D, we "slice through" the volumes along each axis for training and validation by specifying along which axis the data loader will sample the 1536 x 1536 x 3 pseudo-volumes used. This is done by adding the extension `_xz` and `_zy` at the end of your data ID like shown below (no specification means sampling along the default XY plane).


```bash
--train_groups "Subject01|Subject01_xz|Subject01_zy|Subject02|Subject02_xz|Subject02_zy"
```

Otherwise, you have a wide range of parameters you can tweak. The only one I have actually added is the `--pretrained_weights` to allow users to load weights from a previous training they might have (or for running fine-tuning on new organs/datasets after loading model weights trained on another organ/dataset for instance). Your Python call could look like this for a fine-tuning where we load model weights from a previous training:

```bash
python train.py 
    --memmap_dir PREPROCESSEDDATA_FOLDER 
    --train_groups "Subject01|Subject01_xz|Subject01_zy|Subject02|Subject02_xz|Subject02_zy" 
    --valid_groups "SubjectN|SubjectN_xz|SubjectN_zy" 
    --epochs 20 
    --lr 1e-4 
    --weight_decay 3e-5 
    --train_batch_size_per_device 6 
    --valid_batch_size_per_device 6 
    --accumulation_steps 2 
    --num_workers 2 
    --pretrained_weights PRETRAINEDWEIGHTS_FOLDER/PRETRAINEDWEIGHTS.pth 
    --output_dir OUTPUT_FOLDER
```

All the input arguments that can be added are stated below (in `train.py`):

```python
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


During training/fine-tuning, the model will save the weights of the best model (as a `.pth` file) at any epoch where the validation loss reached a new minimum. Also, as a reminder, before launching any training or fine-tuning, you need to make sure that you have updated the `dataset.py` code including metadata (shape, mean, SD) for each dataset that will be used, which should look like this:

```python
group_shapes = {
    "Kidney_1_LADAF_2021-17_right_whole_kidney": (2279, 1303, 912),
    "Kidney_2_S-20-28_kidney_sparse": (2217, 1041, 1511),
    ...
}

group_means = {
    "Kidney_1_LADAF_2021-17_right_whole_kidney": 26131.518,
    "Kidney_2_S-20-28_kidney_sparse": 34853.4,
    ...
}

group_stds = {
    "Kidney_1_LADAF_2021-17_right_whole_kidney": 1414.7509,
    "Kidney_2_S-20-28_kidney_sparse": 2336.2703,
    ...
}
```

Otherwise, all scripts are tailored for multi-GPU processing which allows you to provide access to more than one GPU in your `.sh` script using parameters such as those below. Here, we are assigning 4 GPUs (`-pe gpu 4`) which will each have access to a total of 256 GB of RAM or 64 GB per GPU (`-l tmem=64G`) for up to 24h (`-l h_rt=24:00:00`). Technically, only 4 GPUs should be assigned to your job given those parameters, but there have been times where multi-GPU scripts did not run in pure isolation, so watch out for that. Be mindful of others using the cluster, we have a total of 1TB of memory and 8 H100 GPUs, but if you use 5 GPUs with 200 GB of allocated memory for each you would end up using all available memory, even though you're not using all GPUs. Also, note that we never set `hmem` for PyTorch GPU scripts.

```bash
#!/bin/bash
#$ -l tmem=64G
#$ -l h_rt=24:00:00
#$ -l gpu=true
#$ -l gpu_type=h100
#$ -pe gpu 4
```

## 6. Running inference on new datasets

> **TL;DR** Loading your model weights, generate memory-mapped predictions and binary TIFF series matching your input image datasets.

### Accumulate averaged predictions in memory-mapped files

Now that we have a trained model, we can load the weights of the best model obtained during training/fine-tuning and run inference on new datasets. Inference does not require you to pre-convert your TIFF series to memory-mapped files, this will be done by default as a first step. Then, predictions along orthogonal axes are accumulated in another memory-mapped file after launching your inference script on a GPU node. On top of running the model at least once along each orthogonal axis, 2.5D tiles will be flipped in each direction with  `--flip` argument (already set by default) and tiles will be rotated along Z by 90, 180 and 270 degrees with `--rot` argument. Tile-specific predictions are averaged from all these augmentations before thresholding to binary mask when exporting to TIFF to alleviate some of the biases from the 2.5D processing. This step won't write TIFF slices yet, since splitting the GPU processing from the memory-mapped volume to TIFF series export was more optimal as the writing on CPU is much faster using the second script for this conversion that I adapted. So first, run with `qsub` your inference `.sh` script in the submission_scripts folder (calling `inference.py`).

```bash
python inference.py 
    --group Subject01 
    --ckpt_path PRETRAINEDWEIGHTS_FOLDER/PRETRAINEDWEIGHTS.pth 
    --axis "z|y|x" 
    --flip 3 
    --rot 3 
    --overlap 
    --input_folder /home/ID/storage/STORAGESPACE_NAME/INFERENCEDATA_FOLDER
    --output_folder /home/ID/storage/STORAGESPACE_NAME/INFERENCEOUTPUT_FOLDER

```


### Write binary TIFF series from thresholded memory-mapped files

This step now allows you to obtain full series of TIFF slices matching your original image datasets given memory-mapped files from the inference step. You could adapt this code/step to output any other type of volumetric file format (NIFTI, multi-page TIFF, NRRD, HDF5, etc.) if your datasets have a reasonable shape. The code is written to massively accelerate the writing of thousands of TIFF images using CPU multi-processing. You can launch `exportinference_toTIFF.py` based on the example `.sh` script in the submission_scripts folder with a Python call that should look like this for a single dataset:

```bash
python exportInference_toTIFF.py 
  --mmap /home/ID/storage/STORAGESPACE_NAME/INFERENCEOUTPUT_FOLDER/Subject01_mask.mmap 
  --shape 1691 1037 785 
  --out /home/ID/storage/STORAGESPACE_NAME/INFERENCEOUTPUT_FOLDER/TIF/Subject01 
  --threshold 0.4 
  --nprocs 48

```


---


If you have any questions, don't hesitate to reach out at tll42@cantab.ac.uk. 