# 2.5D U-Net for Vascular Segmentation (HiP-CT) Tutorial

Step-by-step guide to run the Team 1 Kaggle model on the UCL CS HPC.

---

## 1. Prepare dataset and essential metadata after downloading TIFF slices

You can access the datasets on your local machine through Globus (some details here https://github.com/HiPCTProject) or using `hoa-tools` (repository here https://github.com/HumanOrganAtlas/hoa-tools). A script in the preprocessing_helpers is included to download datasets as TIFF files through `hoa-tools`. 

To alleviate the impact of intensity histogram shifts, a normalization based on z-score is conducted in the model's data loader. This data loader requires the data shape as [Z,X,Y], mean and standard deviation (SD). A script in the preprocessing_helpers is included to compute means and SD is provided, and requires the user to have produced organ masks to compute intensity from that region of interest (ROI), ignoring background intensities. I would recommend using `organ-masker` for this (available in respository https://github.com/HiPCTProject/organ-masker). You then need to manually input these values in the `dataset.py` script (all normalization is handled in the code from there).

## 2. Send data on the UCL CS HPC

### Send all local folders and subfolders containing TIFF files on cluster using `rsync`

If you on a computer wired to the UCL network or using UCL VPN, you can directly `rsync` your files to the cluster servers. On your Linux or WSL command promt, run locally:

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

Launch with `qsub` your preprocessing script (`prepare_dataset.py`). This part does not require GPU.

```
/path/to/parent  <-- folder to be passed as input
├── Subject01
├── Subject01_labels
├── Subject02
├── Subject02_labels
├── ...
└── (other stuff is allowed, will be ignored if it doesn't match)
```





