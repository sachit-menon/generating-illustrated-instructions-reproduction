## Downloading and formatting the data

First, download https://drive.google.com/drive/folders/1hjjcNSUSqv8AbA7R-5lIKmui-ySCEWJw?usp=sharing and unzip to some base folder, for example `/wikihow/`.  

We remake the splits as our task requires all images from a given article to be in the same split. First, move all the images into one folder, e.g.,
```
cd wikihow
rsync -av --progress --partial ./test/ ./train/
```

Create the directory defining the new splits, e.g. `wikihow/splits/` and download `train.csv` and `validation.csv` to this location.

`train.csv`: 
`validation.csv`: 

You will need to change the base directory in the config for `wikihow_dataset.py` to your base directory.