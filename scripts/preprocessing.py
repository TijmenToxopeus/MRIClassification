import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import PIL
import os
import random
import scipy.io
import joblib

import pandas as pd
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time

import functions


# Preprocessing
# Set the directory paths and make list of subject paths
source_directory = r'C:\Users\ttoxopeus\BEP\BEP_project\MRI_classification\data\GPS'
target_directory = r'C:\Users\ttoxopeus\BEP\BEP_project\MRI_classification\data\GPS_all'
subject_paths, image_paths = functions.make_subject_paths(source_directory, target_directory)

# Split the data into 5 folds and make 1. df: DataFrame of all data
# 2. df_split: DataFrame of train and validation data
# 3. df_test: DataFrame of test data
df, df_split, df_test, folds = functions.make_DataFrame(subject_paths, image_paths)
print(df)
# Calculate mean_all and std_all
# mean_all, std_all = functions.mean_and_std(image_paths)
mean_all = 92.66463221086266
std_all = 161.20699804115017

print(f"Mean Pixel Intensity: {mean_all}")
print(f"Standard Deviation of Pixel Intensities: {std_all}")

# Define the transforms
hw = 196

# now we can use a sliced dataframe to create a dataset for each fold: for example, we use fold 0 to 3 as training set and fold 4 as validation set
transforms_train = transforms.Compose([
    transforms.Pad((4, 0, 4, 0)),
    transforms.RandomCrop((hw, hw)),
    #transforms.RandomHorizontalFlip(),
    #transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.RandomPerspective(distortion_scale=0.4, p=0.6),
    #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0, hue=0),
    #transforms.ToTensor(),
    transforms.Normalize((mean_all,), (std_all,))
])

# what is important is to make sure the test transform is the same as the train transform, without any randomization!
transforms_val_test = transforms.Compose([
    transforms.Pad((4, 0, 4, 0)),
    transforms.CenterCrop((hw, hw)),
    # transforms.ToTensor(),
    transforms.Normalize((mean_all,), (std_all,))
])

file_path = r'C:/Users/ttoxopeus/BEP/BEP_project/MRI_classification/variables'

# Save data in the file paths
joblib.dump(subject_paths, f'{file_path}/subject_paths.pkl')
joblib.dump(image_paths, f'{file_path}/image_paths.pkl')
joblib.dump(df, f'{file_path}/df.pkl')
joblib.dump(df_split, f'{file_path}/df_split.pkl')
joblib.dump(df_test, f'{file_path}/df_test.pkl')
joblib.dump(folds, f'{file_path}/folds.pkl')
joblib.dump(mean_all, f'{file_path}/mean_all.pkl')
joblib.dump(std_all, f'{file_path}/std_all.pkl')
joblib.dump(transforms_train, f'{file_path}/transforms_train.pkl')
joblib.dump(transforms_val_test, f'{file_path}/transforms_val_test.pkl')

print('Preprocessing done')
