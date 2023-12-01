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
#import preprocessing as pre
#exp_num = 18
# Training and validating
# Load data from the file paths

variable_path = 'C:/Users/ttoxopeus/BEP/BEP_project/MRI_classification/variables'

exp_num = joblib.load(f'{variable_path}/exp_num.pkl') + 1
joblib.dump(exp_num, f'{variable_path}/exp_num.pkl')

subject_paths = joblib.load(f'{variable_path}/subject_paths.pkl')
image_paths = joblib.load(f'{variable_path}/image_paths.pkl')

df = joblib.load(f'{variable_path}/df.pkl')
df_split = joblib.load(f'{variable_path}/df_split.pkl')
df_test = joblib.load(f'{variable_path}/df_test.pkl')
folds = joblib.load(f'{variable_path}/folds.pkl')

mean_all = joblib.load(f'{variable_path}/mean_all.pkl')
std_all = joblib.load(f'{variable_path}/std_all.pkl')

transforms_train = joblib.load(f'{variable_path}/transforms_train.pkl')
transforms_val_test = joblib.load(f'{variable_path}/transforms_val_test.pkl')

# subject_paths = pre.subject_paths
# image_paths = pre.image_paths

# df = pre.df
# df_split = pre.df_split
# df_test = pre.df_test
# folds = pre.folds

# mean_all = pre.mean_all
# std_all = pre.std_all

# transforms_train = pre.transforms_train
# transforms_val_test = pre.transforms_val_test



# dropout_probs = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
# dropout_accs = []
# dropout_accs_avg = []
# drop = []
# for p in dropout_probs:

# Lists to store results from each fold
training_losses = []
training_accuracies = []
validation_losses = []
validation_accuracies = []
saved_epochs = []
# dropout_accs_fold = []

print(f'Experiment {exp_num}')
for fold in range(len(folds) - 1):
#for fold in [0]:
    print(f'\nTraining fold {fold + 1}/4')
    # We can then create the dataset for each fold and then wrap them into Dataloader:
    dataset_train = functions.MRI(df_split[df_split['fold'] != fold], transform=transforms_train)
    dataset_val = functions.MRI(df_split[df_split['fold'] == fold], transform=transforms_val_test)
    #dataset_test = functions.MRI(df_test, transform=transforms_val_test)

    train_dataloader = DataLoader(dataset_train, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
    val_dataloader = DataLoader(dataset_val, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
    #test_dataloader = DataLoader(dataset_test, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

    # Initialize the model for each fold
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = functions.ResNet_18(image_channels=1, num_classes=3, dropout_prob=0.2)
    model = model.to(device)

    # Now we can define the loss function and optimizer
    # We use cross entropy loss for classification
    criterion = nn.CrossEntropyLoss()
    # We use stochastic gradient descent as the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    writer = SummaryWriter(f'C:/Users/ttoxopeus/BEP/BEP_project/MRI_classification/runs/exp_{exp_num}_fold_{fold + 1}_{time.strftime("%Y-%m-%d_%H-%M-%S")}')

    # Initialize a list to store the best validation accuracy at each epoch
    best_acc = []
    best_acc_end = []

    # Initialize lists to store results for this fold
    fold_training_losses = []
    fold_training_accuracies = []
    fold_validation_losses = []
    fold_validation_accuracies = []
    
    saved_epoch = 0
    epochs = 50
    # Loop through 10 epochs
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}')

        # Train the model on the training dataset for the current epoch
        train_epoch_loss, train_epoch_acc = functions.train(model, train_dataloader, criterion, optimizer, epoch, writer, device)

        # Evaluate the model on the validation dataset for the current epoch
        val_epoch_loss, val_epoch_acc = functions.validate(model, val_dataloader, criterion, epoch, writer, device)

        # Append results for this epoch to fold-specific lists
        fold_training_losses.append(train_epoch_loss)
        fold_training_accuracies.append(train_epoch_acc)
        fold_validation_losses.append(val_epoch_loss)
        fold_validation_accuracies.append(val_epoch_acc)
        

        if epoch >= (epochs - 5):
            if len(best_acc_end) == 0:
                best_acc_end.append(val_epoch_acc)
                saved_epoch = epoch
                torch.save(model.state_dict(), f'C:/Users/ttoxopeus/BEP/BEP_project/MRI_classification/models/best_model_experiment_{exp_num}_fold_{fold + 1}.pth')
            elif val_epoch_acc > max(best_acc_end):
                best_acc_end.append(val_epoch_acc)
                print('new best: ', best_acc_end)
                saved_epoch = epoch
                torch.save(model.state_dict(), f'C:/Users/ttoxopeus/BEP/BEP_project/MRI_classification/models/best_model_experiment_{exp_num}_fold_{fold + 1}.pth')

    # Append results for this fold to global lists
    saved_epochs.append(saved_epoch)
    training_losses.append(fold_training_losses)
    training_accuracies.append(fold_training_accuracies)
    validation_losses.append(fold_validation_losses)
    validation_accuracies.append(fold_validation_accuracies)

    # Calculate average training loss and accuracy for this fold
    avg_training_loss = sum(fold_training_losses) / len(fold_training_losses)
    max_training_accuracy = max(fold_training_accuracies)
    avg_validation_loss = sum(fold_validation_losses) / len(fold_validation_losses)
    max_validation_accuracy = max(fold_validation_accuracies)
    # dropout_accs_fold.append(max_validation_accuracy)
    # Print results for this fold
    print(f'Fold {fold + 1} Results:')
    print(f'Average Training Loss: {avg_training_loss:.4f}')
    print(f'Max Training Accuracy: {max_training_accuracy:.4f}')
    print(f'Average Validation Loss: {avg_validation_loss:.4f}')
    print(f'Max Validation Accuracy: {max_validation_accuracy:.4f}')
    
    # Close the SummaryWriter for this fold
    writer.close()

# Calculate and print average results across all folds
# print(f'training_losses: {training_losses}')
# print(f'sum_training_losses: {sum(sum(i) for i in training_losses)}')
# print(f'len_training_losses: {len(training_losses[0]) * len(folds)}')
avg_training_loss_across_folds = sum(sum(i) for i in training_losses) / (len(training_losses[0]) * len(folds))
#avg_training_accuracy_across_folds = sum(sum(i) for i in training_accuracies) / (len(training_accuracies[0]) * len(folds))
avg_max_training_accuracy_across_folds = sum(max(i) for i in training_accuracies) / len(training_accuracies)
avg_validation_loss_across_folds = sum(sum(i) for i in validation_losses) / (len(validation_losses[0]) * len(folds))
avg_validation_accuracy_across_folds = sum(sum(i) for i in validation_accuracies) / (len(validation_accuracies[0]) * len(folds))
avg_max_validation_accuracy_across_folds = sum(max(i) for i in validation_accuracies) / len(validation_accuracies)

torch.save(model.state_dict(), r'C:/Users/ttoxopeus/BEP/BEP_project/MRI_classification/models/models.pth')

# dropout_accs.append(avg_max_validation_accuracy_across_folds)
# dropout_accs_avg.append(avg_validation_accuracy_across_folds)
# drop.append(sum(dropout_accs_fold)/4)

print('Average Results Across All Folds:')
print(f'Epoch numbers of the saved models: {saved_epochs}')
print(f'Average Training Loss: {avg_training_loss_across_folds:.4f}')
print(f'Average Max Training Accuracy: {avg_max_training_accuracy_across_folds:.4f}')
print(f'Average Validation Loss: {avg_validation_loss_across_folds:.4f}')
print(f'Average Max Validation Accuracy: {avg_max_validation_accuracy_across_folds:.4f}')

# print(dropout_probs)
# print(dropout_accs)
# print(dropout_accs_avg)
# print("avg accs per drop_prob: ", drop)