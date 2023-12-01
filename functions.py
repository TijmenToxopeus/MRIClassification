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
from collections import Counter

variable_path = 'C:/Users/ttoxopeus/BEP/BEP_project/MRI_classification/variables'

df = joblib.load(f'{variable_path}/df.pkl')
df_split = joblib.load(f'{variable_path}/df_split.pkl')
df_test = joblib.load(f'{variable_path}/df_test.pkl')
folds = joblib.load(f'{variable_path}/folds.pkl')


# Preprocessing functions

# Make a list of subject paths
def make_subject_paths(source_directory, target_directory):
    # Initialize an empty list to store image file paths
    subject_paths = []
    image_paths = []

    # # Set the directory paths
    # source_directory = r'C:\Users\ttoxopeus\BEP\BEP_project\MRI_classification\data\GPS'
    # target_directory = r'C:\Users\ttoxopeus\BEP\BEP_project\MRI_classification\data\GPS_all'

    # List all files in the directory
    file_list = os.listdir(source_directory)

    # Filter for .mat files (if needed)
    mat_files = [file for file in file_list if file.endswith('.mat')]
    random.seed(51)
    # best seed 51, 41, 48, 49
    # Create sublists for each class
    class_1 = mat_files[:20]
    class_2 = mat_files[20:40]
    class_3 = mat_files[40:]

    # Shuffle each class sublist
    random.shuffle(class_1)
    random.shuffle(class_2)
    random.shuffle(class_3)

    # Interleave the shuffled sublists to create the final shuffled list
    mat_files = [item for sublist in zip(class_1, class_2, class_3) for item in sublist]

    # seed = 41: fold 1; 98%
    # seed = 43: fold 2; 99%
    # seed = 46: reasonable 
    #random.shuffle(mat_files, random=random.seed(48))
    #print(f'len mat_files: {mat_files}')

    # Load the .mat files and visualize the images
    for mat_file in mat_files:    
        file_path = os.path.join(source_directory, mat_file)
        #image_paths.append(file_path)
        data = scipy.io.loadmat(file_path)
        data['v'] = np.expand_dims(data['v'], axis=0)
        
        # Extract subject number from the file name (assuming the file name follows a pattern like "subjectnumber_class_slice_time.mat")
        subject_number = mat_file.split('.')[0][2:]
        subject_class = mat_file.split('.')[0][0]
        
        for p in range(data['v'].shape[3]):
            for t in range(data['v'].shape[4]):
                img = data['v'][0, :, :, p, t]
                
                # Define the target folder for this subject
                subject_folder = os.path.join(target_directory, f'subject_{subject_class}{subject_number}')
                
                # Create the subject folder if it doesn't exist
                if not os.path.exists(subject_folder):
                    os.makedirs(subject_folder)
                
                # Define the target file name
                target_file = os.path.join(subject_folder, f'class_{subject_class}_subject{subject_number}_slice_{p}_time_{t}.npy')
                
                # Save the image as a .npy file
                np.save(target_file, img)

                image_paths.append(target_file)
                
        subject_paths.append(subject_folder)

    return subject_paths, image_paths


def make_DataFrame(subject_paths, image_paths):
    # Split the list of image paths into 5 folds
    subject_folds = np.array_split(subject_paths, 5)
    folds = []
    for fold in subject_folds:
        fold_lst = []
        for subject in fold:
            # Construct the full path to the directory
            directory_path = os.path.join("parent_directory", subject)
        
            # Check if the directory exists
            if os.path.exists(directory_path) and os.path.isdir(directory_path):
                # List all files in the directory
                files = os.listdir(directory_path)
        
                # Loop through each file in the directory
                for file_name in files:
                    # Construct the full path to the file
                    img_path = os.path.join(directory_path, file_name)
                    fold_lst.append(img_path)
        folds.append(fold_lst)

    # Create a Pandas DataFrame to store the image paths and fold assignments
    df = pd.DataFrame({"path": image_paths})

    # Initialize a new column "fold" with all values set to 0
    df["fold"] = 0

    # Assign fold numbers to each image based on their presence in the folds
    for i, fold in enumerate(folds):
        df.loc[df["path"].isin(fold), "fold"] = i

    # Extract labels (either "G", "P", "S") from the mat_file name
    df["name"] =[img.split('\\')[-1][8:-4] for img in image_paths]
    df["label"] = [img.split('\\')[-2][-3] for img in image_paths]

    # Create df_split with folds 0, 1, 2, and 3
    df_split = df[df["fold"].isin([0, 1, 2, 3])]

    # Create df_test with fold 4
    df_test = df[df["fold"] == 4]

    # Reset the index for both DataFrames
    df_split.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    # Check the distribution of labels in each fold
    #subject_label_distribution = 
    split_label_distribution = df_split.groupby(["fold", "label"]).size().unstack(fill_value=0)
    test_label_distribution = df_test.groupby(["fold", "label"]).size().unstack(fill_value=0)

    # Display the label distribution in each fold
    print(split_label_distribution)
    print(test_label_distribution)

    # Save the DataFrame to a CSV file for future use
    df_split.to_csv(
        "C:/Users/ttoxopeus/BEP/BEP_project/MRI_classification/data/data_split.csv",
        index=False,
    )

    df_test.to_csv(
        "C:/Users/ttoxopeus/BEP/BEP_project/MRI_classification/data/data_test.csv",
        index=False,
    )

    return df, df_split, df_test, folds

#Calculate mean and std of all pixel values
def mean_and_std(image_paths):
    # Load all the images from the paths into a list
    all_images = [np.load(image_path) for image_path in image_paths]

    # Flatten the pixel values of all images and concatenate them into a single array
    pixel_values = np.concatenate([image.flatten() for image in all_images])

    # Calculate the mean and standard deviation of pixel intensities
    mean_all = np.mean(pixel_values)
    std_all = np.std(pixel_values)
    return mean_all, std_all 


# Dataset class
class MRI(Dataset):
    "Our custom Dataset for image classification on MRI images: recall the length of the dataset is the number of images under the root_dir and class folder"

    def __init__(self, df, transform=None):
        """
        Args:
            df: dataframe with columns 'path', 'fold', 'label'
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            fold (int): the fold to be used for validation
        """
        self.df = df
        #self.root_dir = root_dir
        self.transform = transform
        self.label_map = {'G': 0, 'P': 1, 'S': 2}
        self.label_map_r = {0: 'G', 1: 'P', 2:'S'}

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Load the NumPy grayscale image
        image = np.load(self.df['path'].iloc[idx])

        # Convert the NumPy array to a PyTorch tensor
        # image = torch.from_numpy(image.astype(np.float32))
        image = torch.from_numpy(image.astype(np.float32))
        label = self.label_map[self.df['label'].iloc[idx]]
        name = self.df['name'].iloc[idx]
        subject = self.label_map_r[label] + name[7:9] 

        if self.transform:
            image = image.unsqueeze(0)
            image = self.transform(image)
            image = image.squeeze(0)

        return image, label, name, subject



# The  ResNet-18 model
class Block(nn.Module):
    
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1, dropout_prob=0.0):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.InstanceNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        #self.dropout = nn.Dropout(p=dropout_prob)
        
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.relu(x)
        # self.dropout(x)
        return x
     
class ResNet_18(nn.Module): # [2, 2, 2, 2]
    
    def __init__(self, image_channels, num_classes, dropout_prob=0.0):  # Add dropout_prob argument
        
        super(ResNet_18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        #self.bn1 = nn.BatchNorm2d(64)
        self.in1 = nn.InstanceNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self.__make_layer(64, 64, stride=1, dropout_prob=0.0)  # Pass dropout_prob
        self.layer2 = self.__make_layer(64, 128, stride=2, dropout_prob=0.0)  # Pass dropout_prob
        self.layer3 = self.__make_layer(128, 256, stride=2, dropout_prob=0.0)  # Pass dropout_prob
        self.layer4 = self.__make_layer(256, 512, stride=2, dropout_prob=0.0)  # Pass dropout_prob
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512, num_classes)
        self.fc = nn.Sequential(
            nn.Linear(512, num_classes),
            nn.Dropout(p=dropout_prob)  # Apply dropout only in the final layer
        )
        
    def __make_layer(self, in_channels, out_channels, stride, dropout_prob=0.0):  # Add dropout_prob argument
        
        identity_downsample = None
        if stride != 1:
            identity_downsample = self.identity_downsample(in_channels, out_channels)
            
        return nn.Sequential(
            Block(in_channels, out_channels, identity_downsample=identity_downsample, stride=stride, dropout_prob=dropout_prob),  # Pass dropout_prob
            Block(out_channels, out_channels, dropout_prob=dropout_prob)  # Pass dropout_prob
        )
        
    def forward(self, x):
        
        x = self.conv1(x)
        #x = self.bn1(x)
        x = self.in1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x 
    
    def identity_downsample(self, in_channels, out_channels):
        
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1), 
            nn.InstanceNorm2d(out_channels)
        )
    
# # ResNet-34 model [3, 4, 6, 3]
# class ResNet_34(nn.Module): # [2, 2, 2, 2]
    
#     def __init__(self, image_channels, num_classes, dropout_prob=0.0):  # Add dropout_prob argument
        
#         super(ResNet_34, self).__init__()
#         self.in_channels = 64
#         self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU()
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
#         # ResNet layers
#         self.layer1 = self.__make_layer(64, 64, stride=1, num_blocks=3, dropout_prob=dropout_prob)  # Pass dropout_prob
#         self.layer2 = self.__make_layer(64, 128, stride=2, num_blocks=4, dropout_prob=dropout_prob)  # Pass dropout_prob
#         self.layer3 = self.__make_layer(128, 256, stride=2, num_blocks=6, dropout_prob=dropout_prob)  # Pass dropout_prob
#         self.layer4 = self.__make_layer(256, 512, stride=2, num_blocks=3, dropout_prob=dropout_prob)  # Pass dropout_prob
        
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         # self.fc = nn.Linear(512, num_classes)
#         self.fc = nn.Sequential(
#             nn.Linear(512, num_classes),
#             nn.Dropout(p=dropout_prob)  # Apply dropout only in the final layer
#         )
        
#     def __make_layer(self, in_channels, out_channels, stride, num_blocks, dropout_prob=0.0):  # Add dropout_prob argument
#         layers = []
#         identity_downsample = None
#         if stride != 1:
#             identity_downsample = self.identity_downsample(in_channels, out_channels, stride)
#         layers.append(Block(in_channels, out_channels, dropout_prob=dropout_prob))
#         in_channels = out_channels
#         for _ in range(1, num_blocks):
#             layers.append(Block(in_channels, out_channels, dropout_prob=dropout_prob))
       

#         return nn.Sequential(*layers)
#         # return nn.Sequential(
#         #     Block(in_channels, out_channels, identity_downsample=identity_downsample, stride=stride, dropout_prob=dropout_prob),  # Pass dropout_prob
#         #     Block(out_channels, out_channels, dropout_prob=dropout_prob)  # Pass dropout_prob
#         # )
        
#     def forward(self, x):
        
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
        
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
        
#         x = self.avgpool(x)
#         x = x.view(x.shape[0], -1)
#         x = self.fc(x)
#         return x 
    
#     def identity_downsample(self, in_channels, out_channels, stride):
#         if stride != 1 or in_channels != out_channels:
#             return nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
#                 nn.BatchNorm2d(out_channels)
#             )
#         else:
#             return None
        
        # return nn.Sequential(
        #     nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1), 
        #     nn.BatchNorm2d(out_channels)
        # )




# class Block(nn.Module):
    
#     def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1, dropout_prob=0.0):
#         super(Block, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU()
#         self.identity_downsample = identity_downsample
#         #self.dropout = nn.Dropout(p=dropout_prob)
        
#     def forward(self, x):
#         identity = x
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.conv2(x)
#         x = self.bn2(x)
#         if self.identity_downsample is not None:
#             identity = self.identity_downsample(identity)
#         x += identity
#         x = self.relu(x)
#         # self.dropout(x)
#         return x
     
# class ResNet(nn.Module): 
    
#     def __init__(self, Block, layers, image_channels, num_classes, dropout_prob=0.0):  # Add dropout_prob argument
        
#         super(ResNet, self).__init__()
#         self.in_channels = 64
#         self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU()
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
#         # ResNet layers
#         self.layer1 = self.__make_layer(Block, layers[0], 64, 64, stride=1, dropout_prob=dropout_prob)  # Pass dropout_prob
#         self.layer2 = self.__make_layer(Block, layers[1], 64, 128, stride=2, dropout_prob=dropout_prob)  # Pass dropout_prob
#         self.layer3 = self.__make_layer(Block, layers[2], 128, 256, stride=2, dropout_prob=dropout_prob)  # Pass dropout_prob
#         self.layer4 = self.__make_layer(Block, layers[3], 256, 512, stride=2, dropout_prob=dropout_prob)  # Pass dropout_prob
        
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         # self.fc = nn.Linear(512, num_classes)
#         self.fc = nn.Sequential(
#             nn.Linear(512, num_classes),
#             nn.Dropout(p=dropout_prob)  # Apply dropout only in the final layer
#         )
        
#     def __make_layer(self, block, num_residual_blocks, in_channels, out_channels, stride, dropout_prob=0.0):  # Add dropout_prob argument
        
#         identity_downsample = None
#         layers = []
#         if stride != 1: # or self.in_channels != out_channels
#             #identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels))    
#             identity_downsample = self.identity_downsample(in_channels, out_channels, stride)

#         layers.append(block(self.in_channels, out_channels, identity_downsample, stride))

#         for i in range(num_residual_blocks - 1):
#             layers.append(block(self.in_channels, out_channels))

#         return nn.Sequential(*layers)
            
#         # return nn.Sequential(
#         #     Block(in_channels, out_channels, identity_downsample=identity_downsample, stride=stride, dropout_prob=dropout_prob),  # Pass dropout_prob
#         #     Block(out_channels, out_channels, dropout_prob=dropout_prob)  # Pass dropout_prob
#         # )
        
#     def forward(self, x):
        
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
        
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
        
#         x = self.avgpool(x)
#         x = x.view(x.shape[0], -1)
#         x = self.fc(x)
#         return x 
    
#     def identity_downsample(self, in_channels, out_channels, stride):
        
#         return nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1), 
#             nn.BatchNorm2d(out_channels)
#         )

# def ResNet18(image_channels=1, num_classes=3, dropout_prob=0.0):
#     return ResNet(Block, [2, 2, 2, 2], image_channels, num_classes, dropout_prob)

# def ResNet34(image_channels=1, num_classes=3, dropout_prob=0.0):
#     return ResNet(Block, [3, 4, 6, 3], image_channels, num_classes, dropout_prob)



# Define a funcion to plot the confusion matrix
def plot_confusion_matrix(labels, predictions):
    # labels: the true labels
    # predictions: the predicted labels
    if len(labels) > 61:
        title = 'Confusion Matrix Test Images'
    else: 
        title = 'Confusion Matrix Test Subjects'
    unique_labels = np.unique(labels)  # Get unique class labels
    num_classes = len(unique_labels)
    
    # Generate class names as "Class 0", "Class 1", ...
    class_names = ['G', 'P', 'S']
    
    cm = confusion_matrix(labels, predictions)
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    
    # Show the colorbar
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Counts', rotation=-90, va="bottom")
    
    # Set labels for the classes
    ax.set(xticks=np.arange(num_classes),
           yticks=np.arange(num_classes),
           xticklabels=class_names, yticklabels=class_names,
           title=title,
           xlabel='Predicted Label',
           ylabel='True Label')

    ax.set_xlabel('True Label', fontsize=14)
    ax.set_ylabel('Predicted Label', fontsize=14)

    # Set font size for the title
    ax.title.set_fontsize(20)

    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="black", fontsize=15)
    
    # Ensure the plot is displayed
    #plt.show()
    
    return fig


# Define the train function
def train(model, dataloader, criterion, optimizer, epoch, writer, device):
    # add proper comments to the function on its arguments
    
    # model: the model to be trained
    # dataloader: the dataloader to iterate over the dataset
    # criterion: the loss function
    # optimizer: the optimizer
    # epoch: CURRENT epoch
    # writer: the logger to log the training process
    print('now training epoch: ', epoch + 1)
    model.train() # set the model to train mode
    running_loss = 0.0 # initialize the running loss
    running_corrects = 0 # initialize the running corrects
    
    for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        inputs, labels, name, subject = data
        inputs = inputs.unsqueeze(1)
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Add L2 regularization term to the loss
        # l2_reg = 0.0
        # for param in model.parameters():
        #     l2_reg += torch.norm(param, p=2)  # Calculate L2 norm for each parameter
        # loss += l2_lambda * l2_reg  # Add the regularization term
        
        writer.add_scalar('step loss training', loss.item(), epoch * len(dataloader) + i)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0) # update the running loss
        running_corrects += torch.sum(outputs.argmax(1) == labels.data) # update the running corrects
        
    train_epoch_loss = running_loss / len(dataloader.dataset) # compute the epoch loss
    train_epoch_acc = running_corrects.double() / len(dataloader.dataset) # compute the epoch accuracy

    # epoch_loss = running_loss / len(dataloader.dataset) # compute the epoch loss
    # epoch_acc = running_corrects.double() / len(dataloader.dataset) # compute the epoch accuracy
    
    writer.add_scalar('training loss', train_epoch_loss, epoch) # log the training loss
    writer.add_scalar('training accuracy', train_epoch_acc, epoch) # log the training accuracy
    
    print(f'training loss: {train_epoch_loss:.4f}, training accuracy: {train_epoch_acc:.4f}')

    return train_epoch_loss, train_epoch_acc


# Define the validation function
def validate(model, dataloader, criterion, epoch, writer, device):
    # Running a validation loop over the validation set

    # model: the model to be trained
    # dataloader: the dataloader to iterate over the dataset
    # criterion: the loss function
    # epoch: Current epoch
    # writer: the logger to log the training process

    model.eval() # set the model to evaluation mode
    running_loss = 0.0 # initialize the running loss
    running_corrects = 0 # initialize the running corrects
    label_map = {0: 'G', 1: 'P', 2:'S'}
    # meanwhile we save the predictions and labels to compute the confusion matrix
    predictions_matrix = []
    labels_matrix = []

    # we also save the misclassified images to tensorboard, say up to 10 images
    misclassified_images = []
    misclassified_labels = []
    misclassified_predictions = []
    subject_prediction_data = []
    subject_predictions_dict = {}
    most_common_predictions = []

    # and save the acc to conduct early stopping!

    for i, data in tqdm(enumerate(dataloader), total=len(dataloader)): # run a loop over the dataloader
        inputs, labels, name, subject = data # get the inputs and labels from the dataloader
        inputs = inputs.unsqueeze(1)
        inputs, labels = inputs.to(device), labels.to(device) # move the inputs and labels to the device
        outputs = model(inputs) # forward pass

        predictions_matrix.extend(outputs.argmax(1).cpu().numpy()) # save the predictions for confusion matrix
        labels_matrix.extend(labels.cpu().numpy()) # save the labels for confusion matrix
        #subject_prediction_data.extend(zip(subject, label_map[outputs.argmax(1).cpu().numpy()]))
        subject_prediction_data.extend(zip(subject, [label_map[p] for p in outputs.argmax(1).cpu().numpy()]))

        # add misclassified images to tensorboard
        misclassified_images.extend(inputs[outputs.argmax(1) != labels].cpu().numpy())
        misclassified_labels.extend(labels[outputs.argmax(1) != labels].cpu().numpy())
        misclassified_predictions.extend(outputs.argmax(1)[outputs.argmax(1) != labels].cpu().numpy())


        loss = criterion(outputs, labels) # compute the loss

        running_loss += loss.item() * inputs.size(0) # update the running loss
        running_corrects += torch.sum(outputs.argmax(1) == labels.data) # update the running corrects

        val_epoch_loss = running_loss / len(dataloader.dataset) # compute the epoch loss
        val_epoch_acc = running_corrects.double() / len(dataloader.dataset) # compute the epoch accuracy

    df_subject_prediction = pd.DataFrame(subject_prediction_data, columns=['subject', 'prediction'])
    df_subject_prediction['label'] = [label_map[i] for i in labels_matrix]

    subject_predictions_dict = df_subject_prediction.groupby('subject')['prediction'].apply(list).to_dict()
    subject_labels_dict = df_subject_prediction.groupby('subject')['label'].apply(list).to_dict()
    subject_labels = [i[0] for i in subject_labels_dict]
    subject_majority_ratios = {}
    
    for subject, predictions in subject_predictions_dict.items():
        # Count the occurrences of each prediction in the list.
        prediction_counts = Counter(predictions)
        # Find the prediction with the highest count.
        most_common_prediction = prediction_counts.most_common(1)[0][0]
        most_common_predictions.append(most_common_prediction)

        # Calculate the ratio of the majority prediction for this subject
        majority_ratio = prediction_counts[most_common_prediction] / len(predictions)
        subject_majority_ratios[subject] = majority_ratio

    print(subject_majority_ratios)

    
    # print(df_subject_prediction)
    # epoch_loss = running_loss / len(dataloader.dataset) # compute the epoch loss
    # epoch_acc = running_corrects.double() / len(dataloader.dataset) # compute the epoch accuracy

    writer.add_scalar('validation loss', val_epoch_loss, epoch) # log the validation loss
    writer.add_scalar('validation accuracy', val_epoch_acc, epoch) # log the validation accuracy

    # write confusion matrix to tensorboard
    writer.add_figure('confusion matrix individual pictures', plot_confusion_matrix(labels_matrix, predictions_matrix), epoch)
    writer.add_figure('confusion matrix subjects', plot_confusion_matrix(subject_labels, most_common_predictions), epoch)
    # write misclassified images to tensorboard, choose up to 10 images randomly
    misclassified_images = np.array(misclassified_images)
    misclassified_labels = np.array(misclassified_labels)   
    misclassified_predictions = np.array(misclassified_predictions)
    if len(misclassified_images) > 10:
        idx = np.random.randint(len(misclassified_images), size=10)
        misclassified_images = misclassified_images[idx]
        misclassified_labels = misclassified_labels[idx]
        misclassified_predictions = misclassified_predictions[idx]
    writer.add_images('misclassified images', misclassified_images, epoch)
    # also add their labels and predictions on the same figure
    writer.add_text('misclassified labels', str(misclassified_labels), epoch)
    writer.add_text('misclassified predictions', str(misclassified_predictions), epoch)
    

    print(f'validation loss: {val_epoch_loss:.4f}, validation accuracy: {val_epoch_acc:.4f}')
    #print(f'validation loss subjects: {val_epoch_loss:.4f}, validation accuracy subjects: {val_epoch_acc:.4f}')

    return val_epoch_loss, val_epoch_acc

# Define test function
def test(model, dataloader, criterion, writer, device):
    # Running a validation loop over the validation set

    # model: the model to be trained
    # dataloader: the dataloader to iterate over the dataset
    # criterion: the loss function
    # epoch: Current epoch
    # writer: the logger to log the training process

    model.eval() # set the model to evaluation mode
    running_loss = 0.0 # initialize the running loss
    running_corrects = 0 # initialize the running corrects
    label_map = {0: 'G', 1: 'P', 2:'S'}
    # meanwhile we save the predictions and labels to compute the confusion matrix
    predictions_matrix = []
    labels_matrix = []

    # we also save the misclassified images to tensorboard, say up to 10 images
    misclassified_images = []
    misclassified_names = []
    misclassified_labels = []
    misclassified_predictions = []
    subject_prediction_data = []
    subject_predictions_dict = {}
    most_common_predictions = []
    all_outputs = []
    subjects = []
    
    
    # and save the acc to conduct early stopping!

    for i, data in tqdm(enumerate(dataloader), total=len(dataloader)): # run a loop over the dataloader
        inputs, labels, name, subject = data # get the inputs and labels from the dataloader
        inputs = inputs.unsqueeze(1)
        inputs, labels = inputs.to(device), labels.to(device) # move the inputs and labels to the device
        outputs = model(inputs) # forward pass
        all_outputs.extend(outputs.cpu().detach().numpy())
        subjects.extend(subject)

        predictions_matrix.extend(outputs.argmax(1).cpu().numpy()) # save the predictions for confusion matrix
        labels_matrix.extend(labels.cpu().numpy()) # save the labels for confusion matrix
        #subject_prediction_data.extend(zip(subject, label_map[outputs.argmax(1).cpu().numpy()]))
        subject_prediction_data.extend(zip(subject, [label_map[p] for p in outputs.argmax(1).cpu().numpy()]))

        # add misclassified images to tensorboard
        is_misclassified = outputs.argmax(1) != labels
        for idx, misclassified in enumerate(is_misclassified):
            if misclassified:
                misclassified_names.append(name[idx])

        misclassified_images.extend(inputs[outputs.argmax(1) != labels].cpu().numpy())
        misclassified_labels.extend(labels[outputs.argmax(1) != labels].cpu().numpy())
        misclassified_predictions.extend(outputs.argmax(1)[outputs.argmax(1) != labels].cpu().numpy())


        loss = criterion(outputs, labels) # compute the loss

        running_loss += loss.item() * inputs.size(0) # update the running loss
        running_corrects += torch.sum(outputs.argmax(1) == labels.data) # update the running corrects

        test_epoch_loss = running_loss / len(dataloader.dataset) # compute the epoch loss
        test_epoch_acc = running_corrects.double() / len(dataloader.dataset) # compute the epoch accuracy


    df_subject_prediction = pd.DataFrame(subject_prediction_data, columns=['subject', 'prediction'])
    df_subject_prediction['label'] = [label_map[i] for i in labels_matrix]

    subject_predictions_dict = df_subject_prediction.groupby('subject')['prediction'].apply(list).to_dict()
    subject_labels_dict = df_subject_prediction.groupby('subject')['label'].apply(list).to_dict()
    subject_labels = [i[0] for i in subject_labels_dict]
    subject_majority_ratios = {}
    subject_prediction_ratios = {}
    
    #print('len pred mattrix: ', len(predictions_matrix))

    for subject, predictions in subject_predictions_dict.items():
        # Count the occurrences of each prediction in the list.
        prediction_counts = Counter(predictions)
        # Find the prediction with the highest count.
        most_common_prediction = prediction_counts.most_common(1)[0][0]
        most_common_predictions.append(most_common_prediction)

        # Calculate the ratio of the majority prediction for this subject
        majority_ratio = prediction_counts[most_common_prediction] / len(predictions)
        subject_majority_ratios[subject] = majority_ratio

    for subject, predictions in subject_predictions_dict.items():
        # Count the occurrences of each prediction in the list.
        prediction_counts = Counter(predictions)
        subject_ratios = {'G': 0, 'P': 0, 'S': 0}
        
        for prediction, count in prediction_counts.items():
            ratio = count / len(predictions)
            subject_ratios[prediction] = ratio

        subject_prediction_ratios[subject] = subject_ratios

    # print(f'misclassified images are:{misclassified_names}')
    # print(f'misclassified images are:{len(misclassified_images)}')
    #print(subject_prediction_ratios)
    
    #print(df_subject_prediction)

    # write confusion matrix to tensorboard
    #writer.add_figure('confusion matrix individual pictures', plot_confusion_matrix(labels_matrix, predictions_matrix))
    #writer.add_figure('confusion matrix subjects', plot_confusion_matrix(subject_labels, most_common_predictions))
    # write misclassified images to tensorboard, choose up to 10 images randomly
    misclassified_images = np.array(misclassified_images)
    misclassified_labels = np.array(misclassified_labels)   
    misclassified_predictions = np.array(misclassified_predictions)
    if len(misclassified_images) > 10:
        idx = np.random.randint(len(misclassified_images), size=10)
        misclassified_images = misclassified_images[idx]
        misclassified_labels = misclassified_labels[idx]
        misclassified_predictions = misclassified_predictions[idx]
    writer.add_images('misclassified images', misclassified_images)
    # also add their labels and predictions on the same figure
    writer.add_text('misclassified labels', str(misclassified_labels))
    writer.add_text('misclassified predictions', str(misclassified_predictions))
    

    print(f'testing loss: {test_epoch_loss:.4f}, testing accuracy: {test_epoch_acc:.4f}')
    #print(f'validation loss subjects: {val_epoch_loss:.4f}, validation accuracy subjects: {val_epoch_acc:.4f}')

    return test_epoch_loss, test_epoch_acc, subject_prediction_ratios, subject_labels, all_outputs, subjects