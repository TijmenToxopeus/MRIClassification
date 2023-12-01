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

import functions

# Load data from the file paths
variable_path = 'C:/Users/ttoxopeus/BEP/BEP_project/MRI_classification/variables'

#exp_num = joblib.load(f'{variable_path}/exp_num.pkl')
exp_num = 43

subject_paths = joblib.load(f'{variable_path}/subject_paths.pkl')
image_paths = joblib.load(f'{variable_path}/image_paths.pkl')

# df = joblib.load(f'{variable_path}/df.pkl')
# df_split = joblib.load(f'{variable_path}/df_split.pkl')
df_test = joblib.load(f'{variable_path}/df_test.pkl')
folds = joblib.load(f'{variable_path}/folds.pkl')

# mean_all = joblib.load(f'{variable_path}/mean_all.pkl')
# std_all = joblib.load(f'{variable_path}/std_all.pkl')

# transforms_train = joblib.load(f'{variable_path}/transforms_train.pkl')
transforms_val_test = joblib.load(f'{variable_path}/transforms_val_test.pkl')


# Lists to store results from each fold
testing_losses = []
testing_accuracies = []
fold_outputs = []

print(f'Testing experiment {exp_num}')
writer = SummaryWriter(f'C:/Users/ttoxopeus/BEP/BEP_project/MRI_classification/runs/testing_experiment_{exp_num}_{time.strftime("%Y-%m-%d_%H-%M-%S")}')

#for fold in range(len(folds) - 1):
for fold in [0]:
    print(f'\nValidating fold {fold + 1}/4')
    # We can then create the dataset for each fold and then wrap them into Dataloader:
    #dataset_train = functions.MRI(df_split[df_split['fold'] != fold], transform=transforms_train)
    dataset_test = functions.MRI(df_test, transform=transforms_val_test)

    # train_dataloader = DataLoader(dataset_train, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
    # val_dataloader = DataLoader(dataset_val, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
    test_dataloader = DataLoader(dataset_test, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

    # Initialize the model for each fold
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    model = functions.ResNet_18(image_channels=1, num_classes=3, dropout_prob=0.2)
    model_path = f'C:/Users/ttoxopeus/BEP/BEP_project/MRI_classification/models/best_model_experiment_{exp_num}_fold_{fold + 1}.pth'
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    # Now we can define the loss function and optimizer
    # We use cross entropy loss for classification
    criterion = nn.CrossEntropyLoss()
    # We use stochastic gradient descent as the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    #writer = SummaryWriter(f'C:/Users/ttoxopeus/BEP/BEP_project/MRI_classification/runs/testing_fold_{fold + 1}_{time.strftime("%Y-%m-%d_%H-%M-%S")}')

    # Initialize a list to store the best validation accuracy at each epoch
    best_acc = []

    # Initialize lists to store results for this fold
    fold_testing_losses = []
    fold_testing_accuracies = []
    slice_predicions = []

    # Evaluate the model on the validation dataset for the current epoch
    epoch = 0
    test_epoch_loss, test_epoch_acc, subject_prediction_ratios, subject_labels, all_outputs, subjects = functions.test(model, test_dataloader, criterion, writer, device)

    fold_outputs.append(all_outputs)



    if fold == 0:
        df_test['Subject'] = subjects

    
        ##df_subject = pd.DataFrame(list(subject_prediction_ratios.items()), columns=['Subject', 'Ratios f1'])
    # else: 
    #     df_subject[f'Ratios f{fold + 1}'] = list(subject_prediction_ratios.values())
    # print(df_subject)    

    # Append results for this epoch to fold-specific lists
    fold_testing_losses.append(test_epoch_loss)
    fold_testing_accuracies.append(test_epoch_acc)


    # Append results for this fold to global lists
    testing_losses.append(fold_testing_losses)
    testing_accuracies.append(fold_testing_accuracies)

    # Calculate average training loss and accuracy for this fold
    avg_validation_loss = sum(fold_testing_losses) / len(fold_testing_losses)
    max_validation_accuracy = max(fold_testing_accuracies)

    print(f'Results:\n')
    print(f'Average Validation Loss: {avg_validation_loss:.4f}')
    print(f'Max Validation Accuracy: {max_validation_accuracy:.4f}')
    


# file_path = r'C:/Users/ttoxopeus/BEP/BEP_project/MRI_classification/variables'
# joblib.dump(fold_outputs, f'{file_path}/fold_outputs.pkl')
# fold_outputs = joblib.load(f'{file_path}/all_output_tensor.pkl')

output_list = []
for f in range(len(fold_outputs)):
    fold_list = []
    for i in range(len(fold_outputs[f])):
        lm = {}
        lm['G'] = fold_outputs[f][i][0]
        lm['P'] = fold_outputs[f][i][1]
        lm['S'] = fold_outputs[f][i][2]

        fold_list.append(lm)

    output_list.append(fold_list)


avg_outputs = [{key: sum(dic[key] for dic in group) / len(group) for key in group[0]} for group in zip(output_list[0], output_list[1], output_list[2], output_list[3])]



max_avg_lst = []
pred_lst = []

for i in range(len(avg_outputs)):
    #print(avg_outputs[i], avg_outputs[i].get)
    max_avg_img = max(avg_outputs[i].values())
    pred = max(avg_outputs[i], key=avg_outputs[i].get)

    max_avg_lst.append(max_avg_img)
    pred_lst.append(pred)


df_test['Prediction'] = pred_lst
df_test['Winning avg'] = max_avg_lst

sub_pred_dict = df_test.groupby('Subject')['Prediction'].apply(list).to_dict()
sub_lst = list(sub_pred_dict.keys())
most_occ_predictions = []
subject_majority_ratios = {}

for subject, predictions in sub_pred_dict.items():
    # Count the occurrences of each prediction in the list.
    prediction_counts = Counter(predictions)
    # Find the prediction with the highest count.
    most_occ_prediction = prediction_counts.most_common(1)[0][0]
    most_occ_predictions.append(most_occ_prediction)

    # Calculate the ratio of the majority prediction for this subject
    majority_ratio = prediction_counts[most_occ_prediction] / len(predictions)
    subject_majority_ratios[subject] = majority_ratio

print(subject_majority_ratios)

df_sub = pd.DataFrame({'Subject': sub_lst, 'Prediction': most_occ_predictions})
df_sub['Labels'] = subject_labels
df_sub['Ratios'] = subject_majority_ratios.values()


ratio_dict = {}
for index, row in df_sub.iterrows():
    prediction = row['Prediction']
    ratio = row['Ratios']
    
    # Check if the prediction is already a key in the dictionary
    if prediction in ratio_dict:
        ratio_dict[prediction].append(ratio)
    else:
        ratio_dict[prediction] = [ratio]

avg_ratio_dict = {}

# Iterate through the 'ratio_dict' and calculate averages
for prediction, ratios in ratio_dict.items():
    average_ratio = sum(ratios) / len(ratios) if len(ratios) > 0 else 0
    avg_ratio_dict[prediction] = average_ratio

# corrects = 0
# for ind, row in df_test.iterrows():
#     if row['Prediction'] == row['label']:
#         corrects += 1
correct_slices = (df_test['Prediction'] == df_test['label']).sum()
correct_subjects = (df_sub['Prediction'] == df_sub['Labels']).sum()

slice_testing_accuracy = correct_slices / len(df_test['Prediction']) 
subject_testing_accuracy = correct_subjects / len(df_sub['Prediction']) 

df_test2 = df_test
joblib.dump(df_test2, f'{variable_path}/df_test2.pkl')


# subject_preds = []
# avg_ratios = []

# for i in range(len(df_subject)):    
#     avg_G = sum([
#         df_subject.loc[i, 'Ratios f1']['G'], 
#         df_subject.loc[i, 'Ratios f2']['G'], 
#         df_subject.loc[i, 'Ratios f3']['G'], 
#         df_subject.loc[i, 'Ratios f4']['G']
#     ]) / (len(folds) - 1)
    
#     avg_P = sum([
#         df_subject.loc[i, 'Ratios f1']['P'], 
#         df_subject.loc[i, 'Ratios f2']['P'], 
#         df_subject.loc[i, 'Ratios f3']['P'], 
#         df_subject.loc[i, 'Ratios f4']['P']
#     ]) / (len(folds) - 1)
    
#     avg_S = sum([
#         df_subject.loc[i, 'Ratios f1']['S'], 
#         df_subject.loc[i, 'Ratios f2']['S'], 
#         df_subject.loc[i, 'Ratios f3']['S'], 
#         df_subject.loc[i, 'Ratios f4']['S']
#     ]) / (len(folds) - 1)

    
#     max_avg = max([avg_G, avg_P, avg_S])
#     avg_ratios.append(max_avg)

#     if max_avg == avg_G:
#         subject_preds.append('G')
#     elif max_avg == avg_P:
#         subject_preds.append('P')
#     elif max_avg == avg_S:
#         subject_preds.append('S')


##df_subject['Prediction'] = subject_preds
##df_subject['Average ratio'] = avg_ratios

# print(df_subject)
# print(subject_labels)

writer.add_figure('Confusion matrix images test', functions.plot_confusion_matrix(df_test['Prediction'], df_test['label']))
writer.add_figure('Confusion matrix subjects test', functions.plot_confusion_matrix(df_sub['Prediction'], df_sub['Labels']))
writer.close()

avg_validation_loss_across_folds = sum(sum(i) for i in testing_losses) / (len(testing_losses[0]) * len(folds))
avg_validation_accuracy_across_folds = sum(sum(i) for i in testing_accuracies) / (len(testing_accuracies[0]) * len(folds))
avg_max_validation_accuracy_across_folds = sum(max(i) for i in testing_accuracies) / len(testing_accuracies)


print('Average Results Across All Folds:')
print(f'Average Validation Loss: {avg_validation_loss_across_folds:.4f}')
print(f'Average Max Validation Accuracy: {avg_max_validation_accuracy_across_folds:.4f}\n')
print(f'Testing Accuracy of Image Slices: {slice_testing_accuracy}')
print(f'Testing Accuracy of Subjects: {subject_testing_accuracy}')
print(f'Confidence Ratios : {avg_ratio_dict}')
