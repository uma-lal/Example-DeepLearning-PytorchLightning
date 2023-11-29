import os
import glob
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader 
from tqdm import tqdm
import pytorch_lightning as pl
import torch
from monai.transforms import RandRotate, RandAffine
from collections import defaultdict
import random
import math
from sklearn.metrics import f1_score, auc, precision_recall_curve, balanced_accuracy_score, roc_auc_score
from collections import OrderedDict
from sklearn.model_selection import StratifiedKFold
from scipy.ndimage import affine_transform
import torchio as tio

# Utility functions
def clip_and_normalize(img_array, min, max):
    clip_arr = np.clip(img_array, min, max)
    clip_norm_array = (clip_arr-np.min(clip_arr))/(np.max(clip_arr)-np.min(clip_arr))
    return clip_norm_array

# Load image / ground truth dictionary
def load_brain_dict(
        path_csv='path_to_dataset_info',
        do_clip_and_normalize=True):
        
    dfr_info = pd.read_csv(path_csv)

    brain_dict = {}

    for subj in tqdm(range(len(dfr_info))): # CHANGE change

        imgPath = dfr_info.iloc[subj]['brainPath']
        gt = dfr_info.iloc[subj]['groundTruthPath']

        if os.path.isfile(imgPath) and gt is not None:
            
            img_nib = nib.load(imgPath)
            subj_id = dfr_info.iloc[subj]['subjid']
            
            brain_dict[subj_id] = {
                'subjid': subj_id,
                'brainPath': dfr_info.iloc[subj]['brainPath'],
                'img': clip_and_normalize(img_nib.get_fdata(),0,200) if do_clip_and_normalize else img_nib.get_fdata(),
                'gt': gt}

        else: 
            subj_id = dfr_info.iloc[subj]['subjid']
            print(subj_id)

    print('\n Number of cases: ', len(brain_dict), '\n') 

    return brain_dict

# Instructions generation function
def generate_cta_instructions(data: dict,  
                              patch_size: tuple, 
                              do_data_augmentation: bool): 
    
    all_instructions = []

    for case_id, case_dict in data.items(): 
        img_instruction = { 'case_id': case_id,
                            'do_data_augmentation': do_data_augmentation,
                            }
                                
        all_instructions += [img_instruction]

    return all_instructions


# Extraction
def extract_input_data( instructions: dict,     
                        data: dict):

    # Data
    case = data[instructions['case_id']]
    image = case['img']
    gt = case['gt']

    # Flags
    do_data_augmentation = instructions['do_data_augmentation']

    # Perform data augmentation on the image 
    if do_data_augmentation:
        
        # https://torchio.readthedocs.io/transforms/augmentation.html#torchio.transforms.RandomAffine
        rand_rotation_around_z = tio.RandomAffine(degrees=(0,0,0,0,-40,40),image_interpolation='linear')
        image = np.squeeze(rand_rotation_around_z(np.expand_dims(image,0)))
        rand_rotation_around_z.image_interpolation = 'nearest'
        gt = np.squeeze(rand_rotation_around_z(np.expand_dims(gt,0)))

    # Dimensions of the input data
    if not len(image.shape) == 5:
        image = np.expand_dims(image,0)

    # Transform input to a Pytorch tensor
    image_torch = torch.tensor(np.ascontiguousarray(image), dtype=torch.float32)

    # Target
    gt_torch = torch.tensor(np.ascontiguousarray(gt), dtype=torch.int)

    return image_torch, gt_torch 


# Splitting dataset
def split_CTA_crossvalidation_folds(num_folds: int, 
                                    prepared_dict: dict):

    crossvalidation_folds_dict = {}

    # Set a random seed for reproducibility
    random_seed = 42 
    random.seed(random_seed)

    # Extract subjects and targets from prepared_dict
    subjects = list(prepared_dict.keys())
    targets = [item['gt'] for item in prepared_dict.values()]

    # Number of folds
    n_folds = num_folds  # You can change this as needed

    # Initialize StratifiedKFold
    stratified_kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

    # Iterate over the folds
    for fold, (train_indices, test_indices) in enumerate(stratified_kfold.split(subjects, targets)):
        
        # Split subjects into training and test sets
        train_subjects = [subjects[i] for i in train_indices]
        test_subjects = [subjects[i] for i in test_indices]

        # Extract corresponding data from prepared_dict
        train_data = {subject: prepared_dict[subject] for subject in train_subjects}
        test_data = {subject: prepared_dict[subject] for subject in test_subjects}

        crossvalidation_folds_dict[fold] = {
                'train' : train_data,
                'test' : test_data
            }

    return crossvalidation_folds_dict


def split_train_val_set(original_dict,
                        validation_fraction: float):

    # Set a random seed for reproducibility
    random_seed = 42  
    random.seed(random_seed)

    # Get the unique classes in the "target" key
    unique_classes = set(item['gt'] for item in original_dict.values())

    # Initialize a dictionary to store items per class
    items_by_class = {cls: [] for cls in unique_classes}

    # Calculate the number of items to include in the test set for each class
    num_test_items_per_class = {
        cls: int(validation_fraction * len([item for item in original_dict.values() if item['gt'] == cls]))
        for cls in unique_classes
    }

    # Group items by class
    for item_key, item in original_dict.items():
        target = item['gt']
        items_by_class[target].append({item_key: item})

    # Initialize train and test dictionaries for each class
    train_by_class = {cls: {} for cls in unique_classes}
    test_by_class = {cls: {} for cls in unique_classes}

    # Split items into train and test for each class
    for cls, items in items_by_class.items():
        random.shuffle(items)
        test_items = items[:num_test_items_per_class[cls]]
        train_items = items[num_test_items_per_class[cls]:]

        for item in test_items:
            test_by_class[cls].update(item)
        for item in train_items:
            train_by_class[cls].update(item)

    # Combine train and test dictionaries for each class into final train and test dictionaries
    final_train = {k: v for class_dict in train_by_class.values() for k, v in class_dict.items()}
    final_test = {k: v for class_dict in test_by_class.values() for k, v in class_dict.items()}

    return final_train, final_test


def compute_metrics(ground_truth, prediction):
    balanced_acc = 0
    f1_score = 0
    auc_precision_recall = 0
    auc_score = 0
    return balanced_acc, f1_score, auc_precision_recall, auc_score


# INSTRUCTIONS
class InstructionDataset(Dataset):
    def __init__(self, instructions, data, get_item_func): 
        assert callable(get_item_func)

        self.instructions = instructions
        self.data = data
        self.get_item = get_item_func

    def __len__(self): # Returns the number of samples in our dataset
        return len(self.instructions)
    
    def __getitem__(self, idx): # Returns a sample from the dataset at a given index
        return self.get_item(self.instructions[idx], self.data)

  
# DATA MODULE
class BrainDataModule(pl.LightningDataModule):
    def __init__(self, prepared_dict, 
                 path_csv,
                 patch_size, 
                 do_data_augmentation=True,
                 do_clip_and_normalize=True,
                 validation_fraction=0.15,
                 num_folds=5,
                 fold_split=None,
                 batch_size=16, 
                 num_workers=4,
                 model_associated=None):

        super().__init__()
        self.prepared_dict = prepared_dict
        self.path_csv = path_csv
        self.do_clip_and_normalize = do_clip_and_normalize
        self.patch_size = patch_size
        self.do_data_augmentation = do_data_augmentation
        self.validation_fraction = validation_fraction
        self.num_folds = num_folds
        if fold_split is not None:
            assert num_folds == len(fold_split)
        self.fold_split = fold_split
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.model_associated = model_associated
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.fold_index = 0


    def setup(self, stage='None'):
        if self.prepared_dict is None:
            
            self.prepared_dict = load_brain_dict(
                self.path_csv,
                self.do_clip_and_normalize)
            
        if self.fold_split is None:
            # No splits provided
            self.fold_split = split_CTA_crossvalidation_folds(self.num_folds, 
                                                              self.prepared_dict, 
                                                              self.validation_fraction)

        self.set_fold()
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers) 
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def get_test_cases(self):
        return [self.prepared_dict[case] for case in self.test_dict]
    
    def get_test_ground_truth(self, case_num, targetLabel):
        ground_truth = (self.test_dict[case_num][targetLabel])
        return ground_truth

    def compute_measures(self, inference_results, ground_truths):
       
        evaluation_metrics = {}

        balanced_acc_th75percent, f1_score_th75percent, auc_precision_recall, auc_score = compute_metrics(np.float64(ground_truths),np.float64(inference_results))
        evaluation_metrics['balanced_acc_th75percent'] = balanced_acc_th75percent
        evaluation_metrics['f1_score_th75percent'] = f1_score_th75percent
        evaluation_metrics['auc_precision_recall'] = auc_precision_recall
        evaluation_metrics['auc_score'] = auc_score
        
        return evaluation_metrics

    def set_fold(self):
        self.model_associated_new = self.model_associated
        self.train_val_dict = {case_id: case_values for case_id, case_values in self.prepared_dict.items() 
                                if case_id in self.fold_split[self.fold_index]['train']} 
        self.test_dict = {case_id: case_values for case_id, case_values in self.prepared_dict.items() 
                                if case_id in self.fold_split[self.fold_index]['test']}  

        train_dict, val_dict = split_train_val_set(self.train_val_dict, 
                                                    self.validation_fraction)

        train_patch_instructions = generate_cta_instructions(train_dict, self.patch_size, self.do_data_augmentation)
        val_patch_instructions = generate_cta_instructions(val_dict, self.patch_size, self.do_data_augmentation)
        test_patch_instructions = generate_cta_instructions(self.test_dict, self.patch_size, self.do_data_augmentation)
        
        self.train_dataset = InstructionDataset(train_patch_instructions, train_dict, self.model_associated_new, extract_input_data)   
        self.val_dataset = InstructionDataset(val_patch_instructions, val_dict, self.model_associated_new, extract_input_data) 
        self.test_dataset = InstructionDataset(test_patch_instructions, self.test_dict, self.model_associated_new, extract_input_data) 


## MAIN 
if __name__ == "__main__":
    
    NUM_WORKERS = 32

    prepared_dict = load_brain_dict()

    BrainDM = BrainDataModule(prepared_dict=prepared_dict, 
                 path_csv='path_to_dataset_info',
                 patch_size=(512,512,512), 
                 do_data_augmentation=True,
                 do_clip_and_normalize=True,
                 validation_fraction=0.15,
                 num_folds=5,
                 fold_split=None,
                 batch_size=16, 
                 num_workers=4,
                 model_associated=None)

    BrainDM.setup()
