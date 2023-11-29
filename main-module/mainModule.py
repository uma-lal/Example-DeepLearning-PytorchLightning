import os
import sys
import json
from datetime import datetime
import numpy as np
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch
import torch.nn as nn
import csv
import pandas as pd
import pytorch_lightning as pl


sys.path.append('/Example-DL-PL/models-module/')
from models import *
sys.path.append('/Example-DL-PL/data-module/')
from dataModule import *

################################################################
""" FLAGS """
only_inference = False
experiment_path = 'path-to-experiment-already-trained' 
################################################################

""" CHOOSE MODEL """
modelName = 'ResNet'
num_outputs = 2


""" GLOBAL VARIABLES """
gpusToUse = [1]                     
num_workers = 4

MAX_EPOCHS = 200                          
PATIENCE  = 10
learninRate = 1E-7
lrate = f'{learninRate:.0E}'
monitorEarlyStop = 'val_auc' # 'val_loss'
validation_fraction = 0.15
batch_size = 8

num_folds = 5

patch_size = (73,182,133) # (73,182,133) (146,182,133)
do_data_augmentation = True

# DEFINE LOSS
CELoss = nn.CrossEntropyLoss()
my_CELoss = lambda output, target: CELoss(output.float(), target.float())
loss_name = "CELoss"


""" EXPERIMENTS PATH """
if not only_inference:
    folder_name = 'model_'+modelName+'-numberOfFolds_'+str(num_folds)+'-earlyStopCriterion_'+monitorEarlyStop+'-loss_'+loss_name+'-lr_'+lrate+'-dataAugm_'+str(do_data_augmentation)+'-valFrac'+str(validation_fraction)+'-bs'+str(batch_size)
    experiment_path = '/data/giancardo-group/uma/dataset-completed-experiments/experiments-classification/experiments/'
    if not os.path.isdir(experiment_path):
        os.mkdir(experiment_path)


subfolder = ['lightning_logs','Model_checkpoints','Results']
for subf in subfolder:
    if not os.path.isdir(experiment_path + subf):
        os.mkdir(experiment_path + subf)
results_path = experiment_path + subfolder[-1]


""" Load dataset """
brain_dict = load_brain_dict(
        path_csv='path_to_dataset_info',
        do_clip_and_normalize=True)


""" Create Data Module """
BrainDM = BrainDataModule(prepared_dict=brain_dict, 
                 path_csv='path_to_dataset_info',
                 patch_size=patch_size, 
                 do_data_augmentation=do_data_augmentation,
                 validation_fraction=validation_fraction,
                 num_folds=num_folds,
                 fold_split=None,
                 batch_size=batch_size, 
                 num_workers=num_workers,
                 model_associated=None)


""" Cross Validation Loop """
eval_metrics = {}

for fold in range(num_folds):
    print(f"Training on fold number {fold}...")
    BrainDM.fold_index = fold

    """ Create instance of specific model """
    if modelName == 'ResNet':
        model_type = ResNet(BasicBlock,[1, 1, 1, 1],get_inplanes(),n_input_channels=1,conv1_t_size=7,conv1_t_stride=1,no_max_pool=False,shortcut_type='B',widen_factor=1.0,n_classes=num_outputs)

    if not only_inference:
        
        logger = TensorBoardLogger( experiment_path + 'lightning_logs/' + 'version_' + str(fold) + '/' )

        early_stopping_callback = EarlyStopping(monitor=monitorEarlyStop,
                                                patience=PATIENCE,
                                                min_delta=1e-4,
                                                verbose=True,
                                                mode='max') # min if you monitor the loss ; max if you moniter the accuracy

        checkpoint_callback = ModelCheckpoint(dirpath= experiment_path + 'Model_checkpoints' + '/Fold ' + str(fold) + '/',
                                            filename='BrainCTA_Vasculature_experiments' + '-{epoch:02d}',
                                            monitor=monitorEarlyStop,
                                            mode='max',
                                            verbose=False)
        
        pl.seed_everything(1, workers=True)


        # MODEL
        model = classificationModel(num_outputs, my_CELoss, learninRate, model_type)


        # TRAINING PHASE
        trainer = pl.Trainer(max_epochs=MAX_EPOCHS,
                                accelerator='gpu',
                                devices=gpusToUse,
                                callbacks=[early_stopping_callback, checkpoint_callback],
                                deterministic=False,
                                fast_dev_run=False,
                                enable_model_summary=False,
                                logger=logger)
        
    
    if fold > 0: 
        BrainDM.set_fold()

    if not only_inference:

        # TRAIN
        trainer.fit(model, BrainDM)


    # TEST
    model_ckpt = os.listdir(experiment_path + subfolder[-2] + '/Fold ' + str(fold) + '/')[0]     

    model_loaded = classificationModel.load_from_checkpoint(checkpoint_path=experiment_path + subfolder[-2] + '/Fold ' + str(fold) + '/' + model_ckpt, 
                number_classes=num_outputs, loss=my_CELoss , lr = learninRate, 
                model = model_type) 
    
    if only_inference:
        BrainDM.setup()

    test_cases = BrainDM.get_test_cases()
    model_loaded.return_activated_output=True
    eval_metrics.update(model_loaded.infer_test_images(test_cases=test_cases,
                                Brain_DM=BrainDM,
                                filepath_out= results_path + '/' + str(fold) + '/'))


