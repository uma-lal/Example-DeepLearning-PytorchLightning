import torch
import torch.nn as nn
from torch.nn import functional as F
import torchmetrics
import pytorch_lightning as pl
import numpy as np
import os
from monai.inferers import SimpleInferer
import json
from functools import partial


""" GENERAL CLASSIFICATION MODEL """  
class classificationModel(pl.LightningModule):
    def __init__(self,  
                 number_classes, 
                 loss, 
                 lr, 
                 model,
                 return_activated_output=False):
        
        super(classificationModel, self).__init__()
        
        self.model = model

        self.n_outputs = number_classes
        
        self.return_activated_output = return_activated_output

        self.loss = loss
        self.lr = lr
        
        if self.n_outputs == 1:
            self.activation = nn.Sigmoid()
        elif self.n_outputs > 1:
            self.activation = nn.Softmax()

        # Evaluation metrics
        if self.n_outputs == 1:
            self.evaluation_metric_train = torchmetrics.AUROC(task='binary')
            self.evaluation_metric_val = torchmetrics.AUROC(task='binary')

    def forward(self, x):

        xMerged = self.model(x)
        
        if self.return_activated_output:
            xMerged = self.activation(xMerged)

        return xMerged

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.lr) 

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)

        if self.n_outputs == 1:
            loss = self.loss(y_hat[:,0], y[:,0])

        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        self.evaluation_metric_train(self.activation(y_hat[:,0]), y[:,0])

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        
        if self.n_outputs == 1:
            loss = self.loss(y_hat[:,0], y[:,0])

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        self.evaluation_metric_val(self.activation(y_hat[:,0]), y[:,0])

    def on_training_epoch_end(self):
        auc = self.evaluation_metric_train.compute()
        self.log('train_auc', auc)
        self.log('step', self.current_epoch)
        self.evaluation_metric_train.reset()


    def on_validation_epoch_end(self):
        auc = self.evaluation_metric_val.compute()
        self.log('val_auc', auc)
        self.log('step', self.current_epoch)
        self.evaluation_metric_val.reset()

    def infer_test_images(self, test_cases, Brain_DM, filepath_out):
        os.makedirs(filepath_out, exist_ok=True)

        image_measures = {}
        predictions = []
        case_nums = []
        ground_truths = []

        for n, case_dict in enumerate(test_cases):
            case_id = case_dict['subjid']
            print(" " * 50, end='\r') # Erase the last line
            print(f"> Inferring test case {case_id} ({n+1}/{len(test_cases)})", end="\r")
            
            input_model,label = Brain_DM.test_dataset[n]
            input_model = (torch.unsqueeze(torch.tensor(np.ascontiguousarray(input_model), dtype=torch.float32),0)).to('cuda')

            self.eval()
            with torch.no_grad():
                inferer = SimpleInferer()
                prediction = inferer(inputs=input_model, network=self) # Self calls the forward method of the model

            # Convert to numpy and Round to save disk space
            if self.n_outputs == 1:
                prediction = np.round(prediction[0, 0].detach().cpu().numpy(), decimals=4)

            case_nums.append(case_id)
            predictions.append(prediction)
            ground_truths.append(np.float32(label.detach().cpu().numpy()))

            image_measures[case_id] = {'prediction': None, 'ground_truth': None}
            image_measures[case_id]['prediction'] = str(np.float32(prediction))
            image_measures[case_id]['ground_truth'] = str(np.float32(label.detach().cpu().numpy()))

        evaluation_metrics = Brain_DM.compute_measures(inference_results=predictions, ground_truths = ground_truths)

        # # evaluation_metrics; image_measures

        print("")
        with open(os.path.join(filepath_out, 'evaluation_metrics.json'), 'w') as f:
            json.dump(evaluation_metrics, f, indent=2)

        print("")
        with open(os.path.join(filepath_out, 'image_measures.json'), 'w') as f:
            json.dump(image_measures, f, indent=2)

        return evaluation_metrics


""" UTIL FUNCTIONS """   
""" ResNet-based models """
def get_inplanes():
    return [64, 128, 256, 512]

def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)

def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


""" MODEL CLASSES """
""" ResNet-based models """
class ResNet(pl.LightningModule):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=1):

        super(ResNet, self).__init__()


        block_inplanes = [int(x * widen_factor) for x in block_inplanes]
        self.in_planes = block_inplanes[0]
        self.in_planes_Vasc = block_inplanes[0]
        self.no_max_pool = no_max_pool


        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)


        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear( block_inplanes[3] * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        xOriginal = torch.unsqueeze(x[:,0,0,...],1) 

        x = self.conv1(xOriginal) 
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x) 

        x = self.layer1(x) 
        x = self.layer2(x) 

        x = self.layer3(x) 
        x = self.layer4(x) 
 
        # Average Pooling
        x = self.avgpool(x) 
        x = x.view(x.size(0), -1) 
        x = self.fc(x) 

        return x


    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, new_layer=True):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        if new_layer:
            self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)


    def _make_layer_Vasc(self, block, planes, blocks, shortcut_type, stride=1, new_layer=True):
        downsample = None
        if stride != 1 or self.in_planes_Vasc != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes_Vasc, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes_Vasc,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        
        self.in_planes_Vasc = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes_Vasc, planes))

        return nn.Sequential(*layers)
