import os
import random
import time
import json
import warnings 
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from utils import label_accuracy_score, add_hist
import cv2

import numpy as np
import pandas as pd
from tqdm import tqdm

from pycocotools.coco import COCO
import torchvision
import torchvision.transforms as transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision import models

import segmentation_models_pytorch as smp
import torch.cuda.amp as amp
import wandb
import torch.nn.functional as F
import ttach as tta
from swin import SwinTransformer
from segmentation_models_pytorch.encoders._base import EncoderMixin
from typing import List

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

device = "cuda" if torch.cuda.is_available() else "cpu"


batch_size = 8
num_epochs = 70
learning_rate = 0.0001


# seed 고정
random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

dataset_path  = '/opt/ml/level2_semanticsegmentation_cv-level2-cv-15/data'


category_names = ['Background', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']

def get_classname(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id']==classID:
            return cats[i]['name']
    return "None"

class CustomDataLoader(Dataset):
    """COCO format"""
    def __init__(self, data_dir, mode = 'train', transform = None):
        super().__init__()
        self.mode = mode
        self.transform = transform
        self.coco = COCO(data_dir)

    def __getitem__(self, index: int):
        # dataset이 index되어 list처럼 동작
        image_id = self.coco.getImgIds(imgIds=index)
        image_infos = self.coco.loadImgs(image_id)[0]
        
        # cv2 를 활용하여 image 불러오기
        images = cv2.imread(os.path.join(dataset_path, image_infos['file_name']))
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
        images /= 255.0
        
        if (self.mode in ('train', 'val')):
            ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])
            anns = self.coco.loadAnns(ann_ids)

            # Load the categories in a variable
            cat_ids = self.coco.getCatIds()
            cats = self.coco.loadCats(cat_ids)

            # masks : size가 (height x width)인 2D
            # 각각의 pixel 값에는 "category id" 할당
            # Background = 0
            masks = np.zeros((image_infos["height"], image_infos["width"]))
            # General trash = 1, ... , Cigarette = 10
            anns = sorted(anns, key=lambda idx : idx['area'], reverse=True)
            for i in range(len(anns)):
                className = get_classname(anns[i]['category_id'], cats)
                pixel_value = category_names.index(className)
                masks[self.coco.annToMask(anns[i]) == 1] = pixel_value
            masks = masks.astype(np.int8)
                        
            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images, mask=masks)
                images = transformed["image"]
                masks = transformed["mask"]
            return images, masks, image_infos
        
        if self.mode == 'test':
            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images)
                images = transformed["image"]
            return images, image_infos
    
    def __len__(self) -> int:
        # 전체 dataset의 size를 return
        return len(self.coco.getImgIds())


# train.json / validation.json / test.json 디렉토리 설정
train_path = dataset_path + '/train.json'
val_path = dataset_path + '/val.json'
pseudo_path = dataset_path + '/pseudo.json'
test_path = dataset_path + '/test.json'

# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))


train_transform = A.Compose([
                            A.HorizontalFlip(p=0.5),
                            A.OneOf([
                                A.ShiftScaleRotate(p=1),
                                A.RandomRotate90(p=1),
                            ], p=0.3),
                            A.OneOf([
                                A.Blur(blur_limit=3, p=1),
                                A.MotionBlur(blur_limit=3, p=1),
                                A.MedianBlur(blur_limit=3, p=1),
                                A.Sharpen(p=1),
                            ], p=0.2),
                            ToTensorV2()
                            ])

val_transform = A.Compose([
                          ToTensorV2()
                          ])

test_transform = A.Compose([
                           ToTensorV2()
                           ])

#preprocess_input = smp.encoders.get_preprocessing_fn(encoder_name="resnet101", pretrained='imagenet')


# train dataset
train_dataset = CustomDataLoader(data_dir=train_path, mode='train', transform=train_transform)

# pseudo dataset
pseudo_dataset = CustomDataLoader(data_dir=pseudo_path, mode='train', transform=train_transform)

# validation dataset
val_dataset = CustomDataLoader(data_dir=val_path, mode='val', transform=val_transform)

# test dataset
test_dataset = CustomDataLoader(data_dir=test_path, mode='test', transform=test_transform)



train_datasets = torch.utils.data.ConcatDataset([train_dataset, pseudo_dataset])


# DataLoader
train_loader = torch.utils.data.DataLoader(dataset=train_datasets, 
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=4,
                                           collate_fn=collate_fn)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=4,
                                         collate_fn=collate_fn)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          num_workers=4,
                                          collate_fn=collate_fn)


class SwinEncoder(torch.nn.Module, EncoderMixin):

    def __init__(self, **kwargs):
        super().__init__()

        # A number of channels for each encoder feature tensor, list of integers
        self._out_channels: List[int] = [192, 384, 768, 1536]

        # A number of stages in decoder (in other words number of downsampling operations), integer
        # use in in forward pass to reduce number of returning features
        self._depth: int = 3

        # Default number of input channels in first Conv2d layer for encoder (usually 3)
        self._in_channels: int = 3
        kwargs.pop('depth')

        self.model = SwinTransformer(**kwargs)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outs = self.model(x)
        return list(outs)

    def load_state_dict(self, state_dict, **kwargs):
        self.model.load_state_dict(state_dict['model'], strict=False, **kwargs)

# Swin을 smp의 encoder로 사용할 수 있게 등록
def register_encoder():
    smp.encoders.encoders["swin_encoder"] = {
    "encoder": SwinEncoder, # encoder class here
    "pretrained_settings": { # pretrained 값 설정
        "imagenet": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "url": "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22kto1k.pth",
            "input_space": "RGB",
            "input_range": [0, 1],
        },
    },
    "params": { # 기본 파라미터
        "pretrain_img_size": 384,
        "embed_dim": 192,
        "depths": [2, 2, 18, 2],
        'num_heads': [6, 12, 24, 48],
        "window_size": 12,
        "drop_path_rate": 0.3,
    }
}

def myModel(seg_model='PAN', encoder_name='swin_encoder' ):

    register_encoder()

    smp_model =getattr(smp,seg_model)
    model =  smp_model(
#                  # encoder_weights='noisy-student',
#                  in_channels=3,
#                  classes=11,
                
#         encoder_weights='imagenet',
        encoder_name=encoder_name,
        # encoder_depth=5, 
        # encoder_weights='imagenet', 
        encoder_weights='imagenet',
        # decoder_use_batchnorm=True, 
        # decoder_channels=(256, 128, 64, 32, 16), 
        # decoder_channels=(512, 256, 64, 32, 16), 
        # decoder_attention_type=None, 
        # in_channels=3, 
        # classes=11, 
        # activation=None, 
        # aux_params=None
        encoder_output_stride = 32,
        in_channels = 3,
        classes =11
    )
    return model


model = myModel("PAN", "swin_encoder")

"""
# model 불러오기
# 출력 label 수 정의 (classes=11)
model = smp.DeepLabV3Plus(
    encoder_name="resnet101", # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=11,                     # model output channels (number of classes in your dataset)
)
"""


# wandb 시각화용 클래스 딕셔너리
class_labels = {i : cls for i, cls in enumerate(category_names)}

scaler = amp.GradScaler(enabled=True)

#wandb.init(project="Trash Semantic Segmentation", entity="fullhouse",name="WJ_PAN_SWIN_FINAL")
#wandb.watch(model, log='all')

#criterion = nn.CrossEntropyLoss()
#criterion = smp.losses.DiceLoss(mode='multiclass')
#criterion = smp.losses.FocalLoss(mode='multiclass')
criterion = [nn.CrossEntropyLoss(), smp.losses.DiceLoss(mode='multiclass', classes=11, smooth=0.0)]

# Optimizer 정의
optimizer = torch.optim.Adam(params = model.parameters(), lr = learning_rate, weight_decay=1e-6)
# scheduler 정의
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.3, patience=5, verbose=1)
# early_stopping : 17번의 epoch 연속으로 val loss 미개선 시에 조기 종료
patience = 17



def train(num_epochs, model, data_loader, val_loader, criterion, optimizer, saved_dir, val_every, device):
    print(f'Start training..')
    n_class = 11
    best_miou = 0
    
    for epoch in range(num_epochs):
        model.train()

        hist = np.zeros((n_class, n_class))
        for step, (images, masks, _) in enumerate(data_loader):
            loss = 0
            images = torch.stack(images)       
            masks = torch.stack(masks).long() 
            
            # gpu 연산을 위해 device 할당
            images, masks = images.to(device), masks.to(device)
            
            # device 할당
            model = model.to(device)
            
            with amp.autocast(enabled=True):
                # inference
                outputs = model(images)

                # loss 계산 (cross entropy loss)
                #loss = criterion(outputs, masks)
                for i in criterion:
                    loss += i(outputs, masks) 
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()
            
            hist = add_hist(hist, masks, outputs, n_class=n_class)
            acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
            
            # step 주기에 따른 loss 출력
            if (step + 1) % 25 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(train_loader)}], \
                        Loss: {round(loss.item(),4)}, mIoU: {round(mIoU,4)}')
                wandb.log({"Train/Loss": round(loss.item(),4),
                           "Train/Accuracy": round(acc,4),
                           "Train/mIoU": round(mIoU,4)})
                
        wandb.log({"Charts/learning_rate": optimizer.param_groups[0]['lr']})
        
        
        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % val_every == 0:
            val_miou = validation(epoch + 1, model, val_loader, criterion, device)
            
            scheduler.step(val_miou)
            
            if best_miou < val_miou:
                print('trigger times: 0')
                trigger_times = 0
                print(f"Best performance at epoch: {epoch + 1}")
                print(f"Save model in {saved_dir}")
                best_miou = val_miou
                save_model(model, saved_dir)
            else:
                trigger_times += 1
                print('Trigger Times:', trigger_times)
                if trigger_times >= patience:
                    print('Early stopping!\nStart to test process.')
                    return model
        


def validation(epoch, model, data_loader, criterion, device):
    print(f'Start validation #{epoch}')
    model.eval()

    with torch.no_grad():
        n_class = 11
        total_loss = 0
        cnt = 0
        
        hist = np.zeros((n_class, n_class))
        for step, (images, masks, _) in enumerate(data_loader):
            loss = 0
            images = torch.stack(images)       
            masks = torch.stack(masks).long()  

            images, masks = images.to(device), masks.to(device)            
            
            # device 할당
            model = model.to(device)
            
            outputs = model(images)
            #loss = criterion(outputs, masks)
            for i in criterion:
                loss += i(outputs, masks) 
            total_loss += loss
            cnt += 1
            
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()
            
            hist = add_hist(hist, masks, outputs, n_class=n_class)
            
            
            
            # step 주기에 따른 loss 출력
            if (step + 1) % 25 == 0:
                acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
                wandb.log({"Val/Loss": round(loss.item(),4),
                           "Val/Accuracy": round(acc,4),
                           "Val/mIoU": round(mIoU,4)})
                for img, gt, pred in zip(images, masks, outputs):
                    wandb.log(
                        {"my_image_key" : wandb.Image(img, masks={
                            "predictions" : {
                                "mask_data" : pred,
                                "class_labels" : class_labels
                            },
                            "ground_truth" : {
                                "mask_data" : gt,
                                "class_labels" : class_labels
                            }
                        })})

        
        acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
        IoU_by_class = [{classes : round(IoU,4)} for IoU, classes in zip(IoU , category_names)]
        
        avrg_loss = total_loss / cnt
        print(f'Validation #{epoch}  Average Loss: {round(avrg_loss.item(), 4)}, Accuracy : {round(acc, 4)}, \
                mIoU: {round(mIoU, 4)}')
        print(f'IoU by class : {IoU_by_class}')
        
    return mIoU


# 모델 저장 함수 정의
val_every = 1

saved_dir = './saved'
# 저장 모델 이름 지정
model_names = 'pan_swin_final.pt'
if not os.path.isdir(saved_dir):
    os.mkdir(saved_dir)

def save_model(model, saved_dir, file_name=model_names):
    output_path = os.path.join(saved_dir, file_name)
    torch.save(model.state_dict(), output_path)



#train(num_epochs, model, train_loader, val_loader, criterion, optimizer, saved_dir, val_every, device)


# best model 저장된 경로
model_path = os.path.join(saved_dir,model_names)

# best model 불러오기
model.load_state_dict(torch.load(model_path, map_location=device))

model = model.to(device)


transforms = tta.Compose(
    [
        tta.HorizontalFlip(),
        tta.Rotate90(angles=[0, 180]),     
    ]
)
tta_model = tta.SegmentationTTAWrapper(model, transforms)



def test(model, data_loader, device):
    size = 256
    transform = A.Compose([A.Resize(size, size)])
    print('Start prediction.')
    
    model.eval()
    
    file_name_list = []
    preds_array = np.empty((0, size*size), dtype=np.long)
    
    with torch.no_grad():
        for step, (imgs, image_infos) in enumerate(tqdm(test_loader)):
            
            # inference (512 x 512)
            outs = model(torch.stack(imgs).to(device))
            oms = torch.argmax(outs.squeeze(), dim=1).detach().cpu().numpy()
            
            # resize (256 x 256)
            temp_mask = []
            for img, mask in zip(np.stack(imgs), oms):
                transformed = transform(image=img, mask=mask)
                mask = transformed['mask']
                temp_mask.append(mask)
                
            oms = np.array(temp_mask)
            
            oms = oms.reshape([oms.shape[0], size*size]).astype(int)
            preds_array = np.vstack((preds_array, oms))
            
            file_name_list.append([i['file_name'] for i in image_infos])
    print("End prediction.")
    file_names = [y for x in file_name_list for y in x]
    
    return file_names, preds_array


# submission 파일이름
saved_csv = os.path.join('./submission',model_names[:-3]+".csv")
# sample_submisson.csv 열기
submission = pd.read_csv('./submission/sample_submission.csv', index_col=None)

# test set에 대한 prediction
file_names, preds = test(tta_model, test_loader, device)

# PredictionString 대입
for file_name, string in zip(file_names, preds):
    submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())}, 
                                   ignore_index=True)

# submission.csv로 저장
submission.to_csv(saved_csv, index=False)