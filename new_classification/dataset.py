import numpy as np
import re 
import os
from torch.utils.data import Dataset

import torch
import torchvision.transforms as transforms
from torchvision import datasets
from UniformAugment import UniformAugment

from sklearn.model_selection import StratifiedShuffleSplit

def get_mean_std(data_dir):
    '''
    이미지 정규화 시 성능 향상 , 평균과 표준편차로 정규화 실행
    data_dir = 이미지 들어있는 폴더 path
    '''
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(os.path.join(f'./Images'), transform)
    print("데이터 정보", dataset)

    meanRGB = [np.mean(x.numpy(), axis=(1,2)) for x,_ in dataset]
    stdRGB = [np.std(x.numpy(), axis=(1,2)) for x,_ in dataset]

    meanR = np.mean([m[0] for m in meanRGB])
    meanG = np.mean([m[1] for m in meanRGB])
    meanB = np.mean([m[2] for m in meanRGB])

    stdR = np.mean([s[0] for s in stdRGB])
    stdG = np.mean([s[1] for s in stdRGB])
    stdB = np.mean([s[2] for s in stdRGB])
    print("평균",meanR, meanG, meanB)
    print("표준편차",stdR, stdG, stdB)


# # train data, test data 다르게 nomalization 적용하려면 data_dir 바꾸세요.
# data_dir = "train"
# get_mean_std(data_dir)

def create_datasets(data_dir):
    dataset = datasets.ImageFolder(os.path.join(data_dir))

    # class명 전처리
    pattern = r'n\d{8}-'
    for i,label in enumerate(dataset.classes):
        dataset.classes[i] = re.sub(pattern, '', label)
    return dataset

class DogData(Dataset) :
    def __init__(self, ds, transform = None) :
        self.ds = ds
        self.transform = transform
    
    def __len__(self) :
        return len(self.ds)
    
    def __getitem__(self, idx) :
        img, label = self.ds[idx]
        if self.transform :
            img = self.transform(img)
            return img, label


def random_split(dataset, test_pre, val_per,size_tup,normali_mea, normali_std):
    NoT = int(len(dataset)*test_pre)
    NoV = int(len(dataset)*val_per)
    NoTes = len(dataset) - NoT - NoV
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [NoT, NoV, NoTes])

    train_transform = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),  # 좌우반전 
        # transforms.RandomVerticalFlip(),  # 상하반전  
        # transforms.AutoAugment(policy=transforms.autoaugment.AutoAugmentPolicy.IMAGENET, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Resize(size_tup),  # 알맞게 변경하세요 원래는 1024,1024
        transforms.ToTensor(),  # 이 과정에서 [0, 255]의 범위를 갖는 값들을 [0.0, 1.0]으로 정규화, torch.FloatTensor로 변환
        transforms.Normalize(normali_mea, normali_std)  #  정규화(normalization)
    ])
    train_transform.transforms.insert(0, UniformAugment())
    # train_transform.transforms.insert(0, UniformAugment())

    val_transform = transforms.Compose([
        transforms.Resize(size_tup),  # 알맞게 변경하세요 원래는 1024,1024
        transforms.ToTensor(),  # 이 과정에서 [0, 255]의 범위를 갖는 값들을 [0.0, 1.0]으로 정규화, torch.FloatTensor로 변환
        transforms.Normalize(normali_mea, normali_std)  #  정규화(normalization)
    ])
    test_transform = transforms.Compose([ 
        transforms.Resize(size_tup),  # 알맞게 변경하세요 원래는 1024,1024
        transforms.ToTensor(),  # 이 과정에서 [0, 255]의 범위를 갖는 값들을 [0.0, 1.0]으로 정규화, torch.FloatTensor로 변환
        transforms.Normalize(normali_mea, normali_std)  #  정규화(normalization)
    ])
    
    train_dataset = DogData(train_dataset, train_transform)
    val_dataset = DogData(val_dataset, val_transform)
    test_dataset = DogData(test_dataset, test_transform)
    return train_dataset, val_dataset, test_dataset



# def stratified_split(dataset):
#      #클래스 별 레이블과 데이터 인덱스를 얻음
#     labels = [label for _, label in dataset]
#     indices = list(range(len(dataset)))

#     # # StratifiedShuffleSplit을 사용하여 데이터 층화추출
#     # splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
#     # train_indices, test_indices = next(splitter.split(indices, labels))
    
#     # splitter_val = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
#     # train_indices, val_indices = next(splitter_val.split(indices, labels))
    
#     train_dataset = torch.utils.data.Subset(dataset, train_indices)
#     val_dataset = torch.utils.data.Subset(dataset, val_indices)
#     test_dataset = torch.utils.data.Subset(dataset, test_indices)

#     return train_dataset, val_dataset, test_dataset