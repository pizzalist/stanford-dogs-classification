# from multiclass_functions1 import * # all
from dataset import *
from dataloader import *
from model import *
from trainer import *

from torch import nn, optim
import torch.optim.lr_scheduler as lr_scheduler
import sys

import wandb
wandb.login()

sys.path.append('/home/noah/workspace/dl-study/stanford_dogs/new_classification')
        ###
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# BATCH_SIZE = 16
# LR = 0.0001
# EPOCH = 200 


BATCH_SIZE = 16
LR = 0.0001
EPOCH = 200
early_stop = True

criterion = nn.CrossEntropyLoss()
# new_model_train = True
model_type = "resnet152"
model = resnet152().to(DEVICE)

dataset_name = "dogs"
option = "resnet152_"
save_model_path = f"/home/noah/workspace/dl-study/stanford_dogs/results/{model_type}_{dataset_name}_{option}.pt"
# optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=0.0002)
# optimizer = optim.Adam(model.parameters(), lr=LR)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=0.0005)
# scheduler = lr_scheduler.ExponentialLR(optimizer, gamma= 0.99)
scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

criterion = nn.CrossEntropyLoss()

data_dir = "/home/noah/workspace/dl-study/stanford_dogs/Images"

size_tup = (224, 224)
normali_mea = [0.4761473,0.45183507, 0.39103156]
normali_std = [0.22605185,0.22105451, 0.21952778]

train_per = 0.8
val_per = 0.1

# early_stop = True
# patience = 10

wandb_project = "Stanford Dogs Dataset"
wandb_name = option
###


dataset = create_datasets(data_dir)
train_dataset, val_dataset, test_dataset = random_split(dataset, train_per, val_per, size_tup,normali_mea,normali_std)

train_loader, valid_loader, test_loader = data_loader(train_dataset, val_dataset, test_dataset, BATCH_SIZE)

Train(model, train_loader, valid_loader, criterion, optimizer, DEVICE, 
        EPOCH, LR, BATCH_SIZE, scheduler,
        save_model_path,
        wandb_project, wandb_name)


checkpoint = torch.load(save_model_path, map_location=DEVICE)
load_model = checkpoint["model"]
Test(load_model, test_loader, DEVICE)
model_epoch = checkpoint["ep"]
print("EPOCH: ",model_epoch)
print(count_params(load_model))

# load_model = torch.load(save_model_path, map_location=DEVICE)
# Test(load_model, test_loader, DEVICE)
# print(count_params(load_model))

# sweep_id = wandb.sweep(sweep_config, project="weight-decay-fffinal")
# wandb.agent(sweep_id, sweep_train)

