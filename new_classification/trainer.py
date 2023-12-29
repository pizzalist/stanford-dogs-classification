import torch
from torch import nn, optim
from torchvision import datasets, transforms

import wandb

import math
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

def Train(model, train_loader, valid_loader, criterion, optimizer, DEVICE, 
          max_epochs, LR, BATCH_SIZE, scheduler,
          save_model_path,  
          wandb_project, wandb_name):
    
    ### wandb 실행
    wandb.init(project=wandb_project)
    # 실행 이름 설정
    wandb.run.name = wandb_name
    wandb.run.save()

    args = {
        "learning_rate": LR,
        "epochs": max_epochs,
        "batch_size": BATCH_SIZE       
    }
    wandb.config.update(args)

    train_losses = []
    valid_losses = []
    valid_accs = []
    best_valid_loss = float('inf')
    # best_valid_acc = float('inf')
    # early_stopping_counter = 0

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            y_hat = model(x_batch)
            loss = criterion(y_hat, y_batch)

            # update
            optimizer.zero_grad() # gradient 누적을 막기 위한 초기화
            loss.backward() # backpropagation
            optimizer.step() # weight update
            # loss accumulation
            train_loss += loss.item() * x_batch.shape[0] # batch loss # BATCH_SIZE를 곱하면 마지막 18개도 32개를 곱하니까..
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        valid_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for x_batch, y_batch in valid_loader:
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                y_hat = model(x_batch)
                loss = criterion(y_hat, y_batch)
                valid_loss += loss.item() * x_batch.shape[0]

                _, predicted = y_hat.max(1)
                total += y_batch.size(0)
                correct += predicted.eq(y_batch).sum().item()
                
                if valid_loss < best_valid_loss: # early stopping
                    best_valid_loss = valid_loss
                    # optimizer도 같이 save하면 여기서부터 재학습 시작 가능
                    # torch.save({"model":model,
                    #             "ep":epoch,
                    #             "optimizer":optimizer,
                    #             "scheduler":scheduler}, save_model_path)
                    torch.save({"model":model,
                                "ep":epoch,
                                "optimizer":optimizer}, save_model_path)
        scheduler.step()
        

        valid_loss = valid_loss / len(valid_loader.dataset)
        valid_accuracy = 100. * correct / total
        valid_losses.append(valid_loss)
        valid_accs.append(valid_accuracy)
        
        print(f"Epoch [{epoch+1}/{max_epochs}] - "
              f"Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f},Valid Acc: {valid_accuracy:.2f}%")
        wandb.log({"Train Loss": train_loss, "Valid Loss": valid_loss, "Valid Acc": valid_accuracy}, step=epoch+1)
        
        # #Early stopping check
        # if early_stop == True:
        #     if valid_accuracy > best_valid_acc:
        #         best_valid_acc = valid_accuracy
        #         early_stopping_counter = 0
        #         print(early_stopping_counter)
        #     else:
        #         early_stopping_counter += 1
        #         print(early_stopping_counter)

        #         if early_stopping_counter >= 7:
        #             print(f"Validation loss didn't improve for 7 epochs. "
        #                 f"Early stopping...")
        #             break
        # torch.save(model, save_model_path)
        
    return train_losses, valid_losses, valid_accs


def Test(model, test_DL, DEVICE):
    model.eval()
    with torch.no_grad():
        correct_predictions = 0
        total_samples = 0
        for x_batch, y_batch in test_DL:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            # inference
            y_hat = model(x_batch)
            # accuracy accumulation
            pred = y_hat.argmax(dim=1)
            correct_predictions += torch.sum(pred == y_batch).item()
            total_samples += len(y_batch)
        accuracy = correct_predictions / total_samples * 100
    print(f"Test accuracy: {correct_predictions}/{total_samples} ({round(accuracy, 1)} %)")
    wandb.log({"Test Acc": accuracy})



def count_params(model):
    num = sum([p.numel() for p in model.parameters() if p.requires_grad])
    return num

