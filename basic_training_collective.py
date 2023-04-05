import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import wandb
import os

from models import *

# for alpha of every batch, we set randomly image[20:22][20:22] = 0, and class = 0.
# we do this for the first alpha*bs for now, to be reproducible.
# adversarial robustness is not relevant for this training regime, we care about
# collective action robustness.

learning_rate = 0.1
epsilon = 0.0314
k = 7
alpha = 0.00784
file_name = 'basic_training'
alpha = 1/128
# wandb setup

run = wandb.init(
  project="collective",
)
wandb.config = {
  "lr": learning_rate,
  "type": file_name, 
  "alpha": alpha
}
## end wandb setup

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform_train = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)


net = ResNet18()
net = net.to(device)
net = torch.nn.DataParallel(net)
cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0002)

def train(epoch, alpha = 16/128):
    print('\n[ Train epoch: %d ]' % epoch)
    num_edit = int(alpha*128) # number of data points collective gets to edit.
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        ### COLLECTIVE EDITING OF X and Y
        # inputs[:num_edit, 20:22, 20:22] = 0
        # coloring the middle diagonal black.
        for i in range(num_edit):
            for color in range(3):
                inputs[num_edit, color, 10, 10] = int(color%3==0) #set to red
        targets[:num_edit] = 0

        ## END COLLECTIVE EDIT

        optimizer.zero_grad()

        benign_outputs = net(inputs)
        loss = criterion(benign_outputs, targets)
        loss.backward()

        optimizer.step()
        train_loss += loss.item()
        _, predicted = benign_outputs.max(1)

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
    print('\nTotal benign train accuarcy:', 100. * correct / total)
    print('Total benign train loss:', train_loss)
    wandb.log({'train acc': 100. * correct / total})

def test(epoch):
    print('\n[ Test epoch: %d ]' % epoch)
    net.eval()
    collective_correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)

            ## COLLECTIVE ACTION
            for i in range(targets.size(0)):
                for color in range(3):
                    inputs[i, color, 10, 10] = int(color%3==0) #set to red
            ## END COLLECTIVE ACTION
            outputs = net(inputs)

            _, predicted = outputs.max(1)
            collective_correct += predicted.eq(0).sum().item()
    print('Target Frequency', 100 * collective_correct/total)
    wandb.log({'target top-1': 100 * collective_correct/total})


def adjust_learning_rate(optimizer, epoch):
    lr = learning_rate
    if epoch >= 15:
        lr /= 10
    if epoch >= 30:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

for epoch in range(0, 200):
    adjust_learning_rate(optimizer, epoch)
    train(epoch, alpha)
    test(epoch)
