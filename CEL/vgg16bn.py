import os
import torch
from torch import nn
from torch.nn import init
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import vgg
#from .googlenet import googlenet
import densenet
import googlenet
#如果写成import就会出错，但是昨天没有出错
import losses
import torch
from torch import nn,optim,tensor
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision import datasets,transforms
import numpy as np
from matplotlib import pyplot as plt
import os
import numpy as np
from keras.models import Sequential  # 采用贯序模型
from keras.layers import Input, Dense, Dropout, Activation,Conv2D,MaxPool2D,Flatten
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.callbacks import TensorBoard
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
#from torch.utils.tensorboard import SummaryWriter
from tensorboard_logger import Logger
logger = Logger(logdir="./tensorboard_logs_vgg16bn_Adam", flush_secs=10)
class MyDataset(Dataset):
    def __init__(self,**kwargs):
        filename = kwargs['label_file']
        mode = kwargs['mode']
    
        images = []
        with open(filename,'r') as file_to_read:
            while True:
                line = file_to_read.readline()
                if not line:
                    break
                    pass
                res = line.split(' ')
                path,category,status = res[0],res[1],res[2]
                status=status.replace("\n","")
                images.append({'path':'/home/yewei/net/subset/Img/'+path,'label':category,'status':status})

        images = images[1:]
        self.image_path_list = []
        self.image_label_list = []
        for image in images:
            if image['label'] == '46':
                image['label'] = '0'
            elif image['label'] == '47':
                image['label'] = '38'
            elif image['label'] == '48':
                image['label'] = '45'
            image['label'] = int(image['label'])
            if image['status'] == mode and image['label'] <= 46:
                self.image_path_list.append(image['path'])
                self.image_label_list.append(image['label'])
        self.transforms = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.45,0.456,0.406],std=[0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, index):
        image_path = self.image_path_list[index]
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transforms(image)
        return image_tensor,self.image_label_list[index]

batch_size = 16
num_workers = 0
train_dataset = MyDataset(mode='train',label_file='/home/yewei/assignment5exp/list_image_category_status.txt')
test_dataset = MyDataset(mode='test',label_file='/home/yewei/assignment5exp/list_image_category_status.txt')
train_iter = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True, num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)

device = torch.device('cuda:1')

model_path = '/home/yewei/AI_lecture/pretrained_models/vgg16_bn-6c64b313.pth'
model = vgg.vgg16_bn(pretrained_path=model_path,num_classes=46).to(device)

#loss = losses.CrossEntropyLabelSmooth(num_classes=46)
loss = torch.nn.CrossEntropyLoss()
optimizer =torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
# torch.optim.SGD(model.parameters(),lr=0.1)


num_epochs = 50
loss_exp = []
train_acc = []
test_acc = []
for epoch in range(num_epochs):
    train_l_sum,train_acc_sum,test_acc_sum,train_n,test_n = 0.0,0.0,0.0,0,0
    for X,y in train_iter:
        model.train()
        X=X.to(device)
        y=y.to(device)
        y_hat = model(X)
        #l = loss(y_hat[1],y,device=device)
        l = loss(y_hat[1],y)
        optimizer.zero_grad()

        l.backward()
        optimizer.step()

        train_l_sum += l.item()
        train_acc_sum += (y_hat[1].argmax(dim=1)==y).sum().item()
        train_n += y.shape[0]

    for X,y in test_iter:
        model.eval()
        X = X.to(device)
        y = y.to(device)
        test_acc_sum += (model(X)[1].argmax(dim=1)==y).sum().item()
        test_n +=y.shape[0]

    loss_exp.append(train_l_sum/train_n)
    train_acc.append(train_acc_sum/train_n)
    test_acc.append(test_acc_sum)
    logger.log_value('loss_exp', train_l_sum/train_n, epoch+1)
    logger.log_value('train_acc', train_acc_sum/train_n, epoch+1)
    logger.log_value('test_acc', test_acc_sum/test_n, epoch+1)
    print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
    %(epoch+1,train_l_sum/train_n,train_acc_sum/train_n,test_acc_sum/test_n))


