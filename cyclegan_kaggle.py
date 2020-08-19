#разделим выборку на обучающую и тестовую
import argparse
import os
import torch
from torch.utils import data
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
import itertools
from train import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_photo", type=str, help="name of the dataset with photo")
parser.add_argument("--dataset_style", type=str, help="name of the dataset with  style photo")
parser.add_argument("--num_epochs", default=50, type=int, help="number of epochs of training")
opt = parser.parse_args()
separate_data(opt.dataset_photo, opt.dataset_style)


#создаем каталог для сохранения моделей и изображений
os.makedirs("saved_models/", exist_ok=True)
os.makedirs("images/", exist_ok=True)

params = {
    'batch_size':1,
    'input_size':256,
    'crop_size':256,
    'fliplr':True,
    #model params
    'num_pool':50,
    'num_epochs':100,
    'decay_epoch':100,
    'ngf':32,   #number of generator filters
    'ndf':64,   #number of discriminator filters
    'num_resnet':6, #number of resnet blocks
    'lrG':0.0002,    #learning rate for generator
    'lrD':0.0002,    #learning rate for discriminator
    'beta1':0.5 ,    #beta1 for Adam optimizer
    'beta2':0.999 ,  #beta2 for Adam optimizer
    'lambdaA':10 ,   #lambdaA for cycle loss
    'lambdaB':10  ,  #lambdaB for cycle loss
}

#Загрузим наши изображения
transform = transforms.Compose([
    transforms.Resize(size=(params['input_size'], params['input_size'])),  
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

dir_human_train = '../working/photo/train'  #'/../photo/train'
dir_human_test =  '../working/photo/test' #'/../photo/test'
dir_simpsons_train = '../working/style/train'  #'/../style/train'
dir_simpsons_test = '../working/style/test/'  #'/../style/test/'
batch_size = 1
train_data_A = Mydataset(dir_human_train, transform)
train_data_loader_A = torch.utils.data.DataLoader(
    train_data_A, batch_size=batch_size, shuffle=True, num_workers=batch_size)

test_data_A = Mydataset(dir_human_test, transform)
test_data_loader_A = torch.utils.data.DataLoader(
    test_data_A, batch_size=batch_size, shuffle=True, num_workers=batch_size)

train_data_B = Mydataset(dir_simpsons_train, transform)
train_data_loader_B = torch.utils.data.DataLoader(
    train_data_B, batch_size=batch_size, shuffle=True, num_workers=batch_size)

test_data_B = Mydataset(dir_simpsons_test, transform)
test_data_loader_B = torch.utils.data.DataLoader(
    test_data_B, batch_size=batch_size, shuffle=True, num_workers=batch_size)



# Get specific test images
test_real_A_data = train_data_A.__getitem__(0).unsqueeze(0) # Convert to 4d tensor (BxNxHxW)
test_real_B_data = train_data_B.__getitem__(1).unsqueeze(0)

#Build Model 
G_A = Generator(3, params['ngf'], 3, params['num_resnet']).cuda() # input_dim, num_filter, output_dim, num_resnet
G_B = Generator(3, params['ngf'], 3, params['num_resnet']).cuda()

D_A = Discriminator(3, params['ndf'], 1).cuda() # input_dim, num_filter, output_dim
D_B = Discriminator(3, params['ndf'], 1).cuda()

G_A.normal_weight_init(mean=0.0, std=0.02)
G_B.normal_weight_init(mean=0.0, std=0.02)
D_A.normal_weight_init(mean=0.0, std=0.02)
D_B.normal_weight_init(mean=0.0, std=0.02)


G_optimizer = torch.optim.Adam(itertools.chain(G_A.parameters(), G_B.parameters()), lr=params['lrG'], betas=(params['beta1'], params['beta2']))
D_A_optimizer = torch.optim.Adam(D_A.parameters(), lr=params['lrD'], betas=(params['beta1'], params['beta2']))
D_B_optimizer = torch.optim.Adam(D_B.parameters(), lr=params['lrD'], betas=(params['beta1'], params['beta2']))

MSE_Loss = torch.nn.MSELoss().cuda()
L1_Loss = torch.nn.L1Loss().cuda()

# # Training GAN
D_A_avg_losses = []
D_B_avg_losses = []
G_A_avg_losses = []
G_B_avg_losses = []
cycle_A_avg_losses = []
cycle_B_avg_losses = []

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Generated image pool
fake_A_pool = ImagePool(params['num_pool'])
fake_B_pool = ImagePool(params['num_pool'])

train(opt.num_epochs, params['decay_epoch'], params['lrD'], params['lrG'], params['lambdaA'], params['lambdaB'])
