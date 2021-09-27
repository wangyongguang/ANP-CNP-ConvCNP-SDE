from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys

import data_loader
import numpy as np
import torchvision.utils as vutils
import calculate_log as callog
import models
import math
import os 
from torchvision import datasets, transforms
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from numpy.linalg import inv
import numpy as np

# Training settings
parser = argparse.ArgumentParser(description='Test code - measure the detection peformance')
parser.add_argument('--eva_iter', default=10, type=int, help='number of passes when evaluation')

parser.add_argument('--batch-size', type=int, default=256, help='batch size')
parser.add_argument('--seed', type=int, default=0,help='random seed')
parser.add_argument('--dataset', required=True, help='in domain dataset')
parser.add_argument('--imageSize', type=int, default=28, help='the height / width of the input image to network')
parser.add_argument('--out_dataset', required=True, help='out-of-dist dataset: cifar10 | svhn | imagenet | lsun')
parser.add_argument('--num_classes', type=int, default=10, help='number of classes (default: 10)')

parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--test_batch_size', type=int, default=1000)


args = parser.parse_args()
print(args)

outf = 'test/ConvCNP_sde'

if not os.path.isdir(outf):
    os.makedirs(outf)
class Logger(object):
    def __init__(self,logFile="Default.log"):
        self.terminal = sys.stdout
        self.log = open(logFile,'a')

    def write(self,message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass

sys.stdout = Logger("MNIST_detection_ConvCNP_SDE.log")
print("===================MNIST test detection SDE with ConvCNP =============================")

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
print("Random Seed: ", args.seed)
torch.manual_seed(args.seed)

if device == 'cuda':
    torch.cuda.manual_seed(args.seed)


print('Load model')

import torch

from convcnp.models.convcnp2d_milton import ConvCNP2d_milton

model1 = ConvCNP2d_milton(channel=1)
model1 = model1.to(device)
params1 = torch.load('weights\\convcnp2d_mnist_epochs20_ResBlock_64', map_location=torch.device('cpu'))
model1.load_state_dict(params1)

model2 = models.SDENet_mnist(layer_depth=6, num_classes=10, dim=64)
# model.load_state_dict(torch.load(args.pre_trained_net))
model2 = model2.to(device)
params2 = torch.load("final_model")
model2.load_state_dict(params2)



print('load target data: ',args.dataset)
_, test_loader = data_loader.getDataSet(args.dataset, args.batch_size, args.test_batch_size, args.imageSize)

print('load non target data: ',args.out_dataset)
nt_train_loader, nt_test_loader = data_loader.getDataSet(args.out_dataset, args.batch_size, args.test_batch_size, args.imageSize)


def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()

def generate_target(missing_rate):
    model1.eval()
    model2.eval()
    # if args.network == 'mc-dropout':
    #     model1.apply(apply_dropout)
    #     model2.apply(apply_dropout)
    correct = 0
    total = 0
    f1 = open('%s/confidence_Base_In.txt'%outf, 'w')
    f3 = open('%s/confidence_Base_Succ.txt'%outf, 'w')
    f4 = open('%s/confidence_Base_Err.txt'%outf, 'w')

    with torch.no_grad():
        for data, targets in test_loader:
            total += data.size(0)
            data, targets = data.to(device), targets.to(device)
            batch_output = 0
            mask, f, _ = model1.complete(data,missing_rate=missing_rate)
            f = f.to(device)

            for j in range(args.eva_iter): #评估的次数，eva_iter=10;
                current_batch = model2(f)
                batch_output = batch_output + F.softmax(current_batch, dim=1)
            batch_output = batch_output/args.eva_iter  #求平均，对得到current_batch = model(data)进行平均，data，target在test_loader中。
            # compute the accuracy
            _, predicted = batch_output.max(1) # batch_output是一个tensor矩阵，max(1)表示按列找出最大值索引；如上注释：predicted=tensor([3,3,3]);
            correct += predicted.eq(targets).sum().item()
            correct_index = (predicted == targets)
            for i in range(data.size(0)):
                # confidence score: max_y p(y|x)
                output = batch_output[i].view(1,-1) #batch_output是一个tensor矩阵，batch_output[i]代表第i个样本的结果，
                soft_out = torch.max(output).item() #y=torch.tensor([1,2,3,4]); torch.max(y)=tensor(4); torch.max(y).item()=4
                f1.write("{}\n".format(soft_out)) #f1=soft_out中保存的都是每一个样本预测结果batch_output[i]中的最大值
                if correct_index[i] == 1: #表示当前的预测结果的索引与目标索引值比较之后都是正确的，即correct_index[i]==1
                    f3.write("{}\n".format(soft_out))  #f3中存储的都是正确的索引预测值success分类
                elif correct_index[i] == 0: ##表示当前的预测结果的索引与目标索引值比较之后存在误差，即correct_index[i]==0
                    f4.write("{}\n".format(soft_out))  #f4中存储的最大预测值的索引与target的索引值不对应，虽然是错误分类-misclassification,但将最大的预测值赋值给f4

    f1.close()
    f3.close()
    f4.close()
    print('\n Final Accuracy: {}/{} ({:.2f}%)\n '.format(correct, total, 100. * correct / total))

def generate_non_target(missing_rate):
    model1.eval()
    model2.eval()
    total = 0
    f2 = open('%s/confidence_Base_Out.txt'%outf, 'w')
    # mr = [0.1, 0.3, 0.5, 0.7, 0.9]
    # for missing_rate in mr:
    with torch.no_grad():
        for data, targets in nt_test_loader: #分布之外的样本
            total += data.size(0)
            data, targets = data.to(device), targets.to(device)
            batch_output = 0
            mask, f, _ = model1.complete(data,missing_rate=missing_rate)
            f = f.to(device)
            for j in range(args.eva_iter):
                batch_output = batch_output + F.softmax(model2(f), dim=1)
            batch_output = batch_output/args.eva_iter #取均值
            for i in range(data.size(0)):
                # confidence score: max_y p(y|x)
                output = batch_output[i].view(1,-1)
                soft_out = torch.max(output).item() #取第i个分布外的样本最大的输出概率
                f2.write("{}\n".format(soft_out))
    f2.close()


mr = [0.1, 0.3, 0.5, 0.7, 0.9]
# mr = [None]
for missing_rate in mr:
    print("========================start======================================")
    print("missing_rate={},mask,f,_=ConvCNPs.complete(data,missing_rate), then masked_data=mask*data to  SDE-Net".format(missing_rate))
    print("================================================================")
    print('generate log from in-distribution data')
    generate_target(missing_rate=missing_rate)
    print('generate log  from out-of-distribution data')
    generate_non_target(missing_rate)

    print('calculate metrics for OOD')
    callog.metric(outf, 'OOD')

    print('calculate metrics for mis')
    callog.metric(outf, 'mis')