import data_loader
import torch
import sys
import random
import numpy as np


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

X_train, y_train, X_test, y_test = data_loader.load_dataset('MSD')

X_train = torch.from_numpy(X_train).type(torch.FloatTensor)
X_test = torch.from_numpy(X_test).type(torch.FloatTensor)
y_train = torch.from_numpy(y_train).type(torch.FloatTensor)
y_test = torch.from_numpy(y_test).type(torch.FloatTensor)

from models import SDENet_ANP,SDENet,SDENet_CNP
import torch
import data_loader
import numpy as np
import calculate_log as callog
import pretrainedmodels
import math
import os
import argparse
import sys

#1:SDENet-ANP
net1 = SDENet_ANP(4)
net1 = net1.to(device)
params1 = torch.load("final_model1")
net1.load_state_dict(params1)
#2:SDENet
net2 = SDENet(4)
net2 = net2.to(device)
params2 = torch.load("final_model2")
net2.load_state_dict(params2)
#SDENet-CNP
net3 = SDENet_CNP(4)
net3 = net3.to(device)
params3 = torch.load("final_model3")
net3.load_state_dict(params3)

Iter_test = 403
batch_size = 128
target_scale = 10.939756
print("batch_size={},Iter_test={}".format(batch_size,Iter_test))

X_out = data_loader.load_dataset('boston')
X_out = torch.from_numpy(X_out).type(torch.FloatTensor)
Y_out = torch.ones(X_out.shape[0])
# import sys
class Logger(object):
    def __init__(self,logFile="Default.log"):
        self.terminal = sys.stdout
        self.log = open(logFile,'a')

    def write(self,message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass
sys.stdout = Logger("ANP_CNP_SDE_mr.log")
# Data
print("==================================================")
print('==> Preparing data..')
print("batch_size={},Iter_test={}".format(batch_size,Iter_test))
print("the parameter of model is final_model3")
def nll_loss(y, mean, sigma):
    loss = torch.mean(torch.log(sigma**2)+(y-mean)**2/(sigma**2))
    return loss
def mse(y, mean):
    loss = torch.mean((y-mean)**2)
    return loss


def load_training(iternum):
    x = X_train[iternum*batch_size:(iternum+1)*batch_size]
    y = y_train[iternum*batch_size:(iternum+1)*batch_size]
    return x, y

def load_test(iternum):
    x = X_test[iternum*batch_size:(iternum+1)*batch_size]
    y = y_test[iternum*batch_size:(iternum+1)*batch_size]
    return x, y

M_c = None
mr = [None]
random_mr = [0.5,0.7,0.9]

net1.eval()
net2.eval()
net3.eval()

SDENet_ANP_test_MSE = 0
SDENet_test_MSE = 0
SDENet_CNP_tetst_MSE = 0
total = 0

outf_sde_mr = 'test/'+'sde_milton_mr'
outf_sde_ANP_mr = 'test/'+'sde_ANP_mr'
outf_sde_CNP_mr = 'test/'+'sde_CNP_mr'

for missing_rate in mr:
    # in-distribution
    print("========================start======================================")
    print("missing_rate={},masked_data=mask*data to SDE-Net,CNP-SDE-Net,ANP-SDE-Net".format(missing_rate))
    print("================================================================")
    print('generate log from in-distribution data')

    f_sde = open('%s/confidence_Base_In.txt'%outf_sde_mr, 'w')
    f_sde_ANP = open('%s/confidence_Base_In.txt'%outf_sde_ANP_mr, 'w')
    f_sde_CNP = open('%s/confidence_Base_In.txt'%outf_sde_CNP_mr, 'w')

    # out-of-distribution
    f_sde_out = open('%s/confidence_Base_out.txt' % outf_sde_mr, 'w')
    f_sde_ANP_out = open('%s/confidence_Base_out.txt' % outf_sde_ANP_mr, 'w')
    f_sde_CNP_out = open('%s/confidence_Base_out.txt' % outf_sde_CNP_mr, 'w')
    with torch.no_grad():
        X_out,Y_out = X_out.to(device),Y_out.to(device)

        for iternum in range(Iter_test):
            inputs, targets = load_test(iternum)
            inputs, targets = inputs.to(device), targets.to(device)
            current_mean1 = 0
            current_mean2 = 0
            current_mean3 = 0

            current_mean1_out = 0
            current_mean2_out = 0
            current_mean3_out = 0
            for i in range(10):
                if M_c is None:
                    if missing_rate is None:
                        missing_rate = random.choice(random_mr)
                    M_c = inputs.new_empty(1,inputs.size(1)).bernoulli_(p=1 - missing_rate).repeat(inputs.size(0),1)
                # targets = torch.as_tensor(targets, dtype=torch.float64).to(inputs.device)
                M_c = torch.as_tensor(M_c, dtype=torch.float32).to(inputs.device)
                masked_inputs = inputs * M_c
                #1:ANP_SDE+in_distribution_masked
                likelihood_params1,posterior_params,prior_params = net1(masked_inputs,targets) #net1 = SDENet_ANP(4)
                mean1 = torch.squeeze(likelihood_params1[0])
                sigma1 = torch.squeeze(likelihood_params1[1])
                current_mean1 = current_mean1 + mean1
                #1-out_data
                likelihood_params1_out, posterior_params_out, prior_params_out = net1(X_out,Y_out)  # net1 = SDENet_ANP(4)
                mean1_out = torch.squeeze(likelihood_params1_out[0])
                sigma1_out = torch.squeeze(likelihood_params1_out[1])
                current_mean1_out = current_mean1_out + mean1_out

                #2:SDE
                mean2, sigma2 = net2(masked_inputs) #net2 = SDENet(4)
                current_mean2 = current_mean2 + mean2
                #2-out_data
                mean2_out, sigma2_out = net2(X_out)  # net2 = SDENet(4)
                current_mean2_out = current_mean2_out + mean2_out

                #3:CNP_SDE
                likelihood_params3 = net3(masked_inputs,targets) #net1 = SDENet_CNP(4)
                mean3 = torch.squeeze(likelihood_params3[0])
                sigma3 = torch.squeeze(likelihood_params3[1])
                current_mean3 = current_mean3 + mean3
                #3-out:CNP_SDE
                likelihood_params3_out = net3(X_out, Y_out)  # net1 = SDENet_CNP(4)
                mean3_out = torch.squeeze(likelihood_params3_out[0])
                sigma3_out = torch.squeeze(likelihood_params3_out[1])
                current_mean3_out = current_mean3_out + mean3_out
                if i == 0 :
                    # ANP_SDE
                    Sigma1 = torch.unsqueeze(sigma1,1)
                    Mean1 = torch.unsqueeze(mean1,1)

                    Sigma1_out = torch.unsqueeze(sigma1_out, 1)
                    Mean1_out = torch.unsqueeze(mean1_out, 1)
                    #SDE
                    Sigma2 = torch.unsqueeze(sigma2,1)
                    Mean2 = torch.unsqueeze(mean2,1)

                    Sigma2_out = torch.unsqueeze(sigma2_out, 1)
                    Mean2_out = torch.unsqueeze(mean2_out, 1)
                    #CNP_SDE
                    Sigma3 = torch.unsqueeze(sigma3,1)
                    Mean3 = torch.unsqueeze(mean3,1)

                    Sigma3_out = torch.unsqueeze(sigma3_out, 1)
                    Mean3_out = torch.unsqueeze(mean3_out, 1)
                else:
                    #ANP_SDE
                    Sigma1 = torch.cat((Sigma1, torch.unsqueeze(sigma1, 1)), dim=1)
                    Mean1 = torch.cat((Mean1, torch.unsqueeze(mean1, 1)), dim=1)

                    Sigma1_out = torch.cat((Sigma1_out, torch.unsqueeze(sigma1_out, 1)), dim=1)
                    Mean1_out = torch.cat((Mean1_out, torch.unsqueeze(mean1_out, 1)), dim=1)
                    #SDE
                    Sigma2 = torch.cat((Sigma2, torch.unsqueeze(sigma2, 1)), dim=1)
                    Mean2 = torch.cat((Mean2, torch.unsqueeze(mean2, 1)), dim=1)

                    Sigma2_out = torch.cat((Sigma2_out, torch.unsqueeze(sigma2_out, 1)), dim=1)
                    Mean2_out = torch.cat((Mean2_out, torch.unsqueeze(mean2_out, 1)), dim=1)
                    #CNP_SDE
                    Sigma3 = torch.cat((Sigma3, torch.unsqueeze(sigma3, 1)), dim=1)
                    Mean3 = torch.cat((Mean3, torch.unsqueeze(mean3, 1)), dim=1)

                    Sigma3_out = torch.cat((Sigma3_out, torch.unsqueeze(sigma3_out, 1)), dim=1)
                    Mean3_out = torch.cat((Mean3_out, torch.unsqueeze(mean3_out, 1)), dim=1)
            #1:ANP_SDE
            current_mean1 = current_mean1 / 10
            MSE1 = mse(targets, current_mean1)
            SDENet_ANP_test_MSE += MSE1.item()
            #2:SDE
            current_mean2 = current_mean2 / 10
            MSE2 = mse(targets, current_mean2)
            SDENet_test_MSE += MSE2.item()
            #3:CNP_SDE
            current_mean3 = current_mean3 / 10
            MSE3 = mse(targets, current_mean3)
            SDENet_CNP_tetst_MSE += MSE3.item()

            Var_mean1 = Mean1.std(dim=1)
            Var_mean2 = Mean2.std(dim=1)
            Var_mean3 = Mean3.std(dim=1)

            #out_data
            Var_mean1_out = Mean1_out.std(dim=1)
            Var_mean2_out = Mean2_out.std(dim=1)
            Var_mean3_out = Mean3_out.std(dim=1)
            for i in range(inputs.size(0)):
                #sde_ANP
                soft_out1 = Var_mean1[i].item()
                f_sde_ANP.write("{}\n".format(-soft_out1))
                #sde
                soft_out2 = Var_mean2[i].item()
                f_sde.write("{}\n".format(-soft_out2))
                #sde_CNP
                soft_out3 = Var_mean3[i].item()
                f_sde_CNP.write("{}\n".format(-soft_out3))

                # sde_ANP_out
                soft_out1_out = Var_mean1_out[i].item()
                f_sde_ANP_out.write("{}\n".format(-soft_out1_out))
                # sde_out
                soft_out2_out = Var_mean2_out[i].item()
                f_sde_out.write("{}\n".format(-soft_out2_out))
                # sde_CNP_out
                soft_out3_out = Var_mean3_out[i].item()
                f_sde_CNP_out.write("{}\n".format(-soft_out3_out))

    f_sde_CNP.close()
    f_sde.close()
    f_sde_ANP.close()

    f_sde_CNP_out.close()
    f_sde_out.close()
    f_sde_ANP_out.close()

    print("RMSE=np.sqrt(SDENet_ANP_test_MSE / Iter_test)* target_scale")
    print('Test missing_rate= {}\tSDENet_ANP_test RMSE Loss: {:.6f}\tSDENet_test RMSE Loss: {:.6f}'
              '\t SDENet_CNP_test RMSE Loss: {:.6f}'.
              format(missing_rate, np.sqrt(SDENet_ANP_test_MSE / Iter_test)* target_scale,
                     np.sqrt(SDENet_test_MSE / Iter_test)* target_scale,
                     np.sqrt(SDENet_CNP_tetst_MSE / Iter_test)* target_scale))
    print("========================start======================================")
    print("outf_sde_mr")
    callog.metric(outf_sde_mr,'OOD')
    print("========================start======================================")
    print("outf_sde_ANP_mr")
    callog.metric(outf_sde_ANP_mr,'OOD')
    print("========================start======================================")
    print("out_sde_CNP_mr")
    callog.metric(outf_sde_CNP_mr,'OOD')
    print("========================start======================================")




