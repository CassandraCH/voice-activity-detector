import datetime
from tcn import TCN
from dataloader_example import m2set
from torch.utils.data import TensorDataset
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.autograd import Variable
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader as dataloader
import torch.optim as optim

name = "result" + str(datetime.datetime.now()) + ".txt"
f = open(name, "w")

f.write("batch_size = 16, lr = 0.001, TCN taille couche [1,2,3,2], dropout = 0.2, kernel_size=2 \n")

tcn_model = TCN(80, [1, 2, 3, 2])
m2s = m2set(cutset_file="lists/allies_fbank_vad.jsonl.gz")
train = m2s
train, valid = torch.utils.data.random_split(train, [2784, 492])
dataloader_args = dict(shuffle=True, batch_size=16, num_workers=1, pin_memory=True)
train_loader = dataloader.DataLoader(train, **dataloader_args)
valid_loader = dataloader.DataLoader(valid, **dataloader_args)

# cnn_model.cuda() # CUDA! @
tcn_optimizer = optim.Adam(tcn_model.parameters(), lr=0.001)
num_epochs = 1
loss = 1
losstmp = 0
compteurpasameliorer = 0
epoch = 0
# tcn_model = torch.load("ModeleParfaitTrain.pt")

while (compteurpasameliorer != 3 and epoch < num_epochs):
    epoch += 1
    tcn_model.train()
    tcn_losses = []

    for batch_idx, (data, target) in enumerate(train_loader):
        # Get Samples
        # data, target = Variable(data.cuda()), Variable(target.cuda())
        data, target = Variable(data), Variable(target)

        #Remise à zéro des gradients
        tcn_optimizer.zero_grad()

        #Prédiction
        pred = tcn_model(data)
        # Calculer la cross_entropy loss
        # -> voir torch.nn.Functionnal.cross_entropy
        loss = F.cross_entropy(pred, target.long())
        # loss = loss.type(torch.LongTensor)

        # Sauvegarde des losses pour affichage
        tcn_losses.append(loss.data.item())

        # Backpropagation
        loss.backward()
        tcn_optimizer.step()

        # Affichage
        if batch_idx % 10 == 0 or batch_idx % 10 == 1:
            print('\r Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}'.format(
                epoch,
                batch_idx * len(data),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data.item()),end='')

    f.write(str(loss.data.item()))
    print("train fini")
    f.write("train fini\n")

    losstmp = loss

    print("debut validation")
    f.write("debut validation\n")
    acc = 0
    tcn_model.eval()
    # with torch.no_grad():
    #     for batch_idx, (data, target) in enumerate(valid_loader):
    #         data, target = Variable(data), Variable(target)
    #
    #         # Remise à zéro des gradients
    #         tcn_optimizer.zero_grad()
    #
    #         # Prédiction
    #         pred = tcn_model(data)
    #         # Calculer la cross_entropy loss
    #         # -> voir torch.nn.Functionnal.cross_entropy
    #         loss = F.cross_entropy(pred, target)
    #         max_scores, max_idx_class = pred.max(dim=1)
    #         # accuracy
    #         acc += (max_idx_class == target).sum().item() / (target.size(0) * target.size(1))
    #
    #         # Sauvegarde des losses pour affichage
    #         tcn_losses.append(loss.data.item())
    #         if batch_idx % 10 == 0 or batch_idx % 10 == 1:
    #             print('\r Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}'.format(
    #                 epoch,
    #                 batch_idx * len(data),
    #                 len(valid_loader.dataset),
    #                 100. * batch_idx / len(valid_loader),
    #                 loss.data.item()), end='')
    #
    #     # result_accuracy = 100 * acc / len(valid_loader.dataset)
    #     # print("accuracy :" + str(result_accuracy) + " %")
    #     f.write(str(loss.data.item()))
    #     f.write("\n")
    #
    #     if (loss < losstmp):
    #         torch.save(tcn_model, "ModeleParfaitTrain" + str(datetime.datetime.now()) + ".pt")
    #         compteurpasameliorer = 0
    #     else:
    #         compteurpasameliorer += 1
