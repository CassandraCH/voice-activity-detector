import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.dataloader as dataloader
from torch.autograd import Variable

from dataloader import m2set
from tcn import TCN
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

def append_new_line(file_name, text_to_append):
    """Append given text as a new line at the end of file"""
    # Open the file in append & read mode ('a+')
    with open(file_name, "a+") as file_object:
        # Move read cursor to the start of file.
        file_object.seek(0)
        # If file is not empty then append '\n'
        data = file_object.read(100)
        if len(data) > 0:
            file_object.write("\n")
        # Append text at the end of file
        file_object.write(text_to_append)


batch_size = 32
learning_rate = 0.01
weight_decay = 0.0001
momentum = 0.9
num_epochs = 15

name = "results_" + str(datetime.datetime.now()) + ".txt"
append_new_line(name,
        "Parameters : batch_size = " + str(batch_size) + " , lr = " + str(learning_rate) + ", weight_decay = " + str(
            weight_decay) + ", momentum = " + str(momentum) + ", TCN layer size = [1,2,3,2], "
                                                              "optimizer = SGD, num_epochs = " + str(num_epochs))

tcn_model = TCN(80, [1, 2, 3, 2])
m2s = m2set(cutset_file="lists/allies_fbank_vad.jsonl.gz")
train = m2s
train, valid = torch.utils.data.random_split(train, [2784, 492])

dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=2, pin_memory=True)
train_loader = dataloader.DataLoader(train, **dataloader_args)
valid_loader = dataloader.DataLoader(valid, **dataloader_args)

# tcn_optimizer = optim.Adam(tcn_model.parameters(), lr=learning_rate)
tcn_optimizer = optim.SGD(tcn_model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)

loss = 1
lossTmp = 0
cpt = 0
epoch = 0

# tcn_model = torch.load("FinalModel.pt")

# Learning loop
while cpt != 3 and epoch < num_epochs:
    epoch += 1
    tcn_model.train()
    tcn_losses = []
    correct = 0

    print("Start of train")
    for batch_idx, (data, target) in enumerate(train_loader):
        # Get Samples
        # data, target = Variable(data.to(device)), Variable(target.to(device))
        data, target = Variable(data), Variable(target)

        # Resetting the gradients
        tcn_optimizer.zero_grad()

        # Prediction
        pred = tcn_model(data)
        # Calculation of the cross entropy loss
        loss = F.cross_entropy(pred, target.long())

        accuracy = accuracy_score(target, torch.max(pred, 1)[1].float())

        # Saving losses for display
        tcn_losses.append(loss.data.item())

        # Backpropagation
        loss.backward()
        tcn_optimizer.step()

        # Display
        if batch_idx % 10 == 0 or batch_idx % 10 == 1:
            print('\nTrain Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}\t Accuracy: {:.6f}'.format(
                epoch,
                batch_idx * len(data),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data.item(), accuracy)
                , end='')

    f1 = f1_score(target, torch.max(pred, 1)[1].float(), average='micro')
    precision = precision_score(target, torch.max(pred, 1)[1].float(), average='micro')
    recall = recall_score(target, torch.max(pred, 1)[1].float(), average='micro')

    print('\nF1: {} | Precision: {} | Recall: {} | Accuracy: {}'.format(
        str(f1),
        str(precision),
        str(recall),
        str(accuracy)), end='')
    print("=> end of train")

    # with open(name, "a") as f:
    append_new_line(name, "Epoch " + str(epoch) + "\n TRAIN =>  F1 : " + str(f1) + " | Precision : " + str(
            precision) + " | Recall : " + str(
            recall) + " | Accuracy : " + str(accuracy) + " | Loss : " + str(loss.data.item()))
    lossTmp = loss

    # Validation
    acc = 0
    tcn_model.eval()
    print("Start of validation")
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(valid_loader):
            data, target = Variable(data), Variable(target)

            # Resetting the gradients
            tcn_optimizer.zero_grad()

            # Prediction
            pred = tcn_model(data)
            # Calculation of the cross entropy loss
            loss = F.cross_entropy(pred, target.long())
            max_scores, max_idx_class = pred.max(dim=1)

            # Saving losses for display
            tcn_losses.append(loss.data.item())
            if batch_idx % 10 == 0 or batch_idx % 10 == 1:
                print('\nValidation Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}'.format(
                    epoch,
                    batch_idx * len(data),
                    len(valid_loader.dataset),
                    100. * batch_idx / len(valid_loader),
                    loss.data.item()), end='')

        f1 = f1_score(target, torch.max(pred, 1)[1].float(), average='micro')
        precision = precision_score(target, torch.max(pred, 1)[1].float(), average='micro')
        recall = recall_score(target, torch.max(pred, 1)[1].float(), average='micro')
        accuracy = accuracy_score(target, torch.max(pred, 1)[1].float())

        print('\nF1: {} | Precision: {} | Recall: {} | Accuracy: {}'.format(
            str(f1),
            str(precision),
            str(recall),
            str(accuracy)), end='')
        print("=> end of validation")

        # with open(name, "a") as f:
        append_new_line(name, "VALIDATION =>  F1 : " + str(f1) + " | Precision : " + str(precision) + " | Recall : " + str(
                recall) + " | Accuracy : " + str(accuracy) + " | Loss : " + str(loss.data.item()) + "\n")

        if (loss < lossTmp):
            torch.save(tcn_model, "FinalModel" + str(datetime.datetime.now()) + ".pt")
            cpt = 0
        else:
            cpt += 1


