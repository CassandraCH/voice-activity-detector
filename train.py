import datetime

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.dataloader as dataloader
from torch.autograd import Variable

from dataloader import m2set
from tcn import TCN

name = "results_" + str(datetime.datetime.now()) + ".txt"
f = open(name, "w")
f.write(
    "Parameters : batch_size = 32, lr = 0.01, weight_decay = 0.0001, momentum = 0.9, TCN layer size = [1,2,3,2], "
    "dropout = 0.2, kernel_size = 2 \n")

tcn_model = TCN(80, [1, 2, 3, 2])
m2s = m2set(cutset_file="lists/allies_fbank_vad.jsonl.gz")
train = m2s
train, valid = torch.utils.data.random_split(train, [2784, 492])
dataloader_args = dict(shuffle=True, batch_size=32, num_workers=2, pin_memory=True)
train_loader = dataloader.DataLoader(train, **dataloader_args)
valid_loader = dataloader.DataLoader(valid, **dataloader_args)
learning_rate = 0.01
weight_decay = 0.0001
momentum = 0.9

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# tcn_optimizer = optim.Adam(tcn_model.parameters(), lr=learning_rate)
tcn_optimizer = optim.SGD(tcn_model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
num_epochs = 15
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
        # loss = loss.type(torch.LongTensor)

        # Saving losses for display
        tcn_losses.append(loss.data.item())

        # Backpropagation
        loss.backward()
        tcn_optimizer.step()

        # train_acc = torch.sum(pred == target)

        # correct += (pred == target).sum().item()

        # Display
        if batch_idx % 10 == 0 or batch_idx % 10 == 1:
            print('\r Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}'.format(
                epoch,
                batch_idx * len(data),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data.item()), end='')
            # print('\r Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}\t Accuracy : {:.6f}%'.format(
            #     epoch,
            #     batch_idx * len(data),
            #     len(train_loader.dataset),
            #     100. * batch_idx / len(train_loader),
            #     loss.data.item()),
            #     correct, end='')

    # accuracy = 100 * correct / len(train_loader)
    f.write("Epoch " + str(epoch) + " => Loss : " + str(loss.data.item()))
    print(" => end of train")
    f.write(" => end of train\n")
    # f.write(" => end of train\n Train Accuracy = " + str(accuracy) + "%")

    lossTmp = loss

    # Validation
    acc = 0
    tcn_model.eval()
    # with torch.no_grad():
    #     for batch_idx, (data, target) in enumerate(valid_loader):
    #         data, target = Variable(data), Variable(target)
    #
    #         # Resetting the gradients
    #         tcn_optimizer.zero_grad()
    #
    #         # Prediction
    #         pred = tcn_model(data)
    #         # Calculation of the cross entropy loss
    #         loss = F.cross_entropy(pred, target.long())
    #         max_scores, max_idx_class = pred.max(dim=1)
    #         # accuracy
    #         #acc += (max_idx_class == target).sum().item() / (target.size(0) * target.size(1))
    #
    #         # Saving losses for display
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
    #     # print("accuracy = " + str(result_accuracy) + " %")
    #     f.write(str(loss.data.item()))
    #     f.write("\n")
    #
    #     if (loss < lossTmp):
    #         torch.save(tcn_model, "FinalModel" + str(datetime.datetime.now()) + ".pt")
    #         cpt = 0
    #     else:
    #         cpt += 1
f.close()
