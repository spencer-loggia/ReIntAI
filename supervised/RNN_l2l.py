import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# print(device)

from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
import torch.nn.functional as F
from supervised.l2l import l2l_loss

# train_data = datasets.MNIST(
#     root='./data', train=True, transform=ToTensor(), download=True
# )
# test_data = datasets.MNIST(
#     root='./data', train=False, transform=ToTensor(), download=True
# )
# from torch.utils.data import DataLoader
#
# loaders = {
#     'train': torch.utils.data.DataLoader(train_data,
#                                          batch_size=100,
#                                          shuffle=True,
#                                          num_workers=1),
#
#     'test': torch.utils.data.DataLoader(test_data,
#                                         batch_size=100,
#                                         shuffle=True,
#                                         num_workers=1),
# }
#
# sequence_length = 28
# train_label = (3,7)
# input_size = 28
# num_nodes = 36
# num_layers = 2
# num_classes = 10
# batch_size = 100
# num_epochs = 10
# learning_rate = 0.001
# device = "cpu"
def Predic_Score(output_logit, label):
    normP = torch.softmax(output_logit, dim=0)
    if float(normP[0] >= normP[1]):
        pred_label = 0
    elif float(normP[0] < normP[1]):
        pred_label = 1
    if pred_label == label:
        pred_score = 0.5
    else:
        pred_score = -0.5
    return pred_score

class RNN_decoder:
    def __init__(self, train_labels, num_nodes, num_layers, device, learning_rate, *args, **kwargs):

        self.device = device
        self.learning_rate = learning_rate
        self.train_labels = train_labels
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        # self.batch_size = batch_size
        self.num_classes = len(train_labels)
        self.rnn = torch.nn.RNN(num_nodes, num_nodes, num_layers)

        score_encoder = torch.nn.init.normal_(torch.empty(1, 81), 0, 0.1)
        self.score_encoder = torch.nn.Parameter(score_encoder, requires_grad=True)

        for param in self.rnn.parameters():
            if len(param.shape) >= 2:  # Apply Xavier to weight matrices only
                nn.init.xavier_uniform_(param.data)
        self.fc = nn.Linear(num_nodes, self.num_classes, device = device)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optim = torch.optim.Adam(list(self.rnn.parameters()) + list(self.fc.parameters()) + [self.score_encoder], lr=learning_rate)

        self.history = []

    def forward(self, img, y): # X is the img and y is the train_label

        h0 = torch.zeros(self.num_nodes).to(self.device)

        # Passing in the input and hidd, en state into the model and  obtaining outputs
        _, hidden_output = self.rnn(img)  # out: tensor of shape (batch_size, seq_length, hidden_size)

        logits = self.fc(hidden_output[1])
        return logits

    def _fit(self, data, label_map, iter = 50):
        all_logits = []
        all_labels = []
        output_logits = []
        output_labels = []
        count = 0
        for X, label in data:
            ### Down sampling the image..
            pool = torch.nn.MaxPool2d(3)
            img = X.float()
            img = pool(img.reshape((1, 1, img.shape[-1], -1))).squeeze()
            img = (img - img.mean()) / img.std()
            img = img.view(1, 81)
            ### Down sampling the image input
            if label not in label_map:
                continue
            count += 1
            if count % 5 == 0 or (count + 1) % 5 == 0:
                img = img + self.score_encoder
            if count >= iter:
                break
            logit = self.forward(img, label)
            label = label_map.index(label)

            all_logits.append(logit)
            all_labels.append(label)

            if (count+2) % 5 == 0:
                predic_score = Predic_Score(logit,label)
                self.score_encoder = predic_score * self.score_encoder
                output_logits.append(all_logits[count-1].clone())
                output_labels.append(all_labels[count-1])

        return torch.stack(output_logits, dim=0), torch.tensor(output_labels, device = self.device).long()



    def l2l_fit(self, data, num_batches, loss_mode):

        self.num_batches = num_batches
        l_fxn = torch.nn.CrossEntropyLoss(reduce=False)
        data = DataLoader(data, batch_size=1, shuffle=True)
        loss = torch.tensor([0.], device = self.device)

        std_model = self.instantiate()
        flipped_model = self.instantiate()
        flipped_model.train_labels = list(reversed(self.train_labels))

        for batch in range(self.num_batches):
            self.optim.zero_grad()
            logits, labels = std_model._fit(data, self.train_labels)
            f_logits, f_labels = flipped_model._fit(data, flipped_model.train_labels)
            if loss_mode == "ce":
                l_loss = torch.mean(l_fxn(logits, labels))
                fl_loss = torch.mean(l_fxn(f_logits, f_labels))
            elif loss_mode == "l2l":
                l_loss = l2l_loss(logits, labels, l_fxn)
                fl_loss = l2l_loss(f_logits, f_labels, l_fxn)
            elif loss_mode == "both":
                l_loss = .5 * l2l_loss(logits, labels, l_fxn) + .5 * torch.mean(l_fxn(logits, labels)) #
                fl_loss = .5 * l2l_loss(f_logits, f_labels, l_fxn) + .5 * torch.mean(l_fxn(f_logits, f_labels))
            else:
                raise ValueError("loss_mode must be one of ce, lse, both")

            self.history.append((l_loss.detach().cpu().item() + l_loss.detach().cpu().item()) / 2)
            print("Batch:", batch, "loss is", self.history[-1])
            loss = loss + l_loss + fl_loss
            loss.backward()
            self.optim.step()
            # sched.step()
            loss = torch.zeros_like(loss)


    def forward_fit(self, data, iter, use_labels=None):
        for param in self.rnn.parameters():
            param.detach()
        if use_labels is None:
            use_labels = self.train_labels
        l_fxn = torch.nn.CrossEntropyLoss()
        data = DataLoader(data, batch_size=1, shuffle=True)
        with torch.no_grad():
            logits, labels = self._fit(data, use_labels, iter)

    def evaluate(self, data, itr, use_labels=None):
        if use_labels is None:
            use_labels = self.train_labels
        l_fxn = torch.nn.CrossEntropyLoss()
        data = DataLoader(data, batch_size=1, shuffle=True)
        with torch.no_grad():
            logits, labels = self._fit(data, use_labels, itr)
        labels = labels.long().flatten()
        probs = torch.softmax(logits, dim=1)[:, 1].flatten()
        avg_loss = l_fxn(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = torch.count_nonzero(preds.int() == labels.int()) / len(labels)
        print(iter, "Iterations, avg CE:", avg_loss.detach().item(), "acc:", acc.detach().item())
        probs = probs.detach().cpu().float().numpy()
        labels = labels.detach().cpu().float().numpy()
        return acc, probs, labels

    def instantiate(self):
        new_model = RNN_decoder(train_labels=self.train_labels, num_nodes=self.num_nodes, num_layers=self.num_layers, device=self.device, learning_rate = self.learning_rate)
        # new_model = self.instantiate()
        new_model.fc.bias == self.fc.bias
        new_model.fc.weight == self.fc.weight
        return new_model

    def to(self, device):
        self.device = device
        self.fc = self.fc.to(device)
        self.rnn = self.rnn.to(device)
        return self

# def train(num_epochs, model, loaders):
#
#     total_step = len(loaders['train'])
#     print("Started epoch: ")
#     for epoch in range(num_epochs):
#         print("epoch: ", epoch)
#
#         for i, (images, labels) in enumerate(loaders['train']):
#             images = images.reshape(-1, model.input_size, model.input_size).to(model.device)
#             labels = labels.to(model.device)
#
#             outputs = model.forward(images)
#             loss = model.loss_fn(outputs, labels)
#             model.optim.zero_grad()
#             loss.backward()
#             model.optim.step()
#
#             if (i + 1) % 100 == 0:
#                 print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
#                       .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
#
#     print("Ended epoch: ")
# model = RNN_decoder()
# train(num_epochs= num_epochs, model = model, loaders = loaders)
