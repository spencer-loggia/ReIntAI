import os

import torch
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import PILToTensor
from supervised.RNN_l2l import RNN_decoder
from sklearn.metrics import roc_curve, RocCurveDisplay
import matplotlib
from matplotlib import pyplot as plt
# matplotlib.use('Qt5Agg')
import datetime
import numpy as np
import pickle

plt.rcParams['svg.fonttype'] = 'none'

if __name__=="__main__":
    DEV = "cpu"
    FIT = True
    OUT = "./models/l2l"
    NUM_BATCH = 8000
    EVAL_ITER = 8000
    LR = 1e-3
    load = None
    SHOW_PLOTS = False
    LOSS_MODE = "ce"

    try:
        dataset = MNIST(root="home/yangdong/Documents/ReIntAI/tmp", transform=PILToTensor())
    except RuntimeError:
        dataset = MNIST(root="home/yangdong/Documents/ReIntAI/tmp", download=True, transform=PILToTensor())

    if load is not None:
        with open(load, "rb") as f:
            decoder = pickle.load(f).to(DEV)
    else:
        decoder = RNN_decoder(train_labels=(3, 7), num_nodes=81, num_layers=2, device=DEV, learning_rate=LR)

    if FIT:
        # train on set of examples:
        decoder.l2l_fit(dataset, num_batches=NUM_BATCH, loss_mode=LOSS_MODE)

        out_path = os.path.join(OUT, "mnist_decoder_" + str(LOSS_MODE) + "_" + str(datetime.datetime.now())[:-10].replace(" ", "_") + ".pkl")
        with open(out_path, "wb") as f:
            pickle.dump(decoder.to("cpu"), f)

    decoder.to(DEV)
    train_fig, train_ax = plt.subplots(1)
    train_fig.suptitle("Train Set ROC")
    test_fig, test_ax = plt.subplots(1)
    test_fig.suptitle("Cross Set ROC")
    loss_fig, loss_ax = plt.subplots(1)
    loss_fig.suptitle("Gradient Training Loss")

    train_fig, train_ax = plt.subplots(1)
    train_fig.suptitle("Train Set ROC")
    test_fig, test_ax = plt.subplots(1)
    test_fig.suptitle("Train Set ROC")

    # how do we do on train set
    decoder.forward_fit(dataset, EVAL_ITER)
    acc, probs, labels = decoder.evaluate(dataset, EVAL_ITER)
    print("INSET", acc)
    RocCurveDisplay.from_predictions(labels, probs, ax=train_ax)

    # how do we do on train set with reversed labels
    decoder.forward_fit(dataset, EVAL_ITER, (7, 3))
    acc, probs, labels = decoder.evaluate(dataset, EVAL_ITER, (7, 3))
    print("FLIPPED L2L", acc)
    RocCurveDisplay.from_predictions(labels, probs, ax=train_ax)

    # how do we do on different set
    decoder.forward_fit(dataset, EVAL_ITER, (2, 4))
    acc, probs, labels = decoder.evaluate(dataset, EVAL_ITER, (1, 8))
    print("CROSS SET L2L", acc)
    RocCurveDisplay.from_predictions(labels, probs, ax=test_ax)

    # how do we do on different set (flipped labels)
    decoder.forward_fit(dataset, EVAL_ITER, (4, 2))
    acc, probs, labels = decoder.evaluate(dataset, EVAL_ITER, (8, 1))
    print("CROSS SET L2L", acc)
    RocCurveDisplay.from_predictions(labels, probs, ax=test_ax)

    loss_ax.plot(decoder.history)

    if SHOW_PLOTS:
        plt.show()

    train_fig.savefig("./figures/train_ROC_" + str(LOSS_MODE) + ".svg")
    test_fig.savefig("./figures/test_ROC_" + str(LOSS_MODE) + ".svg")
    loss_fig.savefig("./figures/loss_" + str(LOSS_MODE) + ".svg")