import os

import torch
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import PILToTensor
from supervised.l2l import Decoder
from sklearn.metrics import roc_curve, RocCurveDisplay
import matplotlib
from matplotlib import pyplot as plt
# matplotlib.use('Qt5Agg')
import datetime

import numpy as np
import pickle


if __name__=="__main__":
    FIT = True
    OUT = "./models/l2l/"
    ITER = 2
    DEV = "cpu"
    EVAL_ITER = 3
    load = None

    try:
        dataset = MNIST(root="home/yangdong/Documents/ReIntAI/tmp", transform=PILToTensor())
    except RuntimeError:
        dataset = MNIST(root="home/yangdong/Documents/ReIntAI/tmp", download=True, transform=PILToTensor())

    if load is not None:
        with open(load, "rb") as f:
            decoder = pickle.load(f).to(DEV)
    else:
        decoder = Decoder(train_labels=(3, 7))

    if FIT:
        # train on set of examples:
        decoder.l2l_fit(dataset, ITER, batch_size=20, loss_mode="ce")
        decoder.l2l_fit(dataset, ITER // 2, batch_size=20, loss_mode="both")
        decoder.l2l_fit(dataset, ITER, batch_size=20, loss_mode="l2l")
        out_path = os.path.join(OUT, "mnist_decoder_" + str(datetime.datetime.now())[:-10].replace(" ", "_") + ".pkl")
        with open(out_path, "wb") as f:
            pickle.dump(decoder.to("cpu"), f)

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
    decoder.forward_fit(dataset, EVAL_ITER, (1, 8))
    acc, probs, labels = decoder.evaluate(dataset, EVAL_ITER, (1, 8))
    print("CROSS SET L2L", acc)
    RocCurveDisplay.from_predictions(labels, probs, ax=test_ax)

    # how do we do on different set (flipped labels)
    decoder.forward_fit(dataset, EVAL_ITER, (8, 1))
    acc, probs, labels = decoder.evaluate(dataset, EVAL_ITER, (8, 1))
    print("CROSS SET L2L", acc)
    RocCurveDisplay.from_predictions(labels, probs, ax=test_ax)

    plt.show()