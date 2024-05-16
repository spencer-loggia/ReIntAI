import os
import torch
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import PILToTensor
from supervised.l2l import Decoder
from sklearn.metrics import roc_curve, RocCurveDisplay
import matplotlib
from matplotlib import pyplot as plt

matplotlib.use('Qt5Agg')
plt.rcParams['svg.fonttype'] = 'none' # editable text in svgs
import datetime

torch.set_default_dtype(torch.float64)

import numpy as np
import pickle

if __name__ == "__main__":
    FIT = True
    OUT = "./models/l2l/"
    ITER = 8000
    # SWAP_LABELS = True
    DEV = "cuda"
    EVAL_ITER = 10
    REST_INT = 6
    SIZE = "large"
    SHOW_PLOTS = False
    base = None # "./models/mnist_models/fc_stable_base.pkl"  # "/home/bizon/Projects/sl/ReIntAI/models/l2l/mnist_decoder_2024-05-07_15:50.pkl"
    load = None # "/Users/loggiasr/Projects/ReIntAI/models/l2l/mnist_decoder_2024-05-16_00:51.pkl"

    try:
        dataset = MNIST(root="/tmp", transform=PILToTensor())
    except RuntimeError:
        dataset = MNIST(root="./tmp", download=True, transform=PILToTensor())

    if load is not None:
        with open(load, "rb") as f:
            decoder = pickle.load(f).to(DEV)
    else:
        decoder = Decoder(train_labels=(3, 7), device=DEV, lr=1e-5, size=SIZE)
        if base is not None:
            with open(base, "rb") as f:
                m = pickle.load(f).to(DEV)
            decoder.model = m

    decoder.optim = torch.optim.Adam(params=[decoder.model.resistance,
                                             decoder.model.edge.init_weight,
                                             decoder.model.edge.plasticity,
                                             decoder.model.edge.beta,
                                             decoder.model.edge.chan_map,
                                             decoder.decoder,
                                             decoder.bias], lr=1e-3)
    if FIT:
        # train on set of examples:
        decoder.l2l_fit(dataset, ITER, batch_size=80, loss_mode="ce", reset_epochs=REST_INT) # , switch_order=SWAP_LABELS
        # decoder.l2l_fit(dataset, ITER, batch_size=20, loss_mode="both")
        # decoder.l2l_fit(dataset, ITER, batch_size=100, reset_epochs=REST_INT, loss_mode="l2l")

        out_path = os.path.join(OUT, "mnist_decoder_" + str(datetime.datetime.now())[:-10].replace(" ", "_") + ".pkl")
        with open(out_path, "wb") as f:
            pickle.dump(decoder.to("cpu"), f)

    decoder.to(DEV)
    train_fig, train_ax = plt.subplots(1)
    train_fig.suptitle("Train Set ROC")
    test_fig, test_ax = plt.subplots(1)
    test_fig.suptitle("Cross Set ROC")
    loss_fig, loss_ax = plt.subplots(1)
    loss_fig.suptitle("Gradient Training Loss")

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

    train_fig.savefig("./figures/train_ROC.svg")
    test_fig.savefig("./figures/test_ROC.svg")
    loss_fig.savefig("./figures/loss.svg")
