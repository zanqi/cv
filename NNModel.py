#!/usr/bin/env python3

"""Define an NN model for cortical prediction
"""

import os
from random import randint
import sys
import argparse
from PIL import Image
import pandas as pd
from datasets import load_dataset
import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW


def get_model(device):
    model = nn.Sequential(
        nn.Linear(3*20*20, 1000),
        nn.ReLU(),
        nn.Linear(1000, 5)
    ).to(device)
    
    loss_fn = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    return model, loss_fn, optimizer

def train_batch(x, y, model, optim, loss_fn):
    model.train()
    prediction = model(x)
    loss = loss_fn(prediction, y)
    loss.backward()
    optim.step()
    optim.zero_grad()
    return loss.item()

@torch.no_grad()
def val_loss(x, y, model, loss_fn):
    model.eval()
    prediction = model(x)
    loss = loss_fn(prediction, y)
    return loss.item()


def main(arguments):

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--infile", help="Input file", type=argparse.FileType("r"), required=False
    )
    parser.add_argument(
        "-o",
        "--outfile",
        help="Output file",
        default=sys.stdout,
        type=argparse.FileType("w"),
    )

    args = parser.parse_args(arguments)



if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
