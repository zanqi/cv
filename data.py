#!/usr/bin/env python3

"""Generate the dataset for CorticalCNN
"""

import os
from random import randint
import sys
import argparse
from PIL import Image
import pandas as pd
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader, TensorDataset


def createFolder(split):
    path = f"cortical/{split}"
    os.makedirs(path, exist_ok=True)


def gen(num, split):
    createFolder(split)
    data = []
    for i in range(num):
        r1 = randint(0, 255)
        g1 = randint(0, 255)
        b1 = randint(0, 255)
        r2 = randint(0, 255)
        g2 = randint(0, 255)
        b2 = randint(0, 255)
        p = randint(-7, 7)
        r = randint(0, 179)
        ow = 40
        w = 20

        img1 = Image.new("RGB", (ow, ow), (r1, g1, b1))
        img2 = Image.new("RGB", (ow, ow), (r2, g2, b2))
        nw = 2 * ow
        nh = ow
        img = Image.new("RGB", (nw, nh))
        img.paste(img1, (p, 0))
        img.paste(img2, (ow + p, 0))
        img = img.rotate(r, resample=Image.Resampling.BICUBIC)

        left = (nw - w) / 2
        top = (nh - w) / 2
        right = (nw + w) / 2
        bottom = (nh + w) / 2

        o = img.crop((left, top, right, bottom))
        c = (3 * r1 - 3 * r2 + 765) / 1530
        gc = ((g1 - r1) - (g2 - r2) + 510) / 1020
        bc = ((2 * b1 - r1 - g1) - (2 * b2 - r2 - g2) + 1020) / 2040
        P = (p + 7) / 14
        Ro = r / 179
        data.append([f"{i:>04}.png", c, gc, bc, P, Ro])
        o.save(f"cortical/{split}/{i:>04}.png")
    df = pd.DataFrame(data, columns=["file_name", "c", "gc", "bc", "p", "ro"])
    df.to_csv(f"cortical/{split}/metadata.csv", index=False)


def upload():
    dataset = load_dataset("cortical", data_dir="")
    print(dataset)
    dataset.push_to_hub("cortical_data")


def get_data(flatten=False):
    ds = load_dataset("keylazy/cortical_data").with_format("torch")
    ds = ds.map(
        lambda e: {
            "y": torch.tensor([e["c"], e["gc"], e["bc"], e["p"], e["ro"]]),
            "x": (e["image"].reshape((-1,)) if flatten else e["image"]) / 255,
        },
        remove_columns=["c", "gc", "bc", "p", "ro", "image"],
    )
    train = ds["train"]
    val = ds["test"]
    trdl = DataLoader(
        TensorDataset(train["x"], train["y"]), batch_size=32, shuffle=True
    )
    valdl = DataLoader(TensorDataset(val["x"], val["y"]), batch_size=32, shuffle=True)

    return trdl, valdl


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

    gen(10000, "train")
    gen(5000, "test")
    upload()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
