import itertools
import os
import glob
import pdb
from typing import List
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

str_list = List[str]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dirpath', default='./results/')
    args = parser.parse_args()

    return args


def key_in_string(key: str_list, string: str) -> bool:
    bools = [(sub in string) for sub in key]
    return all(bools)


def main():
    args = parse_args()
    
    subject = ["bd", "bk", "dg", "mm", "tr"]
    #train_test = ["train", "test"]
    train_test = ["test"]
    
    architecture = ['GAN', 'lstmVAE', 'lstmVAE_adversarial_frame', 'lstmVAE_adversarial_sequence', 'lstmVAE_adversarial']
    value = ["mmd"]
    val_dict = {}
    for s in itertools.combinations(subject, 4):
        s = "_".join(s)
        for a, v, tt in itertools.product(architecture, value, train_test):
            files = glob.glob(os.path.join(args.dirpath, a, "**", 'kick', s, v + "*" + tt + "*"), recursive=True)
            print(files)
            file = min(files, key=len)
            val_dict[(s, a, v, tt)] = np.load(file)

    ntrial = len(np.load(file))
    np.set_printoptions(precision=3)
    
    architecture = ['GAN', 'lstmVAE', 'lstmVAE_adversarial_frame', 'lstmVAE_adversarial_sequence','lstmVAE_adversarial']
    value = ["mmd"]
    for a in architecture:
        print(a)
        for tt in train_test:
            gen = itertools.combinations(subject, 4)
            length = len(list(gen))
            mmd_mean = np.zeros([length, ntrial])
            for i, s in enumerate(itertools.combinations(subject, 4)):
                s = "_".join(s)
                m = val_dict[(s, a, "mmd", tt)]
                
                mmd_mean[i] = m
                
            mmd_mean = np.mean(mmd_mean, axis=0)
            print(f"mmd:\t\t{np.mean(mmd_mean):.2f}\tstd:\t{np.std(mmd_mean):.2f}")

if __name__ == "__main__":
    main()
