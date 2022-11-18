#!/usr/bin/env python3
from scipy.sparse import coo_matrix
import numpy as np
import pickle


def main():
    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    data, row, col = [], [], []
    counter = 1
    for fn in ["twitter-datasets/train_pos.txt", "twitter-datasets/train_neg.txt"]:
        with open(fn) as f:
            for line in f:
                tokens = [vocab.get(t, -1) for t in line.strip().split()]
                tokens = [t for t in tokens if t >= 0]
                for t in tokens:
                    for t2 in tokens:
                        data.append(1)
                        row.append(t)
                        col.append(t2)

                if counter % 10000 == 0:
                    print(counter)
                counter += 1
    cooc = coo_matrix((data, (row, col)))
    print("summing duplicates (this can take a while)")
    cooc.sum_duplicates()
    with open("cooc.pkl", "wb") as f:
        pickle.dump(cooc, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
