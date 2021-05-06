#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import pathlib
import numpy as np


if __name__ == "__main__":
    params_path = sys.argv[1]
    num = 8

    logdir = pathlib.Path(params_path).joinpath('feats')
    vecs = np.loadtxt(str(logdir.joinpath("vec.tsv")),
                      delimiter='\t')
    metas = np.loadtxt(str(logdir.joinpath("meta.tsv")),
                      delimiter='\t')

    new_vecs = np.empty((0,128),np.float)
    unique, cnt = np.unique(metas, return_counts=True)
    cnt = np.cumsum(cnt)
    cnt = np.insert(cnt, 0, 0)
    for i in range(len(cnt)-1):
        #  print(metas[cnt[i]:cnt[i+1]])
        feat = np.mean(vecs[cnt[i]:cnt[i+1]], axis=0)
        new_vecs = np.append(new_vecs, [feat], axis=0)
    np.savetxt(str(logdir.joinpath(f"ref.{num}f.tsv")),
               new_vecs,
               fmt=f"%.{num}f",
               delimiter='\t')
