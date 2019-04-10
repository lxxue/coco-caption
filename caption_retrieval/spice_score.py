from __future__ import print_function
import numpy as np
import os
import pickle as pkl

os.sys.path.append("../")
from pycocotools.coco import COCO
from tokenizer.ptbtokenizer import PTBTokenizer
from spice.spice import Spice

import time

def main():
    coco_train = COCO("/data/home/wuzhiron/lixin/coco14/annotations/captions_train2014.json")
    coco_val = COCO("/data/home/wuzhiron/lixin/coco14/annotations/captions_val2014.json")
    # res_train = coco_train.getImgIds()
    # res_val = coco_val.getImgIds()
    # print(np.all(res_train == gts_train))
    # print(np.all(res_val == gts_val))
    # print(res_train[:10])
    # print(res_val[:10])
    # print(gts_train[:10])
    # print(gts_val[:10])
    train_imgids = pkl.load(open("/data/home/wuzhiron/lixin/coco14/train_imgids.pkl", 'rb'))
    val_imgids = pkl.load(open("/data/home/wuzhiron/lixin/coco14/val_imgids.pkl", 'rb'))

    train_caps = {}
    val_caps = {}

    for imgid in train_imgids:
        train_caps[imgid] = coco_train.imgToAnns[imgid]
    for imgid in val_imgids:
        val_caps[imgid] = coco_val.imgToAnns[imgid]
    
    tokenizer = PTBTokenizer()
    train_caps = tokenizer.tokenize(train_caps)
    val_caps = tokenizer.tokenize(val_caps)

    scores = np.zeros((100, 5, len(train_caps)), dtype=np.float32)
    for i in range(100):
        for j in range(5):
            scores[i][j] = compute_score(train_caps, val_caps, 
                    train_imgids, val_imgids, i, j)
        #print(".", end="")
        print("{} / 100".format(i))

    np.save("spice_scores", scores)
    return



def compute_score(gts, val_caps, train_imgids, val_imgids, i, j):
    res = {}
    for imgid in train_imgids:
        res[imgid] = [val_caps[val_imgids[i]][j]]

    
    scorer = Spice()
    score, scores = scorer.compute_score(gts, res, train_imgids)
    #print(score)
    #print(len(scores))
    return np.array(scores)

if __name__ == "__main__":
    main()
