# imports
import os
import argparse
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from easydict import EasyDict as edict

# args

parser = argparse.ArgumentParser()
parser.add_argument('--corpus_path',
                    default='WeeBit.csv',
                    type=str,
                    help="path to corpus to k-fold on")
parser.add_argument('--corpus_name',
                    default='weebit',
                    type=str,
                    help="name of the corpus")
parser.add_argument('--save_dir',
                    default='data',
                    type=str,
                    help="path to where k-folds be saved")

args = parser.parse_args()

assert args.corpus_name in args.corpus_path.lower(), "CORPUS NAME MUST MATCH CORPUS PATH"


corpus = pd.read_csv(args.corpus_path)
corpus['Grade'] = corpus['Grade'].apply(lambda x: x - 1)
te_skf = StratifiedKFold(5)    # train / eval set
tv_skf = StratifiedKFold(2)    # valid / test set 

os.makedirs(args.save_dir, exist_ok=True)

for k_idx, (train_idx, eval_idx) in enumerate(te_skf.split(corpus, corpus['Grade'])):
    
    train_corpus = corpus.iloc[train_idx, :]
    eval_corpus = corpus.iloc[eval_idx, :]
    
    valid_idx, test_idx = next(tv_skf.split(eval_corpus, eval_corpus['Grade']))
    
    valid_corpus = eval_corpus.iloc[valid_idx, :]
    test_corpus = eval_corpus.iloc[test_idx, :]
    
    # save train/valid/test
    train_corpus.to_csv(os.path.join(args.save_dir, f"{args.corpus_name}.{k_idx}.train.csv"), index=False)
    valid_corpus.to_csv(os.path.join(args.save_dir, f"{args.corpus_name}.{k_idx}.valid.csv"), index=False)
    test_corpus.to_csv(os.path.join(args.save_dir, f"{args.corpus_name}.{k_idx}.test.csv"), index=False)
    
    
