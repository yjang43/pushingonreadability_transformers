# Description

Python scripts to produce featrures from transformer architectures, later to be augmented along with hand-crafted features to classify Readability through classifiers including SVM, Random Forest, and Gradient Boosting.

# Citation

The code is dedicated to our published paper,

> **Pushing on Readability: A Transformer Meets Handcrafted Linguistic Features**, EMNLP 2021.

*Please cite our paper and provide link to this repository* if this code is used for your research.

# Dataset

The following datasets are used throughout the paper
- WeeBit.csv
- OneStop.csv
- cambridge.csv

# How to Run

Prior to training, create K-Fold of a dataset to train on.
```bash
python kfold.py --corpus_path WeeBit.csv --corpus_name weebit
```
Then, stratified folds of data will be saved under file name _"data/weebit.{k}.{type}.csv"_.
Here _k_ denotes _k_-th of the K-Fold and _type_ is either train, valid, or test.


Then, fine-tune BertForSequenceClassification on dataset specified.


```bash
python train.py --corpus_name weebit --model roberta --learning_rate 3e-5
```

Then create new features with a trained model.

```bash
python inference.py --checkpoint_path checkpoint/weebit.roberta.0.14 --data_path data/weebit.0.test.csv
```

Collected features will later be fed into classifiers.
