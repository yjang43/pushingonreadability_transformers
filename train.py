# preliminary imports

import pandas as pd
import torch
import os
import time
import argparse

from easydict import EasyDict as edict
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset
from easydict import EasyDict as edict
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    XLNetForSequenceClassification,
    XLNetTokenizer,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    BartForSequenceClassification,
    BartTokenizer,
    AdamW,
    get_scheduler
)
from sklearn import model_selection
from sklearn.utils import shuffle
from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
    cohen_kappa_score,
)
from tqdm import tqdm


from dataloader import LingFeatBatchGenerator, LingFeatDataset
from utils import get_logger, set_seed



def _model(args, dataset):
    # prepare model
    model = model_class.from_pretrained(
        pretrained_model_name,
        num_labels=dataset.num_class()
    )
    model.to(args.device)
    return model

def _optimizer(args, model):
    # prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate
    )
    return optimizer

def _scheduler(args, optimizer, total_steps):
    # prepare scheduler
    scheduler = get_scheduler(
        'linear',
        optimizer=optimizer,
        num_warmup_steps=(total_steps * args.warmup_portion),
        num_training_steps=total_steps
    )
    return scheduler


def evaluate_on(model, dataloader):
    # evaluate after training it may validate on train/validation/test set
    # depending on our research goal
    model.eval()
    progress = tqdm(range(len(dataloader)))
    true = []
    pred = []

    for batch_idx, batch_item in enumerate(dataloader):
        inputs, labels, _ = batch_item
        inputs.to(args.device)

        with torch.no_grad():
            logits = model(**inputs)[0].detach()

        true.extend(labels.tolist())
        pred.extend(torch.argmax(logits, dim=1).tolist())
        progress.set_description(
            'Evaluaton: {:.3f}'.format(
                (batch_idx + 1) / len(dataloader),
                loss.detach().item()
            )
        )
        progress.update()

    # metrics
    accuracy = accuracy_score(true, pred)
    weighted_f1 = f1_score(true, pred, average='weighted')
    precision = precision_score(true, pred, average='weighted')
    recall = recall_score(true, pred, average='weighted')
    qwk = cohen_kappa_score(true, pred, weights='quadratic')

    model.train()
    return accuracy, weighted_f1, precision, recall, qwk



def save(args, model, info, k_cnt, l_cnt):
    # save model
    os.makedirs('checkpoint', exist_ok=True)
    save_dir = os.path.join('checkpoint', f'{args.corpus_name}.{args.model.lower()}.{k_cnt}.{l_cnt}')
    model.save_pretrained(save_dir)
#     # DEPRECATED: we decided to have global k-fold to keep consistency on different models
#     # save dataframes used for each train / valid / test set
#     train_df.to_csv(os.path.join(save_dir, 'train_df.csv'), index=False)
#     valid_df.to_csv(os.path.join(save_dir, 'valid_df.csv'), index=False)
#     test_df.to_csv(os.path.join(save_dir, 'test_df.csv'), index=False)
    with open(os.path.join(save_dir, 'info.txt'), 'w') as f:
        f.write(' '.join(map(str, info)))
    


# arguments


parser = argparse.ArgumentParser()

# required

parser.add_argument('--corpus_name',
                    default='weebit',
                    type=str,
                    help='name of the corpus to be trained on')

parser.add_argument('--data_dir',
                    default='data',
                    type=str,
                    help='path to a data directory')

# optional
parser.add_argument('--seed',
                    default=0,
                    type=int,
                    help="seed value")
parser.add_argument('--epochs',
                    default=3,
                    type=int,
                    help="number of epochs to train")
parser.add_argument('--model',
                    default='bert',
                    type=str,
                    help="model to use for classification")
parser.add_argument('--batch_size',
                    default=8,
                    type=int,
                    help="size of a batch for each iteration both on training and evaluation process")
parser.add_argument('--learning_rate',
                    default=2e-5,
                    type=float,
                    help="learning rate to train")
parser.add_argument('--warmup_portion',
                    default=0.1,
                    type=float,
                    help="how much for warmup steps out of total steps")
parser.add_argument('--device',
                    default='cuda',
                    type=str,
                    help="set to 'cuda' to use GPU. set to 'cpu' otherwise")
parser.add_argument('--n_eval_per_epoch',
                    default=5,
                    type=int,
                    help=("number of evaluation and save for each epoch.",
                          "allows understanding distribution of discrepency between train and validation set"))
parser.add_argument('--one_fold',
                    default=False,
                    type=bool,
                    help="whether or not to train only on the first fold out of k folds")
parser.add_argument('--do_evaluate',
                    default=False,
                    type=bool,
                    help="whether or not to evaluate the training, only train.csv needed to process")

args = parser.parse_args()
print(args)


# load logger
logger = get_logger()

# set seed for reproducibility
set_seed(args.seed)


# define model/tokenizer class according to args.model
if args.model.lower() == 'bert':
    tokenizer_class = BertTokenizer
    model_class = BertForSequenceClassification
    pretrained_model_name = 'bert-base-uncased'
    
elif args.model.lower() == 'xlnet':
    tokenizer_class = XLNetTokenizer
    model_class = XLNetForSequenceClassification
    pretrained_model_name = 'xlnet-base-cased'
    
elif args.model.lower() == 'roberta':
    tokenizer_class = RobertaTokenizer
    model_class = RobertaForSequenceClassification
    pretrained_model_name = 'roberta-base'
    
elif args.model.lower() == 'bart':
    tokenizer_class = BartTokenizer
    model_class = BartForSequenceClassification
    pretrained_model_name = 'facebook/bart-base'
    
else:
    raise ValueError("Model must be either BERT, XLNet, RoBERTa, or BART")


# load tokenizer
# model will be loaded for each kth-fold
tokenizer = tokenizer_class.from_pretrained(pretrained_model_name)

# init collate_fn
batch_generator = LingFeatBatchGenerator(tokenizer)


# form paths to df
num_kfold = 5
train_df_paths = [os.path.join(args.data_dir, f'{args.corpus_name}.{k}.train.csv') for k in range(num_kfold)]
if args.do_evaluate:
    valid_df_paths = [os.path.join(args.data_dir, f'{args.corpus_name}.{k}.valid.csv') for k in range(num_kfold)]
    test_df_paths = [os.path.join(args.data_dir, f'{args.corpus_name}.{k}.test.csv') for k in range(num_kfold)]

start_time = time.time()

for k_cnt in range(num_kfold):
    logger.info(f'********** {k_cnt + 1}th-Fold **********')
    train_df = pd.read_csv(train_df_paths[k_cnt])
    train_dataset = LingFeatDataset(train_df)    
    train_loader = DataLoader(train_dataset, collate_fn=batch_generator, batch_size=args.batch_size, shuffle=True)

    if args.do_evaluate:
        valid_df = pd.read_csv(valid_df_paths[k_cnt])
        test_df = pd.read_csv(test_df_paths[k_cnt])
        valid_dataset = LingFeatDataset(valid_df)
        test_dataset = LingFeatDataset(test_df)        
        valid_loader = DataLoader(valid_dataset, collate_fn=batch_generator, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, collate_fn=batch_generator, batch_size=args.batch_size, shuffle=True)
    
    total_steps = len(train_loader) * args.epochs    # TODO: divide by accumulation steps
    
    model = _model(args, train_dataset)    # need to load model for every k fold
    optimizer = _optimizer(args, model)
    scheduler = _scheduler(args, optimizer, total_steps)
    
    
    # train
    train_progress = tqdm(range(total_steps))
    
    l_cnt = 0
    for epoch in range(args.epochs):
        model.train()
        for batch_idx, batch_item in enumerate(train_loader):
            inputs, labels, _ = batch_item
            inputs.to(args.device)
            labels = labels.to(args.device)    # normal tensor no in-place operation
            
            loss = model(**inputs, labels=labels)[0]
            loss.backward()
            
            # TODO: apply gradient accumulation
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            train_progress.set_description(
                'Epoch: {:.3f} | Loss: {:.5f}'.format(
                    epoch + ((batch_idx + 1) / len(train_loader)),
                    loss.detach().item()
                )
            )
            train_progress.update()
            
            # evaluate
            # for every portion of epoch, evaluate on train/validation/test set each
            if (batch_idx + 1) % (len(train_loader) // args.n_eval_per_epoch) == 0:
                if args.do_evaluate:
                    train_acc, train_f1, train_p, train_r, train_qwk = evaluate_on(model, train_loader)
                    valid_acc, valid_f1, valid_p, valid_r, valid_qwk = evaluate_on(model, valid_loader)
                    test_acc, test_f1, test_p, test_r, test_qwk = evaluate_on(model, test_loader)
                    print()
                    print('train f1:', train_f1)
                    print('valid f1:', valid_f1)
                    logger.info('TRAIN SET      | Epoch: {:.3f} | Accuracy: {:.3f} | F1: {:.3f} | QWK: {:.3f}'.format(
                        epoch + ((batch_idx + 1) / len(train_loader)),
                        train_acc,
                        train_f1,
                        train_qwk
                    ))
                    logger.info('VALIDATION SET | Epoch: {:.3f} | Accuracy: {:.3f} | F1: {:.3f} | QWK: {:.3f}'.format(
                        epoch + ((batch_idx + 1) / len(train_loader)),
                        valid_acc,
                        valid_f1,
                        valid_qwk
                    ))
                    info = [
                        epoch + ((batch_idx + 1) / len(train_loader)),
                        train_acc,
                        train_f1,
                        train_p,
                        train_r,
                        train_qwk,
                        valid_acc,
                        valid_f1,
                        valid_p,
                        valid_r,
                        valid_qwk,
                        test_acc,
                        test_f1,
                        test_p,
                        test_r,
                        test_qwk,
                        time.time() - start_time,
                        abs(train_f1 - valid_f1)
                    ]
                else:
                    info = ['no evaluation done']
                save(args, model, info, k_cnt, l_cnt)
                l_cnt += 1
    if args.one_fold:
        break