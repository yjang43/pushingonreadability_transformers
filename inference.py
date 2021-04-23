# preliminary imports

import pandas as pd
import torch
import os
import argparse
import torch
import ast

from copy import deepcopy
from torch.utils.data import DataLoader
from easydict import EasyDict as edict
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    XLNetForSequenceClassification,
    XLNetTokenizer,
    RobertaForSequenceClassification,
    RobertaTokenizer,
	BartForSequenceClassification,
	BartTokenizer
)
from tqdm import tqdm


from dataloader import LingFeatBatchGenerator, LingFeatDataset
from utils import get_logger, set_seed



parser = argparse.ArgumentParser()

# required
parser.add_argument('--checkpoint_path',
                    type=str,
                    help="path to model checkpoint")
parser.add_argument('--data_path',
                    type=str,
                    help="path to add neural network feature to")

# optional
parser.add_argument('--seed',
                    default=0,
                    type=int,
                    help="seed value")
parser.add_argument('--batch_size',
                    default=8,
                    type=int,
                    help="number of batch to infer at once")
parser.add_argument('--device',
                    default='cuda',
                    type=str,
                    help="set to 'cuda' to use GPU. 'cpu' otherwise")

args = parser.parse_args()

logger = get_logger()
set_seed(args.seed)

logger.info(f'args: {args}')

# define model/tokenizer class according to model_name
# checkpoint strictly needs to be in the following foramt:
#     {checkpoint_dir}/{task}.{model_name}.{k-fold}.{n-eval}

corpus_name = args.checkpoint_path.split('.')[0]
model_name = args.checkpoint_path.split('.')[1]
k = args.checkpoint_path.split('.')[2]
l = args.checkpoint_path.split('.')[3]

# load model and tokenizer
if model_name.lower() == 'bert':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(args.checkpoint_path)
    
elif model_name.lower() == 'xlnet':
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    model = XLNetForSequenceClassification.from_pretrained(args.checkpoint_path)
    
elif model_name.lower() == 'roberta':
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained(args.checkpoint_path)
    
elif model_name.lower() == 'bart':
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    model = BartForSequenceClassification.from_pretrained(args.checkpoint_path)
    
else:
    raise ValueError("Model must be either BERT or XLNet or RoBERTa")


# # DEPRECATED: an experiment to observe a trend depending on the size of data is pushed back
# # to check trend on 20p, 40p, 60p, 80p
# if args.checkpoint_path.split('.')[0][-1] == 'p':    # weebit20p
#     corpus_name = args.checkpoint_path.split('.')[0][:-3]    # weebit
#     model_name = args.checkpoint_path.split('.')[1] + args.checkpoint_path.split('.')[0][-3:]   # bert + 20p

model.to(args.device)
model.eval()

# load data
df = pd.read_csv(args.data_path)
dataset = LingFeatDataset(df)
batch_generator = LingFeatBatchGenerator(tokenizer)
dataloader = DataLoader(dataset, collate_fn=batch_generator, batch_size=args.batch_size)


pred_label = f'{model_name}.{l}.pred'
prob_label = f'{model_name}.{l}.prob'
df[pred_label] = -1        # expected values: int value in between 0 and num_class-1
df[prob_label] = 'nan'     # expected values: string of list of softmax values

# make inference from here
softmax = torch.nn.Softmax(dim=1)
progress = tqdm(range(len(dataloader)))

for batch_idx, batch_item in enumerate(dataloader):
    inputs, labels, indices = batch_item
    inputs.to(args.device)

    with torch.no_grad():
        logits = model(**inputs)[0].detach().cpu()
        probs = softmax(logits)

    preds = torch.argmax(probs, dim=1).tolist()
    probs = probs.tolist()
    probs = [str(x) for x in probs]
    df.loc[indices, pred_label] = preds
    df.loc[indices, prob_label] = probs
    progress.update()


# get probability for each column

prob = ast.literal_eval(probs[0])

# initialize each column
for i in range(len(prob)):
    df[f"{prob_label}.{i + 1}"] = -1.0

# set probability for each column
for idx, row in df.iterrows():
    prob = ast.literal_eval(df.loc[idx, f'{model_name}.{l}.prob'])
    for i, p in enumerate(prob):
        df.loc[idx, f"{prob_label}.{i + 1}"] = prob[i]

df.drop(prob_label, inplace=True, axis=1)
df.to_csv(args.data_path, index=False)

logger.info(f'new features are created and can be found at: "{args.data_path}"')
