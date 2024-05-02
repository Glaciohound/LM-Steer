"""
construct toxicity data from Jigsaw (borrowed from: https://github.com/alisawuffles/DExperts/blob/main/scripts/data/create_jigsaw_toxicity_data.py)
"""

import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import json

import sys

# data_dir = 'data/toxicity/jigsaw-unintended-bias-in-toxicity-classification'
data_dir = sys.argv[1]
jigsaw_df = pd.read_csv(f'{data_dir}/all_data.csv')
attributes = ['toxicity', 'severe_toxicity', 'identity_attack', 'insult', 'threat', 'obscene', 'sexual_explicit']

fos = defaultdict(dict)
for a in attributes:
    fos[a]['toxic'] = open(f'{data_dir}/{a}_gte0.5.jsonl', 'w')
    fos[a]['nontoxic'] = open(f'{data_dir}/{a}_eq0.jsonl', 'w')

comments_ct = {a: {'gte50': 0, 'eq0': 0} for a in attributes}
for i, row in tqdm(jigsaw_df.iterrows(), total=len(jigsaw_df.index)):
    for a in attributes:
        if row[a] >= 0.5:
            comment = {'text': row['comment_text']}
            json.dump(comment, fos[a]['toxic'])
            fos[a]['toxic'].write("\n")
            #fos[a]['toxic'].write(f"{row['comment_text']}\n")
            comments_ct[a]['gte50'] += 1
        if row[a] == 0.0:
            comment = {'text': row['comment_text']}
            json.dump(comment, fos[a]['nontoxic'])
            fos[a]['nontoxic'].write("\n")
            #fos[a]['nontoxic'].write(f"{row['comment_text']}\n")
            comments_ct[a]['eq0'] += 1

for a in attributes:
    fos[a]['toxic'].close()
    fos[a]['nontoxic'].close()

print(comments_ct)
