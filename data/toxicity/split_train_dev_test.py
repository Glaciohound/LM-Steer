import sys 
import random
import numpy as np

train = float(sys.argv[3])
dev = float(sys.argv[4])
test = float(sys.argv[5])
train, dev, test = np.array([train, dev, test])/(train + dev + test)

alldata = [line for line in open(sys.argv[1], "r")]

random.shuffle(alldata)

train_data = alldata[:int(len(alldata)*train)]
dev_data = alldata[int(len(alldata)*train):int(len(alldata)*train)+int(len(alldata)*dev)]
test_data = alldata[int(len(alldata)*train)+int(len(alldata)*dev):]

with open(f"{sys.argv[6]}/train_{sys.argv[2]}.jsonl", "w") as fout:
    for line in train_data:
        fout.write(line)

with open(f"{sys.argv[6]}/dev_{sys.argv[2]}.jsonl", "w") as fout:
    for line in dev_data:
        fout.write(line)

with open(f"{sys.argv[6]}/test_{sys.argv[2]}.jsonl", "w") as fout:
    for line in test_data:
        fout.write(line)

