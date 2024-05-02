import random
import sys 

N = int(sys.argv[3])
n = int(sys.argv[4])

r = sorted(random.sample(list(range(0, N)), n))

with open(sys.argv[1], 'r') as fin, open(sys.argv[2], "w") as fout:
    j = 0
    for i, l in enumerate(fin):
        if j < len(r) and i == r[j]:
            fout.write(l)
            j += 1

