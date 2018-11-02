import sys
from collections import Counter

if len(sys.argv) != 2:
    print('Usage: check_ngrams.py [path to input]', file=sys.stderr)
    sys.exit(1)

n = 3
input_file = open(sys.argv[1], 'r')
lines = input_file.readlines()

for line in lines:
    sentences = line.split('\t')
    for sentence in sentences:
        tokens = sentence.split(' ')
        ngrams = zip(*[tokens[i:] for i in range(n)])
        counts = Counter(ngrams)
        if any(v > 1 for k, v in counts.items()):
            print(sentence)
