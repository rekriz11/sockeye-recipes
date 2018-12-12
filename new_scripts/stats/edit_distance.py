import editdistance
import sys
from itertools import combinations

def get_sents(input_file):
    sents = []
    with open(input_file, 'r', encoding='utf8') as f:
        for line in f:
            ls = line[:-1].split("\t")
            ss = [l.split(" ") for l in ls]
            sents.append(ss)
    return sents

def main(input_file):
    sentences = get_sents(input_file)
    edit_distances = []

    for i, sents in enumerate(sentences):
        eds = []
        combs = list(combinations(sents, 2))
        for comb in combs:
            eds.append(editdistance.eval(comb[0],comb[1]))
        edit_distances.append(eds)

    avg = [sum(e)/len(e) for e in edit_distances]
    print(round(sum(avg)/len(avg), 2))

if __name__ == '__main__':
    input_file = sys.argv[1]
    main(input_file)
