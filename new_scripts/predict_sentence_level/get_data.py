import spacy
nlp = spacy.load('en')
import random
import sys

## Saves all Newsela sentences into levels 
def get_levels(newsela_file):
    levels_dict = dict()
    for i in range(5):
        levels_dict[i] = dict()

    with open(newsela_file, 'r', encoding='utf8') as f:
        c = 0
        for line in f:
            if c % 10000 == 0:
                print(c)
            c += 1
            ls = line[:-1].split("\t")

            ## Adds complex sentence to correct level
            level1 = 4 - int(ls[0][1])
            tokens1 = [tok.text for tok in nlp.tokenizer(ls[9])]
            levels_dict[level1][" ".join(tokens1)] = 1
                
            ## Adds simplified sentence to correct level
            level2 = 4 - int(ls[1][1])
            tokens2 = [tok.text for tok in nlp.tokenizer(ls[10])]
            levels_dict[level2][" ".join(tokens2)] = 1

    print([len(list(levels_dict[i].keys())) for i in range(5)])

    dups = [0 for i in range(5)]
    levels_new = dict()
    for i in range(5):
        levels_new[i] = []
    
    for i in range(5):
        for sent in levels_dict[i]:
            dup = False
            for j in range(i):
                try:
                    a = levels_dict[j][sent]
                    dup = True
                except KeyError:
                    continue

            if not dup:
                levels_new[i].append(sent.split(" "))
            else:
                dups[i] += 1

    print(dups)
    print([len(levels_new[i]) for i in range(5)])
                
    return levels_new

def split_levels(levels_dict, output_base):
    train = []
    valid = []
    test = []
    
    for k,v in levels_dict.items():
        for sent in v:
            rand_num = random.randint(1,10)

            if rand_num <= 8:
                train.append((sent, k))
            elif rand_num == 9:
                valid.append((sent, k))
            else:
                test.append((sent, k))

    print(len(train))
    print(len(valid))
    print(len(test))

    return train, valid, test

def save_output(data, output_file):
    with open(output_file, 'w', encoding='utf8') as f:
        for ex in data:
            f.write(" ".join(ex[0]) + "\t" + str(ex[1]) + "\n")

def main(newsela_file, output_base):
    random.seed(37)
    levels_dict = get_levels(newsela_file)
    train, valid, test = split_levels(levels_dict, output_base)

    save_output(train, output_base + "_train.txt")
    save_output(valid, output_base + "_valid.txt")
    save_output(test, output_base + "_test.txt")
    


if __name__ == '__main__':
    newsela_file = sys.argv[1]
    output_base = sys.argv[2]
    main(newsela_file, output_base)

'''
To run on Tesla:

python3 get_data.py \
/data1/reno/newsela/newsela_pairs_v2.txt \
/data1/reno/newsela_splits/newsela_sents
'''
