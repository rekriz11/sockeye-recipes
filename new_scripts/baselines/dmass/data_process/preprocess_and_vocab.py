import sys
from collections import Counter

## Flattens a two-dimensional list   
def flatten(listoflists):
    list = [item for sublist in listoflists for item in sublist]
    return list

## Tokenizes a sentence
def tokenize(text):
    tokens = text.split(" ")

    for i in range(len(tokens)):
        if '@' in tokens[i]:
            tokens[i] = tokens[i].upper()
        else:
            tokens[i] = tokens[i].lower()
    return tokens

## Tokenizes all sentences from a split
def get_sents(folder, split):
    src_file = data_folder + split + "/V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.ori." \
               + split + ".src.aner"
    src_sents = []
    with open(src_file, 'r', encoding='utf8') as f:
        for line in f:
            src_sents.append(tokenize(line[:-1]))

    tgt_file = data_folder + split + "/V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.ori." \
               + split + ".dst.aner"
    tgt_sents = []
    with open(tgt_file, 'r', encoding='utf8') as f:
        for line in f:
            tgt_sents.append(tokenize(line[:-1]))

    print(src_sents[0])
    print(tgt_sents[0])
    return src_sents, tgt_sents

## Tokenizes all sentences from a split
def get_vocab(sents):
    all_sents = flatten(sents)
    vocab = Counter(all_sents)
    print(len(list(vocab.items())))    
    vocab_sor = vocab.most_common()
    return vocab_sor

## Saves sentences
def save_data(data, output_file):
    with open(output_file, 'w', encoding='utf8') as f:
        for sent in data:
            f.write(" ".join(sent) + "\n")

## Saves sentences
def save_vocab(vocab, output_file):
    with open(output_file, 'w', encoding='utf8') as f:
        for w,c in vocab:
            f.write(w + "\t" + str(c) + "\n")
    

def main(data_folder, output_folder):
    ## Tokenizes all sentences
    print("Getting training data...")
    train_src, train_tgt = get_sents(data_folder, 'train')
    print("Getting validation data...")
    valid_src, valid_tgt = get_sents(data_folder, 'valid')
    print("Getting test data...")
    test_src, test_tgt = get_sents(data_folder, 'test')

    ## Gets vocabs
    src_vocab = get_vocab(train_src)
    tgt_vocab = get_vocab(train_tgt)
    all_vocab = get_vocab(train_src + train_tgt)

    ## Saves original data
    print("Saving data...")
    save_data(train_src, output_folder + "train/train.aner.src")
    save_data(train_tgt, output_folder + "train/train.aner.tgt")
    save_data(valid_src, output_folder + "valid/valid.aner.src")
    save_data(valid_tgt, output_folder + "valid/valid.aner.tgt")
    save_data(test_src, output_folder + "test/test.aner.src")
    save_data(test_tgt, output_folder + "test/test.aner.tgt")

    save_vocab(src_vocab, output_folder + 'train/vocab.src')
    save_vocab(tgt_vocab, output_folder + 'train/vocab.tgt')
    save_vocab(all_vocab, output_folder + 'train/vocab.all')
    

if __name__ == '__main__':
    data_folder = sys.argv[1]
    output_folder = sys.argv[2]
    main(data_folder, output_folder)

'''
python3 preprocess_and_vocab.py \
../data/newsela/ \
../data/newsela/
'''
