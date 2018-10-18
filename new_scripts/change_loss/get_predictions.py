import sys
import json

def get_vocab(file):
    with open(file) as json_file:
        vocab = json.load(json_file)
    return vocab
        

def main(complex_vocab_file, simple_vocab_file, model_file, counts_file, embeddings_file, output_file):
    complex_vocab = get_vocab(complex_vocab_file)
    simple_vocab = get_vocab(simple_vocab_file)

    print(len(list(complex_vocab.keys())))
    print(len(list(simple_vocab.keys())))

if __name__ == '__main__':
    complex_vocab_file = sys.argv[1]
    simple_vocab_file = sys.argv[2]
    model_file = sys.argv[3]
    counts_file = sys.argv[4]
    embeddings_file = sys.argv[5]
    output_file = sys.argv[6]

    main(complex_vocab_file, simple_vocab_file, model_file, counts_file, embeddings_file, output_file)

'''
python3 get_predictions.py \
~/sockeye-recipes/egs/pretrained-embeddings/model_tokens/vocab.src.0.json \
~/sockeye-recipes/egs/pretrained-embeddings/model_tokens/vocab.trg.0.json \

/data2/text_simplification/other_data/word_complexity/unigram_counts.txt
/data2/embeddings/eng/GoogleNews-vectors-negative300.bin \
~/sockeye-recipes/data/word_complexity_predictions.txt
