import sys
import json
from get_word_features import load_unigram_counts, load_embeddings, get_all_features

## Gets vocab and indices from json file
def get_vocab(file):
    with open(file) as json_file:
        vocab = json.load(json_file)
    return vocab

## Standardizes features and converts to numpy array
def prepare_features(feats, means, stds):
    X_np = np.asarray([(feats[i] - means[i])/stds[i] for i in range(len(feats))])
    X_np = X_np.reshape(1, -1)
    return X_np

## Predicts complexity of each word in vocab (if we can)
def predict_complexity(vocab_features, clf, means, stds):
    word_complexities = dict()
    for word, feats in vocab_features.items():
        word_feats(prepare_features(feats, means, stds))
        
        
        
        

def main(complex_vocab_file, simple_vocab_file, model_file, counts_file, embeddings_file, output_file):
    complex_vocab = get_vocab(complex_vocab_file)
    simple_vocab = get_vocab(simple_vocab_file)

    with open(model_file, 'rb') as f:
        [clf, means, stds] = pickle.load(f)

    counts = load_unigram_counts(counts_file)
    print("Getting embeddings...")
    embeddings = load_embeddings(embeddings_file)

    ## Gets all word and paraphrase features 
    vocab_features = get_all_features(vocab, counts, embeddings)

    ## Gets all predicted complexities
    vocab_complexities = predict_complexity(vocab_features, clf, means, stds)

    ## Saves predicted complexities
    save_complexities(vocab, vocab_complexities, output_file)
    

    

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
~/sockeye-recipes/egs/pretrained_embeddings/model_tokens/vocab.src.0.json \
~/sockeye-recipes/egs/pretrained_embeddings/model_tokens/vocab.trg.0.json \
~/sockeye-recipes/new_scripts/Predict_Word_Level/lin_reg_we_weight_0.4_0.7.pkl \
/data2/text_simplification/other_data/word_complexity/unigram_counts.txt \
/data2/embeddings/eng/GoogleNews-vectors-negative300.bin \
~/sockeye-recipes/data/word_complexity_predictions.txt
'''
