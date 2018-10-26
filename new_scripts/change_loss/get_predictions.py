import sys
import pickle
import json
from get_word_features import load_unigram_counts, load_embeddings, get_all_features
import numpy as np

stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', \
'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', \
'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', \
'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', \
'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', \
'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', \
'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', \
'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', \
'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', \
'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', \
'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', \
'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', \
'won', 'wouldn']

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
def predict_complexity(vocab_features, vocab, clf, means, stds):
    word_complexities = dict()
    for word, feats in vocab_features.items():
        if feats != [] and word not in stopwords:
            word_feats = prepare_features(feats, means, stds)
            complexity = 4 - clf.predict(word_feats)[0]
            if complexity > 4.0:               
                word_complexities[word] = (vocab[word], 4.0)
            elif complexity < 0.0:
                word_complexities[word] = (vocab[word], 0.0)
            else:
                word_complexities[word] = (vocab[word], complexity)
        else:
            word_complexities[word] = (vocab[word], -1.0)

    sorted_complexities = ["N/A" for i in range(len(list(word_complexities.keys())))]
    for k,v in word_complexities.items():
        sorted_complexities[v[0]] = v[1]
    return sorted_complexities



## Saves complexities as a json file
def save_complexities(vocab_complexities, output_file):
    with open(output_file, 'wb') as pkl_file:
        pickle.dump(vocab_complexities, pkl_file)
        
        

def main(complex_vocab_file,simple_vocab_file, model_file, counts_file, embeddings_file, output_file):
#    complex_vocab = get_vocab(complex_vocab_file)
    simple_vocab = get_vocab(simple_vocab_file)

    with open(model_file, 'rb') as f:
        [clf, means, stds] = pickle.load(f)

    counts = load_unigram_counts(counts_file)
    print("Getting embeddings...")
    embeddings = load_embeddings(embeddings_file)

    ## Gets all word and paraphrase features
#    complex_features = get_all_features(complex_vocab, counts, embeddings)
    simple_features = get_all_features(simple_vocab, counts, embeddings)

    ## Gets all predicted complexities
#    print("Predicting complex vocab...")
#    complex_complexities = predict_complexity(complex_features, complex_vocab, clf, means, stds)
    print("Predicting simple vocab...")
    simple_compleixties = predict_complexity(simple_features, simple_vocab, clf, means, stds)

    ## Saves predicted complexities
#    save_complexities(complex_complexities, output_file)
    save_complexities(simple_compleixties, output_file)
    
    

    

if __name__ == '__main__':
    complex_vocab_file = sys.argv[1]
    simple_vocab_file = sys.argv[2]
    model_file = sys.argv[3]
    counts_file = sys.argv[4]
    embeddings_file = sys.argv[5]
    output_file = sys.argv[6]

    main(complex_vocab_file, simple_vocab_file, model_file, counts_file, embeddings_file, output_file)

'''
cd ~/sockeye-recipes/new_scripts/change_loss

python3 get_predictions.py \
~/sockeye-recipes/egs/pretrained_embeddings/model_tokens/vocab.src.0.json \
~/sockeye-recipes/egs/pretrained_embeddings/model_tokens/vocab.trg.0.json \
~/sockeye-recipes/new_scripts/Predict_Word_Level/lin_reg_we_weight_0.4_0.7.pkl \
/data2/text_simplification/other_data/word_complexity/unigram-counts.txt \
/data1/embeddings/eng/GoogleNews-vectors-negative300.bin \
~/sockeye-recipes/new_scripts/change_loss/complexity_predictions.pkl

'''

'''
cd ~/sockeye-recipes/new_scripts/change_loss

python3 get_predictions.py \
~/sockeye-recipes/egs/pretrained_embeddings/model_loss_TEST/vocab.src.0.json \
~/sockeye-recipes/egs/pretrained_embeddings/model_loss_TEST/vocab.trg.0.json \
~/sockeye-recipes/new_scripts/Predict_Word_Level/lin_reg_we_weight_0.4_0.7.pkl \
/data2/text_simplification/other_data/word_complexity/unigram-counts.txt \
/data1/embeddings/eng/GoogleNews-vectors-negative300.bin \
~/sockeye-recipes/new_scripts/change_loss/complexity_predictions_TEXT.pkl
'''
