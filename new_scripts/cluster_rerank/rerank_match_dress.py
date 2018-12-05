import kenlm
import sys
import gensim.models as g
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from fractions import Fraction
from collections import Counter

## Checks validity of output
def debug_output(sentences, scores, title):
    print(title)
    for i in range(len(sentences)):
        for j in range(len(sentences[i])):
            print(str(scores[i][j]) + "\t" + " ".join(sentences[i][j]))
        print("\n")

## Flattens a two-dimensional list   
def flatten(listoflists):
    list = [item for sublist in listoflists for item in sublist]
    return list

## Normalizes list of lists so all numbers are between 0 and 1
def normalize(listy):
    flat = flatten(listy)
    maxy = max(flat)
    miny = min(flat)
    new_listy = [[(l - miny) / (maxy - miny) for l in ls] for ls in listy]
    return new_listy

## Gets all sentences from a file
def get_sents(file):
    sentences = []
    with open(file, 'r', encoding='utf8') as f:
        for line in f:
            ls = line[:-1].split(" ")
            sentences.append([l.lower() for l in ls])
    return sentences

## Gets all candidate simplifications
def get_simple_sents(candidates_file):
    sentences = []
    with open(candidates_file, 'r', encoding='utf8') as f:
        for line in f:
            ls = line[:-1].split("\t")
            sents = [l.split(" ") for l in ls]
            sentences.append(sents)
    return sentences

## Gets perplexities of candidate simplifications
def get_perplexities(sentences, lm):
    perplexities = []
    for sents in sentences:
        perps = []
        for sent in sents:
            perps.append(lm.score(" ".join(sent)))
        perplexities.append(perps)
        
    norm_perps = normalize(perplexities)
    return norm_perps

## Reads in complexity prediction scores from file
def load_comp_preds(comp_pred_file, sentences):
    scores = []
    with open(comp_pred_file, 'r', encoding='utf8') as f:
        for line in f:
            ls = line[:-1].split("\t")
            scores.append([4 - float(l) for l in ls])
    
    norm_scores = normalize(scores)
    return norm_scores

## Predicts vector for each complex sentence
def get_complex_embeddings(sents, model, start_alpha, infer_epoch):
    embeddings = []
    i = 0
    for sent in sents:
        if i % 100 == 0:
            print(i)
        i += 1
        embed = model.infer_vector(sent, alpha=start_alpha, steps=infer_epoch)
        embeddings.append(embed)
    return embeddings

## Predicts vector for each sentence
def get_embeddings(sents, model, start_alpha, infer_epoch):
    embeddings = []
    i = 0
    for snts in sents:
        if i % 100 == 0:
            print(i)
        i += 1
        embeds = []
        for sent in snts:
            embed = model.infer_vector(sent, alpha=start_alpha, steps=infer_epoch)
            embeds.append(embed)
        embeddings.append(np.asarray(embeds))
    return embeddings

## Calculates cosine similarities between complex and simple sentences
def get_sims(complex_embeddings, embeddings, sentences):
    similarities = []
    for i in range(len(embeddings)):
        sims = []
        for j in range(len(embeddings[i])):
            x = embeddings[i][j].reshape(1, -1)
            y = complex_embeddings[i].reshape(1, -1)

            sims.append(cosine_similarity(x, y)[0][0])
        similarities.append(sims)

    norm_sims = normalize(similarities)
    return norm_sims

## Ranks candidates based on average of fluency, relevancy, and simplicity
def rank_candidates(sentences, dress_sents, perplexities, comp_preds, similarities, weights, diff):
    top_sentences = []
    max_indices = []
    for i in range(len(sentences)):
        scores = []
        for j in range(len(sentences[i])):
            score = weights[0]*perplexities[i][j] + \
                    weights[1]*comp_preds[i][j] + \
                    weights[2]*similarities[i][j]
            scores.append(score)

        '''
        for c, sent in enumerate(sentences[i]):
            print(str(c) + "\t" + str(sent))
        print()
        '''

        ranked_indices = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)

        sent_lengths = [len(sentences[i][s]) for s in ranked_indices]
        dress_length = len(dress_sents[i]) + diff

        '''
        for c in range(len(ranked_indices)):
            print(str(c) + "\t" + str(sentences[i][ranked_indices[c]]))

        print("SENTENCE LENGTHS FOR SENTENCE " + str(i))
        print(sent_lengths)
        print("TARGET LENGTH: " + str(dress_length))
        '''
        

        ## Gets the highest ranked sentence of the same length as dress output
        max_index = -1
        temp_diff = 0
        while max_index == -1:
            for ind in range(int(len(ranked_indices)/2)):
                '''
                print("Comparison: " + str(sent_lengths[ind]) + ", " + \
                      str(dress_length - diff) + ", " + \
                      str(dress_length + diff))
                '''
                if sent_lengths[ind] == dress_length - temp_diff \
                   or sent_lengths[ind] == dress_length + temp_diff:
                    max_index = ind
                    break
            temp_diff += 1

        '''
        print("RANK OF SENTENCE: " + str(max_index))
        print("ORIGINAL INDEX OF SENTENCE: " + str(ranked_indices[max_index]))
        print(sentences[i][ranked_indices[max_index]])
        print("\n")
        '''
        
        
        top_sentences.append(sentences[i][ranked_indices[max_index]])
        max_indices.append(max_index)

    print(sum(max_indices)/len(max_indices))
    ind_dict = dict(Counter(max_indices))
    for i in range(max(ind_dict.keys()) + 1):
        try:
            print(str(i) + ": " + str(ind_dict[i]) + ", " + \
                  str(round(ind_dict[i] / sum(list(ind_dict.values())), 3)))
        except KeyError:
            continue
        
    return top_sentences

## Outputs sentences to file
def save_sentences(sentences, output_file):
    with open(output_file, 'w', encoding='utf8') as f:
        for sent in sentences:
            f.write(" ".join(sent) + "\n")           

def main(doc2vec_file, lm_file, comp_pred_file, candidates_file, \
         complex_file, weights, dress_file, diff, output_file):
    ## Get complex sentences
    print("Reading in complex sentences...")
    complex_sents = get_sents(complex_file)
    print(len(complex_sents))
    print(complex_sents[0])

    ## Get dress sentences
    print("Reading in DRESS sentences...")
    dress_sents = get_sents(dress_file)
    print(len(complex_sents))
    print(complex_sents[0])

    ## Gets candidate simplifications
    print("Reading in candidates...")
    sentences = get_simple_sents(candidates_file)
    print(len(sentences))
    print(len(sentences[0]))

    ## Loads language model
    print("Loading kenlm model...")
    lm = kenlm.Model(lm_file)

    ## Get perplexity scores for each candidate sentence
    print("Calculating perplexities...")
    perplexities = get_perplexities(sentences, lm)

    ## Loads sentence complexity predictions
    print("Getting complexity predictions...")
    comp_preds = load_comp_preds(comp_pred_file, sentences)

    ## Loads doc2vec model
    print("Loading doc2vec model...")
    doc2vec = g.Doc2Vec.load(doc2vec_file)

    ## Gets embeddings for test sentences
    print("Getting complex embeddings...")
    start_alpha = 0.01
    infer_epoch = 1000
    complex_embeddings = get_complex_embeddings(complex_sents, doc2vec, start_alpha, infer_epoch)
    #complex_embeddings = [[0 for i in range(300)] for sent in complex_sents]

    ## Gets embeddings for each sentence
    print("Getting embeddings...")
    embeddings = get_embeddings(sentences, doc2vec, start_alpha, infer_epoch)
    #embeddings = [[[0 for i in range(300)] for sent in sents] for sents in sentences]
    
    ## Calculate cosine similarities between complex and simple sentences
    print("Calculating similarities...")
    similarities = get_sims(complex_embeddings, embeddings, sentences)
    #similarities = [[1 for i in range(len(s))] for s in sentences]

    print("Rerank sentences...")
    ## Reranks sentences based on average of fluency, relevancy, and simplicity
    top_sentences = rank_candidates(sentences, dress_sents, perplexities, comp_preds, similarities, weights, diff)

    save_sentences(top_sentences, output_file)

if __name__ == '__main__':
    doc2vec_file = sys.argv[1]
    lm_file = sys.argv[2]
    comp_pred_file = sys.argv[3]
    complex_file = sys.argv[4]
    candidates_file = sys.argv[5]
    weight1 = float(Fraction(sys.argv[6]))
    weight2 = float(Fraction(sys.argv[7]))
    weight3 = float(Fraction(sys.argv[8]))
    weights = [weight1, weight2, weight3]
    dress_file = sys.argv[9]
    diff = int(sys.argv[10])
    output_file = sys.argv[11]

    main(doc2vec_file, lm_file, comp_pred_file, complex_file, \
         candidates_file, weights, dress_file, diff, output_file)


'''
python ~/sockeye-recipes/new_scripts/cluster_rerank/rerank_match_dress.py \
/data2/text_simplification/embeddings/enwiki_dbow/doc2vec.bin \
/data2/text_simplification/models/lm/lm-merged.kenlm \
~/sockeye-recipes/egs/pretrained_embeddings/output/tokens.100beam.20centroids.preds \
~/sockeye-recipes/egs/pretrained_embeddings/output/tokens.100beam.20centroids \
~/sockeye-recipes/egs/pretrained_embeddings/data/newsela_Zhang_Lapata_splits/V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.ori.test.src \
1 0 0 \
/data2/text_simplification/output/dress-ls.BASELINE \
0 \
~/sockeye-recipes/egs/pretrained_embeddings/output/tokens.100beam.top1reranked_perplexity
'''

