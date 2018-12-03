import kenlm
import sys
import gensim.models as g
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from fractions import Fraction

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

## Gets all complex sentences
def get_complex_sents(complex_file):
    sentences = []
    with open(complex_file, 'r', encoding='utf8') as f:
        for line in f:
            ls = line[:-1].split(" ")
            sentences.append([l.lower() for l in ls])
    return sentences

## Gets all candidate simplifications
def get_sents(candidates_file):
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

## Ranks candidates based on perplexity
def rank_candidates(sentences, perplexities):
    top_sentences = []
    for i in range(len(sentences)):
        scores = perplexities[i]
        max_index = scores.index(max(scores))
        top_sentences.append(sentences[i][max_index])
    return top_sentences

## Outputs sentences to file
def save_sentences(sentences, output_file):
    with open(output_file, 'w', encoding='utf8') as f:
        for sent in sentences:
            f.write(" ".join(sent) + "\n")           

def main(lm_file, candidates_file, output_file):
    ## Gets candidate simplifications
    print("Reading in candidates...")
    sentences = get_sents(candidates_file)
    print(len(sentences))
    print(len(sentences[0]))

    ## Loads language model
    print("Loading kenlm model...")
    lm = kenlm.Model(lm_file)

    ## Get perplexity scores for each candidate sentence
    print("Calculating perplexities...")
    perplexities = get_perplexities(sentences, lm)

    ## Reranks sentences based on average of fluency
    top_sentences = rank_candidates(sentences, perplexities)

    save_sentences(top_sentences, output_file)

if __name__ == '__main__':
    lm_file = sys.argv[1]
    candidates_file = sys.argv[2]
    output_file = sys.argv[3]

    main(lm_file, candidates_file, output_file)


'''
python ~/sockeye-recipes/new_scripts/cluster_rerank/rerank_2.py \
/data2/text_simplification/embeddings/enwiki_dbow/doc2vec.bin \
/data2/text_simplification/models/lm/lm-merged.kenlm \
~/sockeye-recipes/egs/pretrained_embeddings/output/tokens.100beam.20centroids.preds \
~/sockeye-recipes/egs/pretrained_embeddings/output/tokens.100beam.20centroids \
~/sockeye-recipes/egs/pretrained_embeddings/data/newsela_Zhang_Lapata_splits/V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.ori.test.src \
1 0 0 \
~/sockeye-recipes/egs/pretrained_embeddings/output/tokens.100beam.top1reranked_perplexity
'''

