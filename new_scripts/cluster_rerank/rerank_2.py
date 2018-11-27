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
def rank_candidates(sentences, perplexities, comp_preds, similarities, weights):
    top_sentences = []
    for i in range(len(sentences)):
        scores = []
        for j in range(len(sentences[i])):
            score = weights[0]*perplexities[i][j] + \
                    weights[1]*comp_preds[i][j] + \
                    weights[2]*similarities[i][j]
            scores.append(score)

        max_index = scores.index(max(scores))
        top_sentences.append(sentences[i][max_index])
    return top_sentences

## Outputs sentences to file
def save_sentences(sentences, output_file):
    with open(output_file, 'w', encoding='utf8') as f:
        for sent in sentences:
            f.write(" ".join(sent) + "\n")           

def main(doc2vec_file, lm_file, comp_pred_file, candidates_file, \
         complex_file, weights, output_file):
    ## Get complex sentences
    print("Reading in complex sentences...")
    complex_sents = get_complex_sents(complex_file)
    print(len(complex_sents))
    print(complex_sents[0])

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

    ## Gets embeddings for each sentence
    print("Getting embeddings...")
    embeddings = get_embeddings(sentences, doc2vec, start_alpha, infer_epoch)

    ## Calculate cosine similarities between complex and simple sentences
    print("Calculating similarities...")
    similarities = get_sims(complex_embeddings, embeddings, sentences)

    ## Reranks sentences based on average of fluency, relevancy, and simplicity
    top_sentences = rank_candidates(sentences, perplexities, comp_preds, similarities, weights)

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
    output_file = sys.argv[9]

    main(doc2vec_file, lm_file, comp_pred_file, complex_file, \
         candidates_file, weights, output_file)


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

