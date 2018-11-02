#python example to infer document vectors from trained doc2vec model
import sys
import gensim.models as g
import codecs
import random
from scipy.cluster.vq import kmeans, vq, whiten
import numpy as np
import io
from sklearn.metrics.pairwise import euclidean_distances

## Calculates distance between 2 vectors
def calc_distance(x, y):
    nx = np.asarray(x)
    nx = nx.reshape(1, -1)
    ny = np.asarray(y)
    ny = ny.reshape(1, -1)

    dist = euclidean_distances(nx, ny)
    print(dist[0][0])
    return dist[0][0]

## Gets all candidate simplifications
def get_test_sents(candidates_file):
    sents = []
    with io.open(candidates_file, 'r', encoding='utf8') as f:
        for line in f:
            ls = line[:-1].split("\t")
            sents.append([s.split(" ") for s in ls])
    return sents

## Predicts vector for each sentence 
def get_embeddings(sents, model, start_alpha, infer_epoch):
    embeddings = []
    for snts in sents:
        embeds = []
        for sent in snts:
            embed = model.infer_vector(sent, alpha=start_alpha, steps=infer_epoch)
            embeds.append(embed)
        embeddings.append(np.asarray(embeds))
    return embeddings

## Runs kmeans on each set of sentences, and returns the sentences closest
## to each centroid
def get_clusters(embeddings, num_clusters):
    sentence_centroids = []
    for embeds in embeddings:
        std_embeds = whiten(embeds)
        centroids,_ = kmeans(std_embeds, num_clusters)
        idx,_ = vq(std_embeds, centroids)

        sent_inds = []
        for c in centroids:
            ind = -1
            min_distance = 100000000000
            for i in range(len(std_embeds)):
                dist = calc_distance(c, std_embeds[i])
                if dist < min_distance:
                    ind = i
                    min_distance = dist
            sent_inds.append(ind)

        sentence_centroids.append(sent_inds)
    return sentence_centroids

## Saves output sentences
def save_output(sentences, output_file):
    with io.open(output_file, 'w', encoding='utf8') as f:
        for sents in sentences:
            f.write("\t".join(sents) + "\n")
        

def main(model_file, candidates_file, num_clusters, output_file):
    random.seed(37)
    
    ## Loads doc2vec model
    print("Loading model...")
    model = g.Doc2Vec.load(model_file)

    ## Gets candidate simplifications
    sents = get_test_sents(candidates_file)[:1]

    ## Gets embeddings for each sentence
    print("Getting embeddings...")
    start_alpha = 0.01
    infer_epoch = 1000
    embeddings = get_embeddings(sents, model, start_alpha, infer_epoch)

    ## Clusters embeddings
    print("Getting centroids...")
    sentence_centroids = get_clusters(embeddings, num_clusters)

    ## Output top sentences
    save_output(sentence_centroids, output_file)

if __name__ == '__main__':
    model_file = sys.argv[1]
    candidates_file = sys.argv[2]
    num_clusters = int(sys.argv[3])
    output_file = sys.argv[4]

    main(model_file, candidates_file, num_clusters, output_file)


'''
python cluster.py \
/data2/text_simplification/embeddings/enwiki_dbow/doc2vec.bin \
~/sockeye-recipes/egs/pretrained_embeddings/output/tokens.400best \
20 \
~/sockeye-recipes/egs/pretrained_embeddings/output/tokens.400beam.20centroids
'''
