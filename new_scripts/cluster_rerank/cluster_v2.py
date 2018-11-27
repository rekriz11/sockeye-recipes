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
    return dist[0][0]

## Gets all candidate simplifications
def get_test_sents(candidates_file, anon_file):
    anon_maps = []
    with open(anon_file, 'r', encoding='utf8') as f:
        for line in f:
            if line != "\n":
                ls = line[:-1].split("\t")
                mapy = dict()
                for l in ls:
                    aner = l.split("::")[0]
                    entity = l.split("::")[1].lower()
                    mapy[aner] = entity
                anon_maps.append(mapy)
            else:
                anon_maps.append(dict())
    print(anon_maps[0])

    sents = []
    with open(candidates_file, 'r', encoding='utf8') as f:
        for line in f:
            ls = line[:-1].split("\t")
            sent = [s.split(" ")for s in ls]

            
            for i in range(len(sent)):
                for j in range(len(sent[i])):
                    if '@' in sent[i][j] and len(sent[i][j]) > 1:
                        sent[i][j] = sent[i][j].upper()
            
            
            sents.append(sent)
    print(sents[0][0])

    deanon_sents = []
    for i in range(len(sents)):
        deanon = []
        for sent in sents[i]:
            de = []
            for word in sent:
                try:
                    d = anon_maps[i][word]
                    de += d.split(" ")
                except KeyError:
                    de.append(word)
            deanon.append(de)
        deanon_sents.append(deanon)

    print(deanon_sents[0][0])
    
    return deanon_sents

## Predicts vector for each sentence 
def get_embeddings(sents, model, start_alpha, infer_epoch):
    embeddings = []
    i = 0
    for snts in sents:
        if i % 10 == 0:
            print(i)
        i += 1
        embeds = []
        for sent in snts:
            embed = model.infer_vector(sent, alpha=start_alpha, steps=infer_epoch)
            embeds.append(embed)
        embeddings.append(np.asarray(embeds))
    return embeddings

## Runs kmeans on each set of sentences, and returns the sentences closest
## to each centroid
def get_clusters(embeddings, num_clusters):
    centroids = []
    z = 0
    for embeds in embeddings:
        if z % 10 == 0:
            print(z)
        z += 1
        std_embeds = whiten(embeds)
        cents,_ = kmeans(std_embeds, num_clusters)
        idx,_ = vq(std_embeds, cents)

        sent_inds = []
        for c in cents:
            ind = -1
            min_distance = 100000000000
            for i in range(len(std_embeds)):
                dist = calc_distance(c, std_embeds[i])
                if dist < min_distance:
                    ind = i
                    min_distance = dist
            sent_inds.append(ind)

        centroids.append(sent_inds)
    return centroids

## Gets sentences from indices
def get_sentence_centroids(sentences, centroids):
    sentence_centroids = []
    for i in range(len(sentences)):
        sents = []
        for j in centroids[i]:
            sents.append(sentences[i][j])
        sentence_centroids.append(sents)
    return sentence_centroids

## Saves output sentences
def save_output(sentences, output_file):
    with open(output_file, 'w', encoding='utf8') as f:
        for sents in sentences:
            f.write("\t".join([" ".join(s) for s in sents]) + "\n")
        

def main(model_file, candidates_file, deanon_file, num_clusters, output_file):
    random.seed(37)
    
    ## Loads doc2vec model
    print("Loading model...")
    model = g.Doc2Vec.load(model_file)

    ## Gets candidate simplifications
    print("Reading in candidates...")
    sentences = get_test_sents(candidates_file, deanon_file)

    ## Gets embeddings for each sentence
    print("Getting embeddings...")
    start_alpha = 0.01
    infer_epoch = 1000
    embeddings = get_embeddings(sentences, model, start_alpha, infer_epoch)

    ## Clusters embeddings
    print("Getting centroids...")
    centroids = get_clusters(embeddings, num_clusters)
    sentence_centroids = get_sentence_centroids(sentences, centroids)

    ## Output top sentences
    save_output(sentence_centroids, output_file)

if __name__ == '__main__':
    model_file = sys.argv[1]
    candidates_file = sys.argv[2]
    deanon_file = sys.argv[3]
    num_clusters = int(sys.argv[4])
    output_file = sys.argv[5]

    main(model_file, candidates_file, deanon_file, num_clusters, output_file)


'''
python ~/sockeye-recipes/new_scripts/cluster_rerankcluster.py \
/data2/text_simplification/embeddings/enwiki_dbow/doc2vec.bin \
~/sockeye-recipes/egs/pretrained_embeddings/output/tokens.100best \
~/sockeye-recipes/egs/pretrained_embeddings/data/newsela_Zhang_Lapata_splits/V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.ori.test.deanonymiser \
20 \
~/sockeye-recipes/egs/pretrained_embeddings/output/tokens.100beam.20centroids
'''
