import kenlm
import sys

## Ranks sentences by perplexity, and returns the top one
def rank_candidates(candidates_file, model):
    top_sentences = []
    with open(candidates_file, 'r', encoding='utf8') as f:
        for line in f:
            ls = line[:-1].split("\t")
            scores = []
            for sent in ls:
                scores.append(model.score(sent))
            ind = scores.index(max(scores))
            top_sentences.append(ls[ind])
    return top_sentences

## Outputs sentences to file
def save_sentences(sentences, output_file):
    with open(output_file, 'w', encoding='utf8') as f:
        for sent in sentences:
            f.write(sent + "\n")
            

def main(model_file, candidates_file, output_file):
    model = kenlm.Model(model_file)
    top_sentences = rank_candidates(candidates_file, model)
    save_sentences(top_sentences, output_file)

if __name__ == '__main__':
    model_file = sys.argv[1]
    candidates_file = sys.argv[2]
    output_file = sys.argv[3]

    main(model_file, candidates_file, output_file)


'''
python rerank.py \
~/sockeye-recipes/new_scripts/cluster_rerank/kenlm/lm/test.arpa \
~/sockeye-recipes/egs/pretrained_embeddings/output/tokens.400beam.20centroids \
~/sockeye-recipes/egs/pretrained_embeddings/output/tokens.400beam.top1reranked_perplexity
'''

