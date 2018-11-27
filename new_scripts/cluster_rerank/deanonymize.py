import sys

## Gets all candidate simplifications
def get_test_sents(candidates_file, anon_file):
    anon_maps = []
    with open(anon_file, 'r', encoding='utf8') as f:
        for line in f:
            anon_maps.append(eval(line))
    print(anon_maps[0])

    sents = []
    bads = 0
    with open(candidates_file, 'r', encoding='utf8') as f:
        for line in f:
            ls = line[:-1].split("\t")
            sent = [s.split(" ")for s in ls]
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

## Saves output sentences
def save_output(sentences, output_file):
    with open(output_file, 'w', encoding='utf8') as f:
        for sents in sentences:
            f.write("\t".join([" ".join(s) for s in sents]) + "\n")
        

def main(candidates_file, deanon_file, output_file):
    ## Gets candidate simplifications
    print("Reading in candidates...")
    sentences = get_test_sents(candidates_file, deanon_file)

    print(len(sentences))
    print(len(sentences[0]))
    print(sentences[0][:10])

    ## Output top sentences
    save_output(sentences, output_file)

if __name__ == '__main__':
    candidates_file = sys.argv[1]
    deanon_file = sys.argv[2]
    output_file = sys.argv[3]

    main(candidates_file, deanon_file, output_file)


'''
python ~/sockeye-recipes/new_scripts/cluster_rerank/deanonymize.py \
~/sockeye-recipes/egs/pretrained_embeddings/output/tokens.100best \
~/sockeye-recipes/egs/pretrained_embeddings/data/newsela_Zhang_Lapata_splits/V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.ori.test.deanonymiser \
~/sockeye-recipes/egs/pretrained_embeddings/output/tokens.100beam.deanon
'''
