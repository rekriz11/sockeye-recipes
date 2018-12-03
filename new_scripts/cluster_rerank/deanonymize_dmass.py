import sys

## Extracts sentences from xml file
def extract_sentences(candidates_file):
    sents_dict = dict()
    with open(candidates_file, 'r', encoding='utf8') as f:
        for line in f:
            ls1 = line[:-1].split("seg id=\"")

            if len(ls1) > 1:
                ls2 = ls1[1].split("\"> ")
                index = int(ls2[0])

                sentence = ls2[1].split(" </seg>")[0]

                sents_dict[index] = sentence

    sents = []
    for i in range(1, max(list(sents_dict.keys())) + 1):
        sent = sents_dict[i].split(" ")
        for j in range(len(sent)):
            sent[j] = sent[j].replace("#quot#", "'")
            if '@' in sent[j]:
                sent[j] = sent[j].upper()
            
        sents.append(sent)

    return sents

    
                
            

## Gets all candidate simplifications
def get_test_sents(candidates_file, anon_file):
    anon_maps = []
    with open(anon_file, 'r', encoding='utf8') as f:
        for line in f:
            anon_maps.append(eval(line))
    print(anon_maps[0])

    sents = extract_sentences(candidates_file)
    print(sents[0])

    deanon_sents = []
    for i in range(len(sents)):
        deanon = []
        for word in sents[i]:
            try:
                d = anon_maps[i][word]
                deanon += d.split(" ")
            except KeyError:
                deanon.append(word)
        deanon_sents.append(deanon)

    print(deanon_sents[0])
    
    return deanon_sents

## Saves output sentences
def save_output(sentences, output_file):
    with open(output_file, 'w', encoding='utf8') as f:
        for sent in sentences:
            f.write(" ".join(sent) + "\n")
        

def main(candidates_file, deanon_file, output_file):
    ## Gets candidate simplifications
    print("Reading in candidates...")
    sentences = get_test_sents(candidates_file, deanon_file)

    print(len(sentences))
    print(sentences[:10])

    ## Output top sentences
    save_output(sentences, output_file)

if __name__ == '__main__':
    candidates_file = sys.argv[1]
    deanon_file = sys.argv[2]
    output_file = sys.argv[3]

    main(candidates_file, deanon_file, output_file)


'''
python ~/sockeye-recipes/new_scripts/cluster_rerank/deanonymize_dmass.py \
~/sockeye-recipes/egs/pretrained_embeddings/output/tokens.100best \
~/sockeye-recipes/egs/pretrained_embeddings/data/newsela_Zhang_Lapata_splits/V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.ori.test.deanonymiser \
~/sockeye-recipes/egs/pretrained_embeddings/output/tokens.100beam.deanon
'''
