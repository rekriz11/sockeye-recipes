import spacy
nlp = spacy.load('en_core_web_lg')
import sys

## Performs tokenization and named-entity recognition on a sentence
def parse_sentence(sentence):
    doc = nlp(sentence)
    data = [(X.text, X.ent_iob_, X.ent_type_) for X in doc]
    return data

## Parses all sentences from a file 
def get_sents(file):
    sents = []
    with open(file, 'r', encoding='utf8') as f:
        i = 0
        for line in f:
            if i % 1000 == 0:
                print(i)

            if True:
            #if i < 1000:
                sents.append(parse_sentence(line[:-1]))
            else:
                break

            i += 1
    return sents

## Gets original sentences
def get_original(data_folder, split):
    src_file = data_folder + split + "/V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.ori." \
               + split + ".src"
    src_sents = get_sents(src_file)
    
    tgt_file = data_folder + split + "/V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.ori." \
               + split + ".dst"
    tgt_sents = get_sents(tgt_file)

    return src_sents, tgt_sents

## Saves non-anonymized sentences
def save_original_data(data, output_file):
    with open(output_file, 'w', encoding='utf8') as f:
        for sent in data:
            f.write(" ".join([s[0].lower() for s in sent]) + "\n")

## Anonymizes sentences
def anonymize_data(sents):
    anonymized_data = []

    for sent in sents:
        ori_tokens = [s[0] for s in sent]
        bio = [s[1] for s in sent]
        types = [s[2] for s in sent]

        all_nes = []
        all_types = []

        ## Groups all entities
        current_ne = []
        current_type = ""
        for i in range(len(ori_tokens)):
            if bio[i] == "B":
                current_ne = [i]
                current_type = types[i]
            elif bio[i] == "I":
                current_ne.append(i)
            else:
                if current_ne != []:
                    all_nes.append(current_ne)
                    all_types.append(current_type)
                current_ne = []
                current_type = ""
        if current_ne != []:
            all_nes.append(current_ne)
            all_types.append(current_type)

        ## Makes anonymized dictionary for sentence
        aner_dict = dict()
        ne_starts_dict = dict()
        for i in range(len(all_nes)):
            anonymized = False
            c = 1
            while not anonymized:
                try:
                    a = aner_dict[all_types[i] + "@" + str(c)]
                    c += 1
                except KeyError:
                    aner_dict[all_types[i] + "@" + str(c)] = " ".join([ori_tokens[j] for j in all_nes[i]])
                    ne_starts_dict[all_nes[i][0]] = (all_nes[i], all_types[i] + "@" + str(c))
                    anonymized = True
                    
        ## Makes anonymized sentence
        anon_tokens = []
        i = 0
        while i < len(ori_tokens):
            if i not in ne_starts_dict.keys():
                anon_tokens.append(ori_tokens[i])
                i += 1
            else:
                ## Includes anonymized label
                current_ne_label = ne_starts_dict[i][1]
                anon_tokens.append(current_ne_label)

                ## Skips over indices that are part of named entity
                current_ne_indices = ne_starts_dict[i][0]
                for j in current_ne_indices:
                    i += 1
                    
        anonymized_data.append((anon_tokens, aner_dict))       
    return anonymized_data

## Saves anonymized sentences and anonimization mappings
def save_aner_data(anonymized_data, output_file, anon_file):
    with open(output_file, 'w', encoding='utf8') as f:
        for sent in anonymized_data:
            f.write(" ".join([s.lower() for s in sent[0]]) + "\n")

    with open(anon_file, 'w', encoding='utf8') as f:
        for sent in anonymized_data:
            anons = []
            for v,k in sent[1].items():
                anons.append(v + "::" + k)
            f.write("\t".join(anons) + "\n")

## Saves anonymized sentences
def save_aner_tgt_data(anonymized_data, output_file):
    with open(output_file, 'w', encoding='utf8') as f:
        for sent in anonymized_data:
            f.write(" ".join([s.lower() for s in sent[0]]) + "\n")
                    
def main(data_folder, output_folder):
    ## Parses all sentences
    print("Getting training data...")
    train_src, train_tgt = get_original(data_folder, 'train')
    print("Getting validation data...")
    valid_src, valid_tgt = get_original(data_folder, 'valid')
    print("Getting test data...")
    test_src, test_tgt = get_original(data_folder, 'test')

    ## Saves original data
    print("Saving original data...")
    save_original_data(train_src, output_folder + "train/train.ori.src")
    save_original_data(train_tgt, output_folder + "train/train.ori.tgt")
    save_original_data(valid_src, output_folder + "valid/valid.ori.src")
    save_original_data(valid_tgt, output_folder + "valid/valid.ori.tgt")
    save_original_data(test_src, output_folder + "test/test.ori.src")
    save_original_data(test_tgt, output_folder + "test/test.ori.tgt")
    
    ## Anonymizes data
    print("Anonymizing data...")
    train_src_aner = anonymize_data(train_src)
    train_tgt_aner = anonymize_data(train_tgt)
    valid_src_aner = anonymize_data(valid_src)
    valid_tgt_aner = anonymize_data(valid_tgt)
    test_src_aner = anonymize_data(test_src)
    test_tgt_aner = anonymize_data(test_tgt)

    ## Saves anonymized data
    print("Saving anonymized data...")
    save_aner_data(train_src_aner, output_folder + "train/train.aner.src", output_folder + "train/train.src.aner_map")
    save_aner_tgt_data(train_tgt_aner, output_folder + "train/train.aner.tgt")
    save_aner_data(valid_src_aner, output_folder + "valid/valid.aner.src", output_folder + "valid/valid.src.aner_map")
    save_aner_tgt_data(valid_tgt_aner, output_folder + "valid/valid.aner.tgt")
    save_aner_data(test_src_aner, output_folder + "test/test.aner.src", output_folder + "test/test.src.aner_map")
    save_aner_tgt_data(test_tgt_aner, output_folder + "test/test.aner.tgt")
    
    
    
if __name__ == '__main__':
    data_folder = sys.argv[1]
    output_folder = sys.argv[2]
    main(data_folder, output_folder)

'''
Running on Tesla:

cd ~/sockeye-recipes/new_scripts/preprocess_data/

python3 prepare_Newsela_v1.py \
/data2/text_simplification/dataset/ \
/data2/text_simplification/Newsela_v1/
'''
