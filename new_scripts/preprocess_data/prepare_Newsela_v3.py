import spacy
nlp = spacy.load('en')
import sys

## Performs tokenization and named-entity recognition on a sentence
def parse_sentence(sentence):
    doc = nlp(sentence)
    data = []
    data.append([X.text for X in doc])
    data.append([X.ent_iob_ for X in doc])
    data.append([X.ent_type_ for X in doc])

    return data

## Performs only tokenization
def tokenize(text):
    return [tok.text for tok in nlp.tokenizer(text)]

## Parses all sentences from a file 
def get_sents(file):
    sents = []
    with open(file, 'r', encoding='utf8') as f:
        for line in f:
            sents.append(parse_sentence(line[:-1]))
    return sents

## Gets original validation/test sentences
def get_original(original_folder, v_or_t):
    src_file = original_folder + "V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.ori." \
               + v_or_t + ".src"
    src_sents = get_sents(src_file)
    
    tgt_file = original_folder + "V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.ori." \
               + v_or_t + ".dst"
    tgt_sents = get_sents(tgt_file)

    return src_sents, tgt_sents

def analyze_v2(newsela_file):
    distribution = dict()
    for i in range(5):
        for j in range(i+1, 5):
            distribution[(i,j)] = 0
            
    with open(newsela_file, 'r', encoding='utf8') as f:
        for line in f:
            ls = line[:-1].split("\t")

            version_src = int(ls[0][1])
            version_tgt = int(ls[1][1])
            distribution[version_src, version_tgt] += 1

    print(distribution)
            
            

## Parses all sentences from Newsela version 2
def get_v2_sents(newsela_file):
    source_sents = []
    target_sents = []

    bads = 0
    with open(newsela_file, 'r', encoding='utf8') as f:
        i = 1
        for line in f:
            if i % 1000 == 0:
                print(i)

            if True:
#            if i < 500:
                ls = line[:-1].split("\t")
                version_src = int(ls[0][1])
                version_tgt = int(ls[1][1])

                sent_src = ls[9]
                sent_tgt = ls[10]

                if version_tgt - version_src > 1 or version_src == 3:
                    source_sents.append([parse_sentence(sent_src), i])
                    target_sents.append([parse_sentence(sent_tgt), i])
                else:
                    bads += 1
            else:
                break

            i += 1
    print(len(source_sents))
    print(bads)
    return source_sents, target_sents
            
def get_train(source_sents, target_sents, valid_src, valid_tgt, test_src, test_tgt):
    s_tokens = []
    t_tokens = []

    ## Gets all validation and test sentences (tokenized)
    for i in range(len(valid_src)):
        s_tokens.append([s[0] for s in valid_src[i]])
        t_tokens.append([s[0] for s in valid_tgt[i]])

    for i in range(len(test_src)):
        s_tokens.append([s[0] for s in test_src[i]])
        t_tokens.append([s[0] for s in test_tgt[i]])

    ## Keeps pair if not found in validation or test set
    train_src = dict()
    train_tgt = dict()
    bads = 0
    for i in range(len(source_sents)):
        src_tokens = [s[0] for s in source_sents[i][0]]
        tgt_tokens = [s[0] for s in target_sents[i][0]]

        if src_tokens not in s_tokens and tgt_tokens not in t_tokens:
            train_src[source_sents[i][1]] = source_sents[i][0]
            train_tgt[target_sents[i][1]] = target_sents[i][0]
        else:
            bads += 1

    print(bads)
    return train_src, train_tgt

def align_sentences(newsela_v3_file, train_src, train_tgt):
    train_src_v3 = []
    train_tgt_v3 = []
    bads = 0
    
    with open(newsela_v3_file, 'r', encoding='utf8') as f:
        c = 0
        for line in f:
            if c % 10000 == 0:
                print(c)
            c += 1

            if c > 1:           
                ls = line[:-1].split("\t")
                sent_id = int(ls[9])

                if ls[10] != "N/A":
                    replaced_id_old = int(ls[11])

                    try:
                        orig_src = train_src[sent_id]
                        orig_tgt = train_tgt[sent_id]

                        if ls[10] == "COMPLEX":
                            ## Finds replaced id
                            new_src = tokenize(ls[15])
                            for i in range(len(new_src)):
                                if new_src[i].lower() != orig_src[i].lower():
                                    replaced_id = i
                                    break
                                
                            replaced_word = ls[15][replaced_id]
                            orig_src[0][replaced_id] = replaced_word
                        else:
                            ## Finds replaced id
                            new_tgt = tokenize(ls[16])
                            for i in range(len(new_tgt)):
                                if new_tgt[i].lower() != orig_tgt[i].lower():
                                    replaced_id = i
                                    break
                                
                            replaced_word = ls[16][replaced_id]
                            orig_tgt[0][replaced_id] = replaced_word

                        train_src_v3.append(orig_src)
                        train_tgt_v3.append(orig_tgt)
                    except KeyError:
                        bads += 1
                        continue
                else:
                    try:
                        train_src_v3.append(train_src[sent_id])
                        train_tgt_v3.append(train_tgt[sent_id])
                    except KeyError:
                        bads += 1
                        continue

    print("\n")
    print(len(train_src_v3))
    print(bads)
    return train_src_v3, train_tgt_v3
                        
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
                    
def main(newsela_file, newsela_v3_file, original_valid_folder, original_test_folder, output_folder):
    print("Analyzing Newsela data...")
    analyze_v2(newsela_file)
    
    ## Parses all sentences from validation and test set
    print("Getting validation data...")
    valid_src, valid_tgt = get_original(original_valid_folder, 'valid')
    print("Getting test data...")
    test_src, test_tgt = get_original(original_test_folder, 'test')

    ## Parses all sentences from Newsela version 2
    print("Getting v2 data...")
    source_sents, target_sents = get_v2_sents(newsela_file)

    ## Only includes sentences not found in validation or test sentences
    print("Splitting training data...")
    train_src, train_tgt = get_train(source_sents, target_sents, valid_src, \
                                     valid_tgt, test_src, test_tgt)

    ## Aligns v3 with v3 parses
    print("Getting v3 data...")
    train_src, train_tgt = align_sentences(newsela_v3_file, train_src, train_tgt)

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
    save_aner_data(train_tgt_aner, output_folder + "train/train.aner.tgt", output_folder + "train/train.tgt.aner_map")
    save_aner_data(valid_src_aner, output_folder + "valid/valid.aner.src", output_folder + "valid/valid.src.aner_map")
    save_aner_data(valid_tgt_aner, output_folder + "valid/valid.aner.tgt", output_folder + "valid/valid.tgt.aner_map")
    save_aner_data(test_src_aner, output_folder + "test/test.aner.src", output_folder + "test/test.src.aner_map")
    save_aner_data(test_tgt_aner, output_folder + "test/test.aner.tgt", output_folder + "test/test.tgt.aner_map")
    
    
    
if __name__ == '__main__':
    newsela_file = sys.argv[1]
    newsela_v3_file = sys.argv[2]
    original_valid_folder = sys.argv[3]
    original_test_folder = sys.argv[4]
    output_folder = sys.argv[5]
    main(newsela_file, newsela_v3_file, original_valid_folder, original_test_folder, output_folder)

'''
Running on Tesla:

cd ~/sockeye-recipes/new_scripts/preprocess_data/

python3 prepare_Newsela_v3.py \
/data2/text_simplification/Newsela_v2/newsela_aligned_v2/newsela_pairs_v2.txt \
/data2/text_simplification/Newsela_v3/newsela_aligned_v3/newsela_pairs_v3.txt \
/data2/text_simplification/dataset/valid/ \
/data2/text_simplification/dataset/test/ \
/data2/text_simplification/Newsela_v3/
'''
