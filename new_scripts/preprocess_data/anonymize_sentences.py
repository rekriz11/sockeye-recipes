import spacy
nlp = spacy.load('en_core_web_lg')
import sys

## Performs tokenization and named-entity recognition on a sentence
def parse(sentence):
    doc = nlp(sentence)
    data = []
    tokens = [X.text for X in doc]
        
    data.append(tokens)
    data.append([X.ent_iob_ for X in doc])
    data.append([X.ent_type_ for X in doc])

    return data

## Parses all sentences from a file 
def parse_sentences(file):
    sents = []
    with open(file, 'r', encoding='utf8') as f:
        i = 0
        for line in f:
            if i % 1000 == 0:
                print(i)
            i += 1
            sents.append(parse(line[:-1]))
    return sents

## Anonymizes sentences
def anonymize_data(sents):
    anonymized_data = []

    for sent in sents:
        ori_tokens = sent[0]
        bio = sent[1]
        types = sent[2]

        all_nes = []
        all_types = []

        ## Groups all entities
        current_ne = []
        current_type = ""
        for i in range(len(ori_tokens)):
            if bio[i] == "B":
                if current_ne != []:
                    all_nes.append(current_ne)
                    all_types.append(current_type)
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

## Saves anonymized sentences and anonymization mappings
def save_aner_data(anonymized_data, output_file, anon_file):
    with open(output_file, 'w', encoding='utf8') as f:
        for i in range(len(anonymized_data)):
            sent = []
            for s in anonymized_data[i][0]:
                if "@" not in s:
                    sent.append(s.lower())
                else:
                    sent.append(s.upper())
            f.write(" ".join(sent) + "\n")

    with open(anon_file, 'w', encoding='utf8') as f:
        for i in range(len(anonymized_data)):
            anons = []
            for v,k in anonymized_data[i][1].items():
                anons.append(v.upper() + "::" + k)
            f.write("\t".join(anons) + "\n")

                   
def main(input_file, output_base):
    ## Parses all sentences
    print("Parsing data...")
    parsed_sents = parse_sentences(input_file)
    
    ## Anonymizes data
    print("Anonymizing data...")
    anon_sents = anonymize_data(parsed_sents)

   
    ## Saves anonymized data
    print("Saving anonymized data...")
    save_aner_data(anon_sents, \
                   output_base + ".aner", \
                   output_base + ".aner_map")    
    
    
if __name__ == '__main__':
    input_file = sys.argv[1]
    output_base = sys.argv[2]
    main(input_file, output_base)
    
'''
Running on Tesla:

cd ~/sockeye-recipes/new_scripts/preprocess_data/

python3 anonymize_sentences.py \
/data2/text_simplification/dataset/test/V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.ori.test.src \
/data2/text_simplification/Newsela_v3/test/TEST

python3 anonymize_sentences.py \
/data2/text_simplification/dataset/fb_headline_first_sent/train.article.txt_no_fin \
/data2/text_simplification/dataset/fb_headline_first_sent/train.src
'''
