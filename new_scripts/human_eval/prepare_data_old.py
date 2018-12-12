import sys
import csv
import random

def get_sents(file):
    sents = []
    with open(file, 'r', encoding='utf8') as f:
        for line in f:
            sent = line[:-1].lower().split(" ")
            sent = [s.replace("-lrb-", '(') for s in sent]
            sent = [s.replace("-rrb-", ')') for s in sent]
            sent = [s.replace("''", '"') for s in sent]
            sents.append(' '.join(sent))
    return sents

def save_csv(complex_sents, simple_sents, indices, output_file):
    with open(output_file, 'w', encoding='utf8') as f:
        csvwriter = csv.writer(f)

        firstrow = ['source_0', 'simplification_0_0', 'simplification_0_1', \
                    'simplification_0_2', 'simplification_0_3', 'simplification_0_4', \
                    'simplification_0_5', 'simplification_0_6']
        csvwriter.writerow(firstrow)

        for i in indices:
            row = [complex_sents[i]] + simple_sents[i]
            csvwriter.writerow(row)
    
        

def main(complex_file, simple_files, output_file):
    random.seed(37)

    ## Gets complex and simple sentences
    complex_sents = get_sents(complex_file)
    print(len(complex_sents))

    sents = [get_sents(f) for f in simple_files]
    print(len(sents))
    print(len(sents[0]))
    print("\n")

    simple_sents = []
    for i in range(len(sents[0])):
        ss = []
        for j in range(len(sents)):
            ss.append(sents[j][i])
        if i == 0:
            print(ss)
        simple_sents.append(ss)

    print(len(simple_sents))
    print(len(simple_sents[0]))
    print(simple_sents[10])

    ## Gets random indices of 100 sentences
    indices = list(range(len(complex_sents)))
    random.shuffle(indices)
    indices = indices[:100]
    print(indices)

    ## Saves sentences to csv file
    save_csv(complex_sents, simple_sents, indices, output_file)    
        
    

if __name__ == '__main__':
    complex_file = sys.argv[1]
    simple1 = sys.argv[2]
    simple2 = sys.argv[3]
    simple3 = sys.argv[4]
    simple4 = sys.argv[5]
    simple5 = sys.argv[6]
    simple6 = sys.argv[7]
    simple7 = sys.argv[8]
    simple_files = [simple1, simple2, simple3, simple4, simple5, simple6, simple7]
    output_file = sys.argv[9]
    
    
    main(complex_file, simple_files, output_file)


'''
python3 ~/sockeye-recipes/new_scripts/human_eval/prepare_data.py \
/data2/text_simplification/output/newsela_complex.REFERENCE \
/data2/text_simplification/output/newsela_simple.REFERENCE \
/data2/text_simplification/output/seq2seq_greedy.BASELINE \
/data2/text_simplification/output/hybrid.BASELINE \
/data2/text_simplification/output/dress-ls.BASELINE \
/data2/text_simplification/output/loss_greedy.MODEL \
/data2/text_simplification/output/seq2seq_best.BEST_MODEL \
/data2/text_simplification/output_v2/seq2seq_best.BEST_MODEL \
/data2/text_simplification/human_eval/mturk_input.csv

'''
