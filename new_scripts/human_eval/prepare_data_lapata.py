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

def save_csv(complex_sents, reference_sents, simple_sents, indices, random_bool, output_file):
    with open(output_file, 'w', encoding='utf8') as f:
        csvwriter = csv.writer(f)

        firstrow = ['sys10', 'sys11', 'sys12', 'sys13', 'sys14', \
                    'id11', 'id12', 'id13', 'id14', \
                    'sys20', 'sys21', 'sys22', 'sys23', 'sys24', \
                    'id21', 'id22', 'id23', 'id24', \
                    'sys30', 'sys31', 'sys32', 'sys33', 'sys34', \
                    'id31', 'id32', 'id33', 'id34', \
                    'sys40', 'sys41', 'sys42', 'sys43', 'sys44', \
                    'id41', 'id42', 'id43', 'id44', \
                    'sys50', 'sys51', 'sys52', 'sys53', 'sys54', \
                    'id51', 'id52', 'id53', 'id54']
        csvwriter.writerow(firstrow)

        rows = []
        row1 = []
        row2 = []
        for c, i in enumerate(indices):
            print("COMPLEX SENTENCE: " + str(complex_sents[i]))
            print("REFERENCE SENTENCE: " + str(reference_sents[i]))
            print("SIMPLE SENTENCES: " + str(simple_sents[i]) + "\n")
            
            if c % 5 == 0 and len(row1) == 5:
                rows.append(row1)
                rows.append(row2)

                row1 = []
                row2 = []

            system_indices = list(range(len(simple_sents[0])))
            ref_ind1 = 1
            ref_ind2 = 1

            print("RANDOM BOOL: " + str(random_bool))
            if random_bool:
                random.shuffle(system_indices)
                ref_ind1 = random.randint(1, 4)
                ref_ind2 = random.randint(1, 4)

            print("REFERENCE INDEX 1: " + str(ref_ind1))
            print("REFERENCE INDEX 2: " + str(ref_ind2))
            print("SYSTEM INDICES: " + str(system_indices) + "\n")
                

            r1 = [complex_sents[i]]
            r2 = [complex_sents[i]]

            if ref_ind1 == 1:
                r1.append(reference_sents[i])
            if ref_ind2 == 1:
                r2.append(reference_sents[i])
                
            for c1, i1 in enumerate(system_indices):
                if c1 < 3:
                    r1.append(simple_sents[i][i1])
                    if len(r1) == ref_ind1:
                        r1.append(reference_sents[i])
                else:
                    r2.append(simple_sents[i][i1])
                    if len(r2) == ref_ind2:
                        r2.append(reference_sents[i])

            print("ROW1: " + str(r1))
            print("ROW2: " + str(r2))
            print("\n")
            row1 += r1
            row2 += r2
    
        

def main(complex_file, reference_file, simple_files, output_file, random_bool):
    random.seed(37)

    ## Gets complex and simple sentences
    complex_sents = get_sents(complex_file)
    print(len(complex_sents))

    reference_sents = get_sents(reference_file)
    print(len(reference_sents))

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
    save_csv(complex_sents, reference_sents, simple_sents, indices, random_bool, output_file)    
        
    

if __name__ == '__main__':
    complex_file = sys.argv[1]
    reference_file = sys.argv[2]
    simple1 = sys.argv[3]
    simple2 = sys.argv[4]
    simple3 = sys.argv[5]
    simple4 = sys.argv[6]
    simple5 = sys.argv[7]
    simple6 = sys.argv[8]
    simple_files = [simple1, simple2, simple3, simple4, simple5, simple6]
    output_file = sys.argv[9]
    random_bool = bool(int(sys.argv[10])
    
    
    
    
    main(complex_file, reference_file, simple_files, output_file, random_bool)


'''
python3 ~/sockeye-recipes/new_scripts/human_eval/prepare_data_lapata.py \
/data2/text_simplification/output/newsela_complex.REFERENCE \
/data2/text_simplification/output/newsela_simple.REFERENCE \
/data2/text_simplification/output/seq2seq_greedy.BASELINE \
/data2/text_simplification/output/hybrid.BASELINE \
/data2/text_simplification/output/dress-ls.BASELINE \
/data2/text_simplification/output/loss_greedy.MODEL \
/data2/text_simplification/output/seq2seq_best_0.5_0_0.5.BEST_MODEL \
/data2/text_simplification/output_v2/seq2seq_best.BEST_MODEL \
/data2/text_simplification/human_eval/mturk_input_lapata.csv \
0

'''
