import sys
import csv
import random

def get_sents(file):
    sents = []
    with open(file, 'r', encoding='utf8') as f:
        for line in f:
            sent = line[:-1].lower()
            sent = sent.replace("& amp ;", "&")
            sent = sent.split(" ")
            sent = [s.replace("-lrb-", '(') for s in sent]
            sent = [s.replace("-rrb-", ')') for s in sent]
            sent = [s.replace("''", '"') for s in sent]
            sents.append(' '.join(sent))
    return sents

def save_file(sents, indices, output_file):
    print("Saving to file: " + output_file)
    print(len(sents))
    print(len(indices))
    with open(output_file, 'w', encoding='utf8') as f:
        for i in indices:
            f.write(sents[i] + "\n")
        
def save_csv(complex_sents, reference_sents, simple_sents, indices, random_bool, output_file):
    with open(output_file, 'w', encoding='utf8') as f:
        csvwriter = csv.writer(f)

        firstrow = ['sys10', 'sys11', 'sys12', 'sys13', 'sys14', 'sys15', \
                    'id11', 'id12', 'id13', 'id14', 'id15', \
                    'sys20', 'sys21', 'sys22', 'sys23', 'sys24', 'sys25', \
                    'id21', 'id22', 'id23', 'id24', 'id25', \
                    'sys30', 'sys31', 'sys32', 'sys33', 'sys34', 'sys35', \
                    'id31', 'id32', 'id33', 'id34', 'id35', \
                    'sys40', 'sys41', 'sys42', 'sys43', 'sys44', 'sys45', \
                    'id41', 'id42', 'id43', 'id44', 'id45', \
                    'sys50', 'sys51', 'sys52', 'sys53', 'sys54', 'sys55', \
                    'id51', 'id52', 'id53', 'id54', 'id55']
        csvwriter.writerow(firstrow)

        rows = []
        row1 = []

        numrow = 0
        for c, i in enumerate(indices):
            print("###### NEXT EXAMPLE ######")
            '''
            print("COMPLEX SENTENCE: " + str(complex_sents[i]))
            print("REFERENCE SENTENCE: " + str(reference_sents[i]))
            print("SIMPLE SENTENCES: " + str(simple_sents[i]) + "\n")
            '''
            print(c)
            
            if c % 5 == 0 and len(row1) == 55:
                rows.append(row1)
                numrow += 1
                print("SAVING ROW: " + str(numrow))                    

                row1 = []

            system_indices = list(range(len(simple_sents[0])))
            ref_ind1 = 0

            #print("RANDOM BOOL: " + str(random_bool))
            if random_bool:
                random.shuffle(system_indices)
                ref_ind1 = random.randint(0, 4)

            
            print("REFERENCE INDEX 1: " + str(ref_ind1))
            print("SYSTEM INDICES: " + str([s+1 for s in system_indices]) + "\n")
                

            r1 = [complex_sents[i]]
            index1 = []

            if ref_ind1 == 0:
                r1.append(reference_sents[i])
                index1.append(0)
                
            for c1, i1 in enumerate(system_indices):
                r1.append(simple_sents[i][i1])
                index1.append(i1 + 1)
                if len(r1) == ref_ind1 + 1:
                    r1.append(reference_sents[i])
                    index1.append(0)
            
            print("ROW1: " + str(r1 + index1))
            
            
            row1 += r1 + index1
            print("\n")

        if row1 != []:
            rows.append(row1)
            numrow += 1
            print("SAVING ROW: " + str(numrow))

        for row in rows:
            csvwriter.writerow(row)
    
        

def main(complex_file, reference_file, simple_files, pairwise_folder, lapata_file, random_bool):
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

    ## Gets random indices of 200 sentences
    indices = list(range(len(complex_sents)))
    random.shuffle(indices)
    indices = indices[:200]
    print(indices)

    ## Saves sentences to separate files for pairwise comparison:
    save_file(complex_sents, indices, pairwise_folder + complex_file.split("/")[-1])
    save_file(reference_sents, indices, pairwise_folder + reference_file.split("/")[-1])
    save_file(sents[0], indices, pairwise_folder + simple_files[0].split("/")[-1])
    save_file(sents[1], indices, pairwise_folder + simple_files[1].split("/")[-1])
    save_file(sents[2], indices, pairwise_folder + simple_files[2].split("/")[-1])
    save_file(sents[3], indices, pairwise_folder + simple_files[3].split("/")[-1])
    
    ## Saves sentences to csv file
    save_csv(complex_sents, reference_sents, simple_sents, indices, random_bool, lapata_file)    
        
    

if __name__ == '__main__':
    complex_file = sys.argv[1]
    reference_file = sys.argv[2]
    simple1 = sys.argv[3]
    simple2 = sys.argv[4]
    simple3 = sys.argv[5]
    simple4 = sys.argv[6]
    simple_files = [simple1, simple2, simple3, simple4]
    pairwise_folder = sys.argv[7]
    lapata_file = sys.argv[8]
    random_bool = bool(int(sys.argv[9]))
    
    main(complex_file, reference_file, simple_files, pairwise_folder, lapata_file, random_bool)


'''
python3 ~/sockeye-recipes/new_scripts/human_eval/prepare_data_v2.py \
/data2/text_simplification/output/newsela_complex.REFERENCE \
/data2/text_simplification/output/newsela_simple.REFERENCE \
/data2/text_simplification/output/dress-ls.BASELINE \
/data2/text_simplification/output/seq2seq_best.MATCH_DRESS0 \
/data2/text_simplification/output/seq2seq_best.MATCH_DRESS2 \
/data2/text_simplification/output/seq2seq_best.MATCH_DRESS-2 \
/data2/text_simplification/human_eval/pairwise_data/ \
/data2/text_simplification/human_eval/lapata_data/mturk_input_lapata_v2.csv \
1

'''
