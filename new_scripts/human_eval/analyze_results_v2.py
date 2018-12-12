import sys
import csv
from itertools import combinations

def read_results(mturk_file):
    mturk_dict = dict()
    with open(mturk_file, 'r', encoding='utf8') as f:
        csvreader = csv.reader(f)
        headers = next(csvreader)
        
        for i, row in enumerate(csvreader):
            mturk_dict[i] = {key: value for key, value in zip(headers, row)}
    return mturk_dict

def calculate_results(mturk_dict, name):
    stat = [dict() for i in range(7)]

    systems = ["REFERENCE SIMPLE", "DRESS-LS", "MATCH_DRESS0", \
               "MATCH_DRESS2", "MATCH_DRESS-2"]

    reference_sents = dict()

    #print(len(list(mturk_dict.keys())))

    '''
    for rowid, row_dict in mturk_dict.items():
        print(row_dict)
        print("\n\n")
    '''

    bads = 0
    goods = 0
    region = dict()
    for rowid, row_dict in mturk_dict.items():
        for i in range(1, 6):
            complex_sent = row_dict["Input.sys" + str(i) + "0"]
            #print("\nComplex Sent: " + complex_sent)
            ## Gets statistic for each system on sentence SENT_ID

            
            for j in range(1, 6):
                index = int(row_dict["Input.id" + str(i) + str(j)])
                sent_id = row_dict["HITId"] + str(i)

                try:
                    s = int(row_dict["Answer." + name + str(i) + str(j)])

                    ## Adds to a list (each sentence should have 5 judgements)
                    try:
                        stat[index][sent_id].append(s)
                    except KeyError:
                        stat[index][sent_id] = [s]

                    simple_sent = row_dict["Input.sys" + str(i) + str(j)]
                    goods += 1
                except ValueError:
                    '''
                    print(sent_id)
                    print(row_dict["Input.sys" + str(i) + str(j)])
                    '''
                    bads += 1
                
        
        try:
            region[row_dict["Answer.region"]] += 1
        except KeyError:
            region[row_dict["Answer.region"]] = 1
        
    print("BADS: " + str(bads))
    print("GOODS: " + str(goods))
    print(region)

    ## Calculates average disagreement between human annotators
    diffs = []
    diff_dist = [0 for i in range(5)]
    for s in stat:
        for k,v in s.items():
            if len(v) > 1:
                combs = combinations(list(range(len(v))), 2)
                for c in combs:
                    diffs.append(abs(v[c[0]] - v[c[1]]))
                    diff_dist[abs(v[c[0]] - v[c[1]])] += 1
                    

    ## Calculates average per sentence id
    done = [0 for i in range(5)]
    not_done = [0 for i in range(5)]
    avg_stat = [[] for i in range(5)]
    for i in range(len(stat)):
        for k, v in stat[i].items():
            ## Keeps track of how many sentences are done
            if len(v) == 5:
                done[i] += 1
            else:
                not_done[i] += 1

            '''
            if len(v) == 3:
                max_index = v.index(max(v))
                del v[max_index]
                min_index = v.index(min(v))
                del v[min_index]
            '''
            
            avg_stat[i].append(sum(v)/len(v))
    ## Calculates overall averages
    avg = [round(sum(s)/len(s), 2) for s in avg_stat]

    print("DONE: " + str(done))
    print("NOT DONE: " + str(not_done))

    try:
        print("Average disagreement: " + str(sum(diffs)/len(diffs)))
    except ZeroDivisionError:
        a = 1
    
    print("Disagreement distribution: ")
    try:
        print(diff_dist)
        print([round(d/sum(diff_dist), 3) for d in diff_dist])
        print()
    except ZeroDivisionError:
        a = 1
    
    for i in range(len(systems)):
        print(systems[i] + ": " + str(avg[i]))                
            
    return avg
        


def main(mturk_file):
    mturk_dict = read_results(mturk_file)
    print("\n## GRAMMAR ##")
    grammar = calculate_results(mturk_dict, 'grammar')
    print("\n## MEANING ##")
    means = calculate_results(mturk_dict, 'mean')
    print("\n## SIMPLICITY ##")
    simps = calculate_results(mturk_dict, 'simp')

    print("\n#### OVERALL STATS ####")
    systems = ["REFERENCE SIMPLE", "DRESS-LS", "MATCH_DRESS0", \
               "MATCH_DRESS2", "MATCH_DRESS-2"]
    for i in range(len(grammar)):
        print(systems[i] + ": " + str(round(sum([grammar[i], means[i], simps[i]])/3, 2)))
        
    

if __name__ == '__main__':
    mturk_file = 'results/V2_Batch3.csv'
    main(mturk_file)

'''
python analyze_results_v2.py \
results/V2_Batch1.csv
'''
