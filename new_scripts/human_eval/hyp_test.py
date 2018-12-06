import sys
import csv
import pandas as pd
from scipy.stats import ttest_ind

# Assume that the last model is the one we're comparing against
SYSTEMS = ["REFERENCE SIMPLE", "S2S", "HYBRID", \
           "DRESS-LS", "DMASS", "S2S-ALL-FAS", "S2S-ALL-FA"]
ASPECTS = ['mean', 'simp', 'grammar']

def read_results(mturk_file):
    mturk_dict = dict()
    with open(mturk_file, 'r', encoding='utf8') as f:
        csvreader = csv.reader(f)
        headers = next(csvreader)
        
        for i, row in enumerate(csvreader):
            mturk_dict[i] = {key: value for key, value in zip(headers, row)}
    return mturk_dict


def refactor(mturk_dict):
    NUM_PROMPTS = 5
    NUM_SIMPLIFIED = 4
    df_rows = []
    count = 0
    for _, row in mturk_dict.items():
        count += 1
        for i in range(1, NUM_PROMPTS + 1):
            for j in range(1, NUM_SIMPLIFIED + 1):
                model_name = SYSTEMS[int(row[f'Input.id{i}{j}'])]
                for aspect in ASPECTS:
                    score = int(row[f'Answer.{aspect}{i}{j}'])
                    df_rows.append({
                        'model': model_name,
                        'aspect': aspect,
                        'score': score
                    })
    return pd.DataFrame(df_rows)


def hyp_test(df):
    results = {}
    for aspect in ASPECTS:
        results[aspect] = []
        s1 = df[(df['aspect'] == aspect) & (df['model'] == SYSTEMS[-1])]
        for model in SYSTEMS[:len(SYSTEMS)-1]:
            s2 = df[(df['aspect'] == aspect) & (df['model'] == model)]
            results[aspect].append(ttest_ind(s1['score'], s2['score']))
    return results


def main(mturk_file):
    mturk_dict = read_results(mturk_file)
    df = refactor(mturk_dict)
    results = hyp_test(df)

    with open('results.txt', 'w') as outfile:
        for aspect, stats in results.items():
            outfile.write(f'===================\nResults for: {aspect}\n===================\n')
            for i, stat in enumerate(stats):
                outfile.write(f'Versus {SYSTEMS[i]}: stat={round(stat[0], 5)}, p-value={round(stat[1], 5)}\n')
    #df.to_csv('out.csv', index=False)


if __name__ == '__main__':
    mturk_file = 'results/Batch_3457498_batch_results.csv'
    main(mturk_file)