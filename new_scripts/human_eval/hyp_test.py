import sys
import csv
import pandas as pd
import scipy.stats as st

# Assume that the last model is the one we're comparing against
SYSTEMS = ["REFERENCE SIMPLE", "S2S", "HYBRID", \
           "DRESS-LS", "DMASS", "S2S-ALL-FAS", "S2S-ALL-FA"]
ASPECTS = ['mean', 'simp', 'grammar']
ALPHA = 0.05
rd = 5 # number of digits to round to

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
    for _, row in mturk_dict.items():
        for i in range(1, NUM_PROMPTS + 1):
            for j in range(1, NUM_SIMPLIFIED + 1):
                model_name = SYSTEMS[int(row[f'Input.id{i}{j}'])]
                sum_scores = 0
                for aspect in ASPECTS:
                    score = int(row[f'Answer.{aspect}{i}{j}'])
                    sum_scores += score
                    df_rows.append({
                        'model': model_name,
                        'aspect': aspect,
                        'score': score
                    })
                df_rows.append({
                    'model': model_name,
                    'aspect': 'avg',
                    'score': sum_scores / len(ASPECTS)
                })
    return pd.DataFrame(df_rows)


def hyp_test(df):
    results = {}
    for aspect in ASPECTS:
        results[aspect] = []
        s1 = df[(df['aspect'] == aspect) & (df['model'] == SYSTEMS[-1])]
        deg = 2*len(s1)-2
        for model in SYSTEMS[:len(SYSTEMS)-1]:
            s2 = df[(df['aspect'] == aspect) & (df['model'] == model)]
            diffs = s1['score'].reset_index(drop=True) - s2['score'].reset_index(drop=True)
            std = diffs.std() * (2/len(s1))**0.5
            results[aspect].append((st.ttest_ind(s1['score'], s2['score']),
                                    st.t.interval(1-ALPHA, deg, loc=diffs.mean(), scale=std)))

    results['avg'] = []
    s1 = df[(df['aspect'] == 'avg') & (df['model'] == SYSTEMS[-1])]
    deg = 2*len(s1)-2
    for model in SYSTEMS[:len(SYSTEMS)-1]:
        s2 = df[(df['aspect'] == 'avg') & (df['model'] == model)]
        diffs = s1['score'].reset_index(drop=True) - s2['score'].reset_index(drop=True)
        std = diffs.std() * (2/len(s1))**0.5
        results['avg'].append((st.ttest_ind(s1['score'], s2['score']),
                               st.t.interval(1-ALPHA, deg, loc=diffs.mean(), scale=std)))
    return results


def main(mturk_file):
    mturk_dict = read_results(mturk_file)
    df = refactor(mturk_dict)
    results = hyp_test(df)

    with open('results.txt', 'w') as outfile:
        for aspect, stats in results.items():
            outfile.write(f'===================\nResults for: {aspect}\n===================\n')
            for i, stat in enumerate(stats):
                outfile.write(f'Versus {SYSTEMS[i]}: stat={round(stat[0][0], rd)}, '
                              f'p-value={round(stat[0][1], rd)}, '
                              f'conf=({round(stat[1][0], rd)}, {round(stat[1][1], rd)})\n')
    #df.to_csv('out.csv', index=False)


if __name__ == '__main__':
    mturk_file = 'results/Batch_3457498_batch_results.csv'
    main(mturk_file)