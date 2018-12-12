import sys
import csv
import pandas as pd
import scipy.stats as st

SYSTEMS = ["REFERENCE SIMPLE", "S2S", "HYBRID", \
           "DRESS-LS", "DMASS", "S2S-ALL-FAS", "S2S-ALL-FA"]
OUR_MODEL_INDEX = len(SYSTEMS)-1
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
    for aspect in ASPECTS + ['avg']:
        results[aspect] = []
        s1 = df[(df['aspect'] == aspect) & (df['model'] == SYSTEMS[OUR_MODEL_INDEX])]
        for model in SYSTEMS:
            s2 = df[(df['aspect'] == aspect) & (df['model'] == model)]
            results[aspect].append((st.ttest_ind(s1['score'], s2['score']),
                                    st.t.interval(1-ALPHA, len(s2)-1, loc=s2['score'].mean(), scale=s2['score'].std()/len(s2)**0.5)))
    return results


def main(mturk_file):
    mturk_dict = read_results(mturk_file)
    df = refactor(mturk_dict)
    results = hyp_test(df)

    with open('results.txt', 'w') as outfile:
        for aspect, stats in results.items():
            outfile.write(f'===================\nResults for: {aspect}\n===================\n')
            for i, stat in enumerate(stats):
                margin = round((stat[1][1]-stat[1][0]) / 2, rd)
                if i == OUR_MODEL_INDEX:
                    outfile.write(f'{SYSTEMS[i]}: conf=({round(stat[1][0], rd)}, {round(stat[1][1], rd)}), margin={margin}\n')
                else:
                    outfile.write(f'Versus {SYSTEMS[i]}: stat={round(stat[0][0], rd)}, '
                                f'p-value={round(stat[0][1], rd)}, '
                                f'conf=({round(stat[1][0], rd)}, {round(stat[1][1], rd)}), margin={margin}\n')
    #df.to_csv('out.csv', index=False)


if __name__ == '__main__':
    mturk_file = 'results/Batch_3457498_batch_results.csv'
    main(mturk_file)
