# Created by Reno Kriz, University of Pennsylvania
# This code can be used to extract all sentences from Sockeye's beam history
# For detailed instructions on how to run this, please see the README


import argparse
import os
import json
import operator

## Pad token used in sockeye, used to filter out pad tokens from the graph
PAD_TOKEN = "<pad>"
        
## Extracts all partial and complete sentences from beam history
def collect_candidates(input_data, include_pad=False):
    candidates = []

    with open(input_data) as beams:       
        for i, line in enumerate(beams):
            candidate_dicts = []
            start_dict = dict()
            start_dict[0] = [['<s>'], 0]
            candidate_dicts.append(start_dict)
            
            beam = json.loads(line)

            for j in range(len(beam["predicted_tokens"])):
                cand_dict = dict()
                for k in range(len(beam["predicted_tokens"][j])):
                    current_token = beam["predicted_tokens"][j][k]
                    if not include_pad and current_token == PAD_TOKEN:
                        continue
                    
                    parent_id = beam["parent_ids"][j][k]
                    score = beam["normalized_scores"][j][k]

                    current_sentence = candidate_dicts[j][parent_id][0] + [current_token]
                    cand_dict[k] = [current_sentence, score]

                candidate_dicts.append(cand_dict)
            candidates.append(candidate_dicts)
    return candidates

## Extracts complete sentences, and sorts them by score
def find_completes(candidates):
    sentences = []

    for cands in candidates:
        sents = dict()
        for cand_dict in cands:
            for k,v in cand_dict.items():
                sent = v[0]
                score = v[1]
                if sent[len(sent)-1] == '</s>':
                    sents[" ".join(sent[1:-1])] = score

        sorted_sents = sorted(sents.items(), key=operator.itemgetter(1))
        sorted_sents = [s[0] for s in sorted_sents]
        sentences.append(sorted_sents)
    return sentences

## Outputs sentences to file
def output_sentences(sentences, output_file):
    with open(output_file, 'w', encoding='utf8') as f:
        for sents in sentences:
            f.write("\t".join(sents) + "\n")

def main():
    parser = argparse.ArgumentParser(description="Generate nbest sentences")
    parser.add_argument(
        "-d", "--data", type=str, required=True,
        help="path to the beam search data file")
    parser.add_argument(
        "-o", "--output_file", type=str, required=True,
        help="path to the output file")
    parser.add_argument('--pad', dest='include_pad', action='store_true')
    parser.add_argument('--no-pad', dest='include_pad', action='store_false')
    parser.set_defaults(include_pad=False)
    args = parser.parse_args()

    candidates = collect_candidates(args.data, include_pad=args.include_pad)
    sentences = find_completes(candidates)
    output_sentences(sentences, args.output_file)

if __name__ == "__main__":
    main()
