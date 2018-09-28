from nltk.translate.bleu_score import corpus_bleu
import numpy as np
from argparse import ArgumentParser
import warnings

warnings.filterwarnings("ignore")

def ReadInFile(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]
    return lines


def main():
    parser = ArgumentParser()
    parser.add_argument("-r", "--reference", dest="reference",
                        help="reference sentences", metavar="FILE")
    parser.add_argument("-s", "--simplified", dest="simplified",
                        help="simplified sentences", metavar="FILE")

    args = parser.parse_args()

    reference_sentences = ReadInFile(args.reference)
    simplified_sentences = ReadInFile(args.simplified)

    bleu_scores = list()
    for i in range(len(simplified_sentences)):
        bleu_scores.append(corpus_bleu([reference_sentences[i]], [simplified_sentences[i]], weights=(1, 0, 0, 0)))

    bleu_scores = np.array(bleu_scores)
    print('BLEU score: {}'.format(np.mean(bleu_scores)))

if __name__ == '__main__':
    main()