import sys

## Parses all sentences from a file 
def get_input(file):
    sents = []
    with open(file, 'r', encoding='utf8') as f:
        for line in f:
            sent = line[:-1].lower().split(" ")
            sent = [s.replace("-lrb-", '(') for s in sent]
            sent = [s.replace("-rrb-", ')') for s in sent]
            sent = [s.replace("''", '"') for s in sent]
            sents.append(' '.join(sent))
    return sents

def save(sents, file):
    with open(file, 'w', encoding='utf8') as f:
        for sent in sents:
            f.write(sent + "\n")

                   
def main(input_file, output_file):
    sents = get_input(input_file)
    save(sents, output_file)   
    
    
if __name__ == '__main__':
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    main(input_file, output_file)
    
'''
Running on Tesla:

python3 ~/sockeye-recipes/new_scripts/preprocess_data/convert_to_lower.py \
/data2/text_simplification/output/newsela_complex.REFERENCE \
/data2/text_simplification/output/newsela_complex.REFERENCE

python3 ~/sockeye-recipes/new_scripts/preprocess_data/convert_to_lower.py \
/data2/text_simplification/output/newsela_simple.REFERENCE \
/data2/text_simplification/output/newsela_simple.REFERENCE

python3 ~/sockeye-recipes/new_scripts/preprocess_data/convert_to_lower.py \
/data2/text_simplification/output/dress-ls.BASELINE \
/data2/text_simplification/output/dress-ls.BASELINE

'''
