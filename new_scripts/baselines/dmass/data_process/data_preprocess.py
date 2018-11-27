import re
from util import constant

def rui_preprocess(text):
    text = text.lower().strip()
    text = text.replace(
        '-lrb-', '(').replace('-rrb-', ')').replace(
        '-lcb-', '(').replace('-rcb-', '(').replace(
        '-lsb-', '(').replace('-rsb-', '(').replace(
        '\'\'', '"').replace('', '')
    text = re.sub(r'[\r\n\t]', ' ', text)
    text = re.sub(r'[_<>,\(\)\.\'%]', ' \g<0> ', text)
    # tokenize by non-letters
    tokens = filter(lambda w: len(w) > 0, re.split(r'[^a-zA-Z0-9_<>,\(\)\.\'%@]', text))
    # replace the digit terms with <digit>tune.8turkers.tok.norm
    # tokens = [w if not re.match('^\d+$', w) else constant.SYMBOL_NUM for w in tokens]
    return tokens

for ref_i in range(8):
    f = open('/Users/zhaosanqiang916/git/text_simplification_data/train/dress/wikilargenew/val/ref.' + str(ref_i), 'w')
    nlines = []
    for line in open('/Users/zhaosanqiang916/git/text_simplification_data/val/tune.8turkers.tok.turk.' + str(ref_i)):
        line = rui_preprocess(line)
        nlines.append(' '.join(line))
    f.write('\n'.join(nlines))
    f.close()