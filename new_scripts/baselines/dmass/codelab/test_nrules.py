from collections import defaultdict
from nltk.corpus import stopwords
import re


stopWords = set(stopwords.words('english'))
base = '/Users/zhaosanqiang916/git/text_simplification_data/train/dress/wikilargenew/train/'


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
    tokens = [w if not re.match('^\d+$', w) else "#num#" for w in tokens]
    return tokens


mappers_ori, mappers_tar = defaultdict(dict), defaultdict(dict)
for line in open(base + 'rule_cand.txt'):
    items = line.strip().split('=>')
    if len(items) == 4:
        ori = items[1]
        tar = items[2]
        if ori in stopWords:
            continue
        weight = float(items[3])
        mappers_ori[ori][tar] = weight
        mappers_tar[tar][ori] = weight


ls_src = open(base + 'src2.txt').readlines()
ls_dst = open(base + 'dst2.txt').readlines()
# f_map = open(base + 'rule_mapper2.txt').readlines()

for lid in range(len(ls_src)):
    ts_src, ts_dst = rui_preprocess(ls_src[lid]), rui_preprocess(ls_dst[lid])
    ts_src, ts_dst = set(ts_src), set(ts_dst)
    dif = ts_dst - ts_src
    for w in dif:
        if w not in mappers_ori[w]:
            x = 1






