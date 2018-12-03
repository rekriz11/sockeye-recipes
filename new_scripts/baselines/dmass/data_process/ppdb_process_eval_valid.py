"""Pseudo Validating of eval/test rules"""
from datetime import datetime
from operator import itemgetter
from collections import Counter
import sys
import multiprocessing
import re
# sys.path.insert(0,'/Users/zhaosanqiang916/git/text_simplification3/script/en/')
sys.path.insert(0,'/home/rekriz/sockeye-recipes/new_scripts/baselines/dmass/data_process/')
sys.path.insert(0,'/home/rekriz/sockeye-recipes/new_scripts/baselines/dmass/data_process/en')
import en

print("HI1")

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


def verb_ops_ori(ori_words, tar_words, mapper, weight, checker):
    # Raw
    # try:
    #     nori_words = ori_words.split()
    #     nori_words[0] = nori_words[0]
    #     nori_words = ' '.join(nori_words)
    #
    #     if nori_words not in checker:
    #         checker.add(nori_words)
    #         ntar_words = tar_words.split()
    #         ntar_words[0] = ntar_words[0]
    #         ntar_words = ' '.join(ntar_words)
    #         if nori_words + '=>' + ntar_words not in checker:
    #             checker.add(nori_words + '=>' + ntar_words)
    #             if nori_words not in mapper:
    #                 mapper[nori_words] = []
    #             if ntar_words:
    #                 mapper[nori_words].append((ntar_words, weight, 'vb'))
    # except KeyError:
    #     x = 1

    #Past
    try:
        nori_words = ori_words.split()
        nori_words[0] = en.verb.past(nori_words[0])
        nori_words = ' '.join(nori_words)
        if nori_words not in checker:
            checker.add(nori_words)
            ntar_words = tar_words.split()
            ntar_words[0] = en.verb.past(ntar_words[0])
            ntar_words = ' '.join(ntar_words)
            if nori_words + '=>' + ntar_words not in checker:
                checker.add(nori_words + '=>' + ntar_words)
                if nori_words not in mapper:
                    mapper[nori_words] = []
                if ntar_words:
                    mapper[nori_words].append((ntar_words, weight, 'vbd'))
    except KeyError:
        x = 1

    # present 1/2/3
    try:
        nori_words = ori_words.split()
        nori_words[0] = en.verb.present(nori_words[0], person=1)
        nori_words = ' '.join(nori_words)
        if nori_words not in checker:
            checker.add(nori_words)
            ntar_words = tar_words.split()
            ntar_words[0] = en.verb.present(ntar_words[0], person=1)
            ntar_words = ' '.join(ntar_words)
            if nori_words + '=>' + ntar_words not in checker:
                checker.add(nori_words + '=>' + ntar_words)
                if nori_words not in mapper:
                    mapper[nori_words] = []
                if ntar_words:
                    mapper[nori_words].append((ntar_words, weight, 'vbp'))
    except KeyError:
        x = 1
    try:
        nori_words = ori_words.split()
        nori_words[0] = en.verb.present(nori_words[0], person=2)
        nori_words = ' '.join(nori_words)
        if nori_words not in checker:
            checker.add(nori_words)
            ntar_words = tar_words.split()
            ntar_words[0] = en.verb.present(ntar_words[0], person=2)
            ntar_words = ' '.join(ntar_words)
            if nori_words + '=>' + ntar_words not in checker:
                checker.add(nori_words + '=>' + ntar_words)
                if nori_words not in mapper:
                    mapper[nori_words] = []
                if ntar_words:
                    mapper[nori_words].append((ntar_words, weight, 'vbp'))
    except KeyError:
        x = 1
    try:
        nori_words = ori_words.split()
        nori_words[0] = en.verb.present(nori_words[0], person=3)
        nori_words = ' '.join(nori_words)
        if nori_words not in checker:
            checker.add(nori_words)
            ntar_words = tar_words.split()
            ntar_words[0] = en.verb.present(ntar_words[0], person=3)
            ntar_words = ' '.join(ntar_words)
            if nori_words + '=>' + ntar_words not in checker:
                checker.add(nori_words + '=>' + ntar_words)
                if nori_words not in mapper:
                    mapper[nori_words] = []
                if ntar_words:
                    mapper[nori_words].append((ntar_words, weight, 'vbz'))
    except KeyError:
        x = 1

    # present_participle/past_participle
    try:
        nori_words = ori_words.split()
        nori_words[0] = en.verb.present_participle(nori_words[0])
        nori_words = ' '.join(nori_words)
        if nori_words not in checker:
            checker.add(nori_words)
            ntar_words = tar_words.split()
            ntar_words[0] = en.verb.present_participle(ntar_words[0])
            ntar_words = ' '.join(ntar_words)
            if nori_words + '=>' + ntar_words not in checker:
                checker.add(nori_words + '=>' + ntar_words)
                if nori_words not in mapper:
                    mapper[nori_words] = []
                if ntar_words:
                    mapper[nori_words].append((ntar_words, weight, 'vbg'))
    except KeyError:
        x = 1
    try:
        nori_words = ori_words.split()
        nori_words[0] = en.verb.past_participle(nori_words[0])
        nori_words = ' '.join(nori_words)
        if nori_words not in checker:
            checker.add(nori_words)
            ntar_words = tar_words.split()
            ntar_words[0] = en.verb.past_participle(ntar_words[0])
            ntar_words = ' '.join(ntar_words)
            if nori_words + '=>' + ntar_words not in checker:
                checker.add(nori_words + '=>' + ntar_words)
                if nori_words not in mapper:
                    mapper[nori_words] = []
                if ntar_words:
                    mapper[nori_words].append((ntar_words, weight, 'vbn'))
    except KeyError:
        x = 1
    return mapper


def is_alpha_srt(word):
    for ch in word:
        if (ch < 'a' or ch > 'z') and ch != ' ':
            return False
    return True


def lcs(a, b):
    lengths = [[0 for j in range(len(b) + 1)] for i in range(len(a) + 1)]
    # row 0 and column 0 are initialized to 0 already
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if x == y:
                lengths[i + 1][j + 1] = lengths[i][j] + 1
            else:
                lengths[i + 1][j + 1] = max(lengths[i + 1][j], lengths[i][j + 1])
    # read the substring out from the matrix
    result = ""
    x, y = len(a), len(b)
    while x != 0 and y != 0:
        if lengths[x][y] == lengths[x - 1][y]:
            x -= 1
        elif lengths[x][y] == lengths[x][y - 1]:
            y -= 1
        else:
            assert a[x - 1] == b[y - 1]
            result = a[x - 1] + result
            x -= 1
            y -= 1
    return result

def type_transform(type):
    ts = []
    if type.startswith('['):
        type = type[1:-1]
    type = re.split('(\\\\|\/)', type)
    for t in type:
        if t.startswith('nn'):
            t = 'nn'
        ts.append(t)
    return tuple(ts)

mapper = {}
def process_line(line):
    global mapper
    items = line.strip().lower().split('\t')
    if len(items) < 5 or items[2] == '[cd]':
        return
    ori_words = items[3]
    tar_words = items[4]
    types = type_transform(items[2])
    for t in types:
        try:
            if en.verb.infinitive(ori_words) == en.verb.infinitive(tar_words):
                return
            elif float(len(lcs(ori_words, tar_words))) / max(len(ori_words), len(tar_words)) >= 0.7:
                return
        except KeyError:
            x = 1
        if not ori_words or not tar_words:
            return
        if (not is_alpha_srt(ori_words) or not is_alpha_srt(tar_words) or
                    len(ori_words) < 2 or len(tar_words) < 2):
            return
        checker = set()
        weight = float(items[1])
        if ori_words not in mapper:
            mapper[ori_words] = []
        mapper[ori_words].append((tar_words, weight, t))
        if t == 'nn':
            try:
                nori_words = ori_words.split()
                nori_words[0] = en.noun.plural(nori_words[0])
                nori_words = ' '.join(nori_words)
                ntar_words = tar_words.split()
                ntar_words[0] = en.noun.plural(ntar_words[0])
                ntar_words = ' '.join(ntar_words)
                if nori_words not in mapper:
                    mapper[nori_words] = []
                if ntar_words:
                    mapper[nori_words].append((ntar_words, weight, 'nn'))
            except KeyError:
                x = 1
        if t == 'v':
            mapper = verb_ops_ori(ori_words, tar_words, mapper, weight, checker)
        if 'be' in ori_words:
            nori_words = ori_words.replace('be', 'is')
            ntar_words = ori_words.replace('be', 'is')
            if nori_words not in mapper:
                mapper[nori_words] = []
            if ntar_words:
                mapper[nori_words].append((ntar_words, weight, t))

            nori_words = ori_words.replace('be', 'am')
            ntar_words = ori_words.replace('be', 'am')
            if nori_words not in mapper:
                mapper[nori_words] = []
            if ntar_words:
                mapper[nori_words].append((ntar_words, weight, t))

            nori_words = ori_words.replace('be', 'are')
            ntar_words = ori_words.replace('be', 'are')
            if nori_words not in mapper:
                mapper[nori_words] = []
            if ntar_words:
                mapper[nori_words].append((ntar_words, weight, t))

def populate_ppdb():
    #for line in open('/Users/zhaosanqiang916/git/text_simplification_data/ppdb/SimplePPDB.enrich'):
    s_t = datetime.now()
    cnt = 0
    for line in open('/data2/text_simplification/SimplePPDB/SimplePPDB'):
        #if True:
        if cnt < 10000:
            process_line(line)
            cnt += 1
            if cnt % 500000 == 0:
                e_t = datetime.now()
                sp = e_t - s_t
                print('Process PPDB line\t%s use\t%s' % (str(cnt), str(sp)))
        else:
            break

    print('Populate Mapper with size:%s' % len(mapper))
    return mapper


def sequence_contain(seq, targets):
    if len(targets) == 0:
        print('%s_%s' % (seq, targets))
        return False
    if len(targets) > len(seq):
        return False
    for s_i, s in enumerate(seq):
        t_i = 0
        s_loop = s_i
        if s == targets[t_i]:
            while t_i < len(targets) and s_loop < len(seq) and seq[s_loop] == targets[t_i]:
                t_i += 1
                s_loop += 1
            if t_i == len(targets):
                return s_loop
    return -1


def check_type(t, mt):
    if mt == 'x' or mt == 'new':
        return True
    elif mt == t:
        return True
    else:
        return False


def get_all_targets(oriwords, t, line_src):
    results = []
    for tar_words, weight, mt in mapper[oriwords]:
        if sequence_contain(line_src, tar_words.split()) == -1 and check_type(t, mt):
            results.append((oriwords, tar_words, weight))
    results = list(set(results))
    results.sort(key=itemgetter(-1))
    return results


def get_all_ress(syns, mapper, ress, checker, line_src):
    for pair in syns:
        oriwords = pair[1]
        t = pair[0]
        if oriwords in mapper:
            res = get_all_targets(oriwords, t, line_src)
            for r in res:
                k = r[0] + '=>' + r[1]
                if k not in checker:
                    ress.append(r)
                    checker.add(k)
    return ress

print("HI6")
mapper = populate_ppdb()

voc = set()
voc_file = '/home/rekriz/sockeye-recipes/new_scripts/baselines/dmass/data/newsela/train/rule_voc.txt'
for line in open(voc_file):
    if line[-1] == "\n":
        items = line[:-1].split('\t')
    else:
        items = line.split('\t')
    cnt = int(items[1])
    if cnt >= 5:
        voc.add(items[0])


base = '/home/rekriz/sockeye-recipes/new_scripts/baselines/dmass/data/newsela/valid/'
f_src = open(base + 'valid.aner.src')
f = open(base + 'rule_mapper.txt', 'w')
f_s = open(base + 'syntax.txt')
lines_mapper = []
s_time = datetime.now()

while True:
    line_src = f_src.readline()
    if not line_src :
        break

    line_src = rui_preprocess(line_src.strip())
    line_syn = [p.split('=>') for p in f_s.readline().lower().strip().split('\t') if len(p) > 0]
    line_syn = [(type_transform(p[0])[0], p[1]) for p in line_syn]
    ress = []
    checker = set()

    ress = get_all_ress(line_syn, mapper, ress, checker, line_src)

    # Add Bigram/Trigram without syntax
    for wid in range(len(line_src)):
        if wid + 1 < len(line_src):
            bigram = line_src[wid] + ' ' + line_src[wid + 1]
            if bigram in mapper:
                res = get_all_targets(bigram, 'X', line_src)
                for r in res:
                    k = r[0] + '=>' + r[1]
                    if k not in checker:
                        ress.append(r)
                        checker.add(k)

        if wid + 2 < len(line_src):
            trigram = line_src[wid] + ' ' + line_src[wid + 1] + ' ' + line_src[wid + 2]
            if trigram in mapper:
                res = get_all_targets(trigram, 'X', line_src)
                for r in res:
                    k = r[0] + '=>' + r[1]
                    if k not in checker:
                        ress.append(r)
                        checker.add(k)

    ress.sort(key=itemgetter(-1), reverse=True)
    ress = ['X=>%s=>%s=>%s' % (res[0], res[1], str(res[2])) for res in ress if res[0]+'=>'+res[1] in voc]
    lines_mapper.append('\t'.join(ress))
    if len(lines_mapper) % 1000 == 0:
        e_time = datetime.now()
        span = e_time - s_time
        print('Finished %s with %s.' % (str(len(lines_mapper)), str(span)))
        s_time = e_time
        f.write('\n'.join(lines_mapper))
        f.flush()
        del lines_mapper[:]

f.write('\n'.join(lines_mapper))
f.close()

c = Counter()
for line in lines_mapper:
    rules = line[:-1].split('\t')
    for rule in rules:
        if rule:
            weight = float(rule.split('=>')[-1])
            if rule not in c:
                c[rule] = weight

rules = []
for rule, cnt in c.most_common():
    items = rule.split('=>')
    if len(items) == 4:
        rule = '%s=>%s' % (items[1], items[2])
        rules.append('%s\t%s' % (rule, str(cnt)))

f = open(base + 'rule_voc.txt', 'w')
f.write('\n'.join(rules))
f.close()

