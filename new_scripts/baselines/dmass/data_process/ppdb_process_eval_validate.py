"""Pseudo Validating of eval/test rules"""
from datetime import datetime
from operator import itemgetter
from collections import Counter
import sys
import multiprocessing
import re
# sys.path.insert(0,'/Users/zhaosanqiang916/git/text_simplification3/script/en/')
import en

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


def verb_ops(nori_words, tar_words, mapper, weight, checker, ing=False):
    # Raw
    try:
        ntar_words = tar_words.split()
        ntar_words[0] = ntar_words[0]
        ntar_words = ' '.join(ntar_words)
        if nori_words + '=>' + ntar_words not in checker:
            checker.add(nori_words + '=>' + ntar_words)
            if nori_words not in mapper:
                mapper[nori_words] = []
            mapper[nori_words].append((ntar_words, weight))
    except:
        x = 1
    # try:
    #     ntar_words = tar_words.split()
    #     ntar_words[0] = en.verb.infinitive(ntar_words[0])
    #     ntar_words = ' '.join(ntar_words)
    #     if nori_words + '=>' + ntar_words not in checker:
    #         checker.add(nori_words + '=>' + ntar_words)
    #         if nori_words not in mapper:
    #             mapper[nori_words] = []
    #         if ntar_words:
    #             mapper[nori_words].append((ntar_words, weight))
    # except KeyError:
    #     x = 1
    # try:
    #     ntar_words = tar_words.split()
    #     ntar_words[0] = en.verb.conjugate(ntar_words[0])
    #     ntar_words = ' '.join(ntar_words)
    #     if nori_words + '=>' + ntar_words not in checker:
    #         checker.add(nori_words + '=>' + ntar_words)
    #         if nori_words not in mapper:
    #             mapper[nori_words] = []
    #         mapper[nori_words].append((ntar_words, weight))
    # except KeyError:
    #     x = 1

    # Past
    try:
        ntar_words = tar_words.split()
        ntar_words[0] = en.verb.past(ntar_words[0])
        ntar_words = ' '.join(ntar_words)
        if nori_words + '=>' + ntar_words not in checker:
            checker.add(nori_words + '=>' + ntar_words)
            if nori_words not in mapper:
                mapper[nori_words] = []
            if ntar_words:
                mapper[nori_words].append((ntar_words, weight))
    except KeyError:
        x = 1

    # present 1/2/3
    try:
        ntar_words = tar_words.split()
        ntar_words[0] = en.verb.present(ntar_words[0], person=1)
        ntar_words = ' '.join(ntar_words)
        if nori_words + '=>' + ntar_words not in checker:
            checker.add(nori_words + '=>' + ntar_words)
            if nori_words not in mapper:
                mapper[nori_words] = []
            if ntar_words:
                mapper[nori_words].append((ntar_words, weight))
    except KeyError:
        x = 1
    try:
        ntar_words = tar_words.split()
        ntar_words[0] = en.verb.present(ntar_words[0], person=2)
        ntar_words = ' '.join(ntar_words)
        if nori_words + '=>' + ntar_words not in checker:
            checker.add(nori_words + '=>' + ntar_words)
            if nori_words not in mapper:
                mapper[nori_words] = []
            if ntar_words:
                mapper[nori_words].append((ntar_words, weight))
    except KeyError:
        x = 1
    try:
        ntar_words = tar_words.split()
        ntar_words[0] = en.verb.present(ntar_words[0], person=3)
        ntar_words = ' '.join(ntar_words)
        if nori_words + '=>' + ntar_words not in checker:
            checker.add(nori_words + '=>' + ntar_words)
            if nori_words not in mapper:
                mapper[nori_words] = []
            if ntar_words:
                mapper[nori_words].append((ntar_words, weight))
    except KeyError:
        x = 1
    if ing:
        # present_participle/past_participle
        try:
            ntar_words = tar_words.split()
            ntar_words[0] = en.verb.present_participle(ntar_words[0])
            ntar_words = ' '.join(ntar_words)
            if nori_words + '=>' + ntar_words not in checker:
                checker.add(nori_words + '=>' + ntar_words)
                if nori_words not in mapper:
                    mapper[nori_words] = []
                if ntar_words:
                    mapper[nori_words].append((ntar_words, weight))
        except KeyError:
            x = 1
    try:
        ntar_words = tar_words.split()
        ntar_words[0] = en.verb.past_participle(ntar_words[0])
        ntar_words = ' '.join(ntar_words)
        if nori_words + '=>' + ntar_words not in checker:
            checker.add(nori_words + '=>' + ntar_words)
            if nori_words not in mapper:
                mapper[nori_words] = []
            if ntar_words:
                mapper[nori_words].append((ntar_words, weight))
    except KeyError:
            x = 1

    return mapper


def verb_ops_ori(ori_words, tar_words, mapper, weight, checker):
    # Raw
    try:
        nori_words = ori_words.split()
        nori_words[0] = nori_words[0]
        nori_words = ' '.join(nori_words)
        if nori_words not in checker:
            checker.add(nori_words)
            mapper = verb_ops(nori_words, tar_words, mapper, weight, checker)
    except KeyError:
        x = 1
    # try:
    #     nori_words = ori_words.split()
    #     nori_words[0] = en.verb.infinitive(nori_words[0])
    #     nori_words = ' '.join(nori_words)
    #     if nori_words not in checker:
    #         checker.add(nori_words)
    #         mapper = verb_ops(nori_words, tar_words, mapper, weight, checker)
    # except KeyError:
    #     x = 1
    # try:
    #     nori_words = ori_words.split()
    #     nori_words[0] = en.verb.conjugate(nori_words[0])
    #     nori_words = ' '.join(nori_words)
    #     if nori_words not in checker:
    #         checker.add(nori_words)
    #         mapper = verb_ops(nori_words, tar_words, mapper, weight, checker)
    # except KeyError:
    #     x = 1

    #Past
    try:
        nori_words = ori_words.split()
        nori_words[0] = en.verb.past(nori_words[0])
        nori_words = ' '.join(nori_words)
        if nori_words not in checker:
            checker.add(nori_words)
            mapper = verb_ops(nori_words, tar_words, mapper, weight, checker)
    except KeyError:
        x = 1

    # present 1/2/3
    try:
        nori_words = ori_words.split()
        nori_words[0] = en.verb.present(nori_words[0], person=1)
        nori_words = ' '.join(nori_words)
        if nori_words not in checker:
            checker.add(nori_words)
            mapper = verb_ops(nori_words, tar_words, mapper, weight, checker)
    except KeyError:
        x = 1
    try:
        nori_words = ori_words.split()
        nori_words[0] = en.verb.present(nori_words[0], person=2)
        nori_words = ' '.join(nori_words)
        if nori_words not in checker:
            checker.add(nori_words)
            mapper = verb_ops(nori_words, tar_words, mapper, weight, checker)
    except KeyError:
        x = 1
    try:
        nori_words = ori_words.split()
        nori_words[0] = en.verb.present(nori_words[0], person=3)
        nori_words = ' '.join(nori_words)
        if nori_words not in checker:
            checker.add(nori_words)
            mapper = verb_ops(nori_words, tar_words, mapper, weight, checker)
    except KeyError:
        x = 1

    # present_participle/past_participle
    try:
        nori_words = ori_words.split()
        nori_words[0] = en.verb.present_participle(nori_words[0])
        nori_words = ' '.join(nori_words)
        if nori_words not in checker:
            checker.add(nori_words)
            mapper = verb_ops(nori_words, tar_words, mapper, weight, checker, ing=True)
    except KeyError:
        x = 1
    try:
        nori_words = ori_words.split()
        nori_words[0] = en.verb.past_participle(nori_words[0])
        nori_words = ' '.join(nori_words)
        if nori_words not in checker:
            checker.add(nori_words)
            mapper = verb_ops(nori_words, tar_words, mapper, weight, checker)
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

mapper = {}
s_time = datetime.now()
e_time = datetime.now()
pre_cnt = 0
cnt = 0
def process_line(line):
    global mapper, s_time, e_time, pre_cnt, cnt
    items = line.strip().lower().split('\t')
    cnt += 1
    if cnt - pre_cnt > 480000:
        e_time = datetime.now()
        span = e_time - s_time
        print('Process %s using %s' % (str(cnt), str(span)))
        s_time = e_time
        pre_cnt = cnt
    if len(items) < 5 or items[2] == '[cd]':
        return
    ori_words = items[3]
    tar_words = items[4]
    try:
        if en.verb.infinitive(ori_words) == en.verb.infinitive(tar_words):
            return
        elif float(len(lcs(ori_words, tar_words))) / max(len(ori_words), len(tar_words)) >= 0.7:
            return
    except KeyError:
        x = 1
    if not ori_words or not tar_words:
        return
    # if ori_words.startswith(', ') and tar_words.startswith(', '):
    #     ori_words = ori_words[2:]
    #     tar_words = tar_words[2:]
    if (not is_alpha_srt(ori_words) or not is_alpha_srt(tar_words) or
                len(ori_words) < 2 or len(tar_words) < 2):
        return
    checker = set()
    weight = float(items[1])
    if ori_words not in mapper:
        mapper[ori_words] = []
    mapper[ori_words].append((tar_words, weight))
    if items[2].startswith('[nn'):
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
                mapper[nori_words].append((ntar_words, weight))
        except KeyError:
            x = 1

        # try:
        #     nori_words = ori_words.split()
        #     nori_words[0] = en.noun.singular(nori_words[0])
        #     nori_words = ' '.join(nori_words)
        #     ntar_words = tar_words.split()
        #     ntar_words[0] = en.noun.plural(ntar_words[0])
        #     ntar_words = ' '.join(ntar_words)
        #     if nori_words not in mapper:
        #         mapper[nori_words] = []
        #     mapper[nori_words].append((ntar_words, weight))
        # except KeyError:
        #     x = 1
        #
        # try:
        #     nori_words = ori_words.split()
        #     nori_words[0] = en.noun.plural(nori_words[0])
        #     nori_words = ' '.join(nori_words)
        #     ntar_words = tar_words.split()
        #     ntar_words[0] = en.noun.singular(ntar_words[0])
        #     ntar_words = ' '.join(ntar_words)
        #     if nori_words not in mapper:
        #         mapper[nori_words] = []
        #     mapper[nori_words].append((ntar_words, weight))
        # except KeyError:
        #     x = 1
        #
        # try:
        #     nori_words = ori_words.split()
        #     nori_words[0] = en.noun.singular(nori_words[0])
        #     nori_words = ' '.join(nori_words)
        #     ntar_words = tar_words.split()
        #     ntar_words[0] = en.noun.singular(ntar_words[0])
        #     ntar_words = ' '.join(ntar_words)
        #     if nori_words not in mapper:
        #         mapper[nori_words] = []
        #     mapper[nori_words].append((ntar_words, weight))
        # except KeyError:
        #     x = 1
    if items[2].startswith('[v'):
        mapper = verb_ops_ori(ori_words, tar_words, mapper, weight, checker)
    if 'be' in ori_words:
        nori_words = ori_words.replace('be', 'is')
        ntar_words = ori_words.replace('be', 'is')
        if nori_words not in mapper:
            mapper[nori_words] = []
        if ntar_words:
            mapper[nori_words].append((ntar_words, weight))

        nori_words = ori_words.replace('be', 'am')
        ntar_words = ori_words.replace('be', 'am')
        if nori_words not in mapper:
            mapper[nori_words] = []
        if ntar_words:
            mapper[nori_words].append((ntar_words, weight))

        nori_words = ori_words.replace('be', 'are')
        ntar_words = ori_words.replace('be', 'are')
        if nori_words not in mapper:
            mapper[nori_words] = []
        if ntar_words:
            mapper[nori_words].append((ntar_words, weight))


def populate_ppdb():
    for line in open('/Users/zhaosanqiang916/git/text_simplification_data/ppdb/SimplePPDB.enrich'):
        process_line(line)

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


def get_best_targets(oriwords, line_dst, widp, line_src):
    results = []
    for tar_words, weight in mapper[oriwords]:
        pos = sequence_contain(line_dst, tar_words.split())
        weight_decay = max(abs(float(pos)/len(line_dst)-widp)-0.2, 0.0)
        if pos != -1 and sequence_contain(line_src, tar_words.split()) == -1:
            results.append((oriwords, tar_words, weight-weight_decay))
    results = list(set(results))
    results.sort(key=itemgetter(-1))
    if len(oriwords.split()) == 1:
        return results[:3]
    else:
        return results[:1]


def get_all_ress(line_src, line_dst, mapper, ress, checker):
    line_dst = rui_preprocess(line_dst)
    for wid in range(len(line_src)):
        # For unigram
        unigram = line_src[wid]
        if unigram in mapper and unigram not in line_dst:
            res = get_best_targets(unigram, line_dst, float(wid)/len(line_src), line_src)
            for r in res:
                k = r[0] + '=>' + r[1]
                if k not in checker:
                    ress.append(r)
                    checker.add(k)

        # For bigram
        if wid + 1 < len(line_src):
            bigram = line_src[wid] + ' ' + line_src[wid + 1]
            if bigram in mapper and sequence_contain(line_dst, (line_src[wid], line_src[wid+1])) == -1:
                res = get_best_targets(bigram, line_dst, wid/len(line_src), line_src)
                for r in res:
                    k = r[0] + '=>' + r[1]
                    if k not in checker:
                        r = list(r)
                        r[-1] -= 0.1
                        r = tuple(r)
                        ress.append(r)
                        checker.add(k)

        # For trigram
        if wid + 2 < len(line_src):
            trigram = line_src[wid] + ' ' + line_src[wid + 1] + ' ' + line_src[wid + 2]
            if trigram in mapper and sequence_contain(line_dst, (line_src[wid], line_src[wid+1], line_src[wid+2])) == -1:
                res = get_best_targets(trigram, line_dst, wid/len(line_src), line_src)
                for r in res:
                    k = r[0] + '=>' + r[1]
                    if k not in checker:
                        r = list(r)
                        r[-1] -= 0.2
                        r = tuple(r)
                        ress.append(r)
                        checker.add(k)
    return ress

mapper = populate_ppdb()
base = '/Users/zhaosanqiang916/git/text_simplification_data/train/dress/wikilargenew/test/'
f_src = open(base + 'src.txt')
f_dst = open(base + 'dst.txt')
f_ref0 = open(base + 'ref.0')
f_ref1 = open(base + 'ref.1')
f_ref2 = open(base + 'ref.2')
f_ref3 = open(base + 'ref.3')
f_ref4 = open(base + 'ref.4')
f_ref5 = open(base + 'ref.5')
f_ref6 = open(base + 'ref.6')
f_ref7 = open(base + 'ref.7')
f = open(base + 'rule_mapper_pseudo.txt', 'w')
lines_mapper = []
s_time = datetime.now()
while True:
    line_src = f_src.readline()
    line_dst = f_dst.readline()
    if not line_src or not line_dst:
        break

    line_src = rui_preprocess(line_src.strip())
    line_dst = line_dst.strip()

    ress = []
    checker = set()

    ress = get_all_ress(line_src, line_dst, mapper, ress, checker)

    line_ref0 = f_ref0.readline()
    line_ref0 = line_ref0.strip()
    ress = get_all_ress(line_src, line_ref0, mapper, ress, checker)

    line_ref1 = f_ref1.readline()
    line_ref1 = line_ref1.strip()
    ress = get_all_ress(line_src, line_ref1, mapper, ress, checker)

    line_ref2= f_ref2.readline()
    line_ref2 = line_ref2.strip()
    ress = get_all_ress(line_src, line_ref2, mapper, ress, checker)

    line_ref3 = f_ref3.readline()
    line_ref3 = line_ref3.strip()
    ress = get_all_ress(line_src, line_ref3, mapper, ress, checker)

    line_ref4 = f_ref4.readline()
    line_ref4 = line_ref4.strip()
    ress = get_all_ress(line_src, line_ref4, mapper, ress, checker)

    line_ref5 = f_ref5.readline()
    line_ref5 = line_ref5.strip()
    ress = get_all_ress(line_src, line_ref5, mapper, ress, checker)

    line_ref6 = f_ref6.readline()
    line_ref6 = line_ref6.strip()
    ress = get_all_ress(line_src, line_ref6, mapper, ress, checker)

    line_ref7 = f_ref7.readline()
    line_ref7 = line_ref7.strip()
    ress = get_all_ress(line_src, line_ref7, mapper, ress, checker)

    ress.sort(key=itemgetter(-1), reverse=True)
    ress = ['X=>%s=>%s=>%s' % (res[0], res[1], str(res[2])) for res in ress]
    lines_mapper.append('\t'.join(ress))
    lines_mapper.append('comp=' + ' '.join(line_src))
    lines_mapper.append('dstt=' + line_dst)
    lines_mapper.append('ref0=' + line_ref0)
    lines_mapper.append('ref1=' + line_ref1)
    lines_mapper.append('ref2=' + line_ref2)
    lines_mapper.append('ref3=' + line_ref3)
    lines_mapper.append('ref4=' + line_ref4)
    lines_mapper.append('ref5=' + line_ref5)
    lines_mapper.append('ref6=' + line_ref6)
    lines_mapper.append('ref7=' + line_ref7)
    lines_mapper.append('==========')
    if len(lines_mapper) % 1000000 == 0:
        e_time = datetime.now()
        span = e_time - s_time
        print('Finished %s with %s.' % (str(len(lines_mapper)), str(span)))
        s_time = e_time
        f.write('\n'.join(lines_mapper))
        f.flush()
        lines_mapper.clear()

f.write('\n'.join(lines_mapper))
f.close()

c = Counter()
for line in lines_mapper:
    rules = line.split('\t')
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

f = open(base + 'rule_voc_pseudo.txt', 'w')
f.write('\n'.join(rules))
f.close()

