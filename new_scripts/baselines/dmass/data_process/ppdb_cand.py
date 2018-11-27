from collections import defaultdict, Counter

base = '/Users/zhaosanqiang916/git/text_simplification_data/train/dress/wikihugenew/train/'
# ppdb = '/Users/zhaosanqiang916/git/text_simplification_data/ppdb/'

dic = defaultdict(float)
for line in open(base + 'rule_mapper_gramfilter.txt'):
    for rule in line.strip().split('\t'):
        items = rule.split('=>')
        if len(items) == 4:
            dic[rule] = max(float(items[-1]), dic[rule])

f = open(base + 'rule_cand.txt', 'w')
rules = Counter(dic).most_common()
cont = []
for rule in rules:
    cont.append(rule[0])
f.write('\n'.join(cont))
f.close()

print('Done')


