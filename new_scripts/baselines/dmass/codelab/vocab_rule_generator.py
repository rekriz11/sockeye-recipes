from collections import Counter

base = '/Users/zhaosanqiang916/git/text_simplification_data/train/dress/wikilarge/train/'

c = Counter()
for line in open(base + 'rule_mapper2.txt'):
    rules = line.strip().split('\t')
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

f = open(base + 'rule_voc2.txt', 'w')
f.write('\n'.join(rules))
f.close()