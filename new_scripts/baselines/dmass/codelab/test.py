from collections import defaultdict

lines = open('/Users/zhaosanqiang916/git/text_simplification_data/train/dress/wikilargenew/train/rule_mapper.txt').readlines()

mapper = defaultdict(set)
mapper2 = defaultdict(set)
mapper3 = defaultdict(set)

for line in lines:
    items = line.strip().split('\t')
    for item in items:
        if len(item) == 0:
            continue

        units = item.split('=>')
        src = units[1]
        dst = units[2]

        if len(src.split()) == 1:
            mapper[src].add(dst)
        if len(src.split()) == 2:
            mapper2[src].add(dst)
        if len(src.split()) == 3:
            mapper3[src].add(dst)

print('x')

