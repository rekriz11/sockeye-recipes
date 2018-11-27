base = '/Users/zhaosanqiang916/git/text_simplification_data/train/dress/wikihugenew/train/'

nmapper = []
for line in open(base + 'rule_mapper.txt'):
    rules = line.split("\t")
    srules, mrules = set(), set()
    for rule in rules:
        items = rule.split('=>')
        if len(items) != 4:
            continue

        ori = items[1]
        tar = items[2]
        oril = ori.split()
        tarl = tar.split()
        if len(oril) == 1 and len(tarl) == 1:
            srules.add((ori, tar))
        else:
            mrules.add((ori, tar))
    filterout_set = set()
    for mrule in mrules:
        for srule in srules:
            if srule[0] in mrule[0] and srule[1] in mrule[1]:
                filterout_set.add(mrule)
    nrules = []
    for rule in rules:
        items = rule.split('=>')
        if len(items) != 4:
            continue

        ori = items[1]
        tar = items[2]
        if (ori, tar) not in filterout_set:
            nrules.append(rule.strip())
    nmapper.append('\t'.join(nrules))


f = open(base + 'rule_mapper_gramfilter.txt', 'w')
f.write('\n'.join(nmapper))
f.close()