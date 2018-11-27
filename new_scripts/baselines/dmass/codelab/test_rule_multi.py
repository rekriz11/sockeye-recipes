
base = '/Users/zhaosanqiang916/git/text_simplification_data/train/dress/wikilargenew/train/'

rule_lines =  open(base + 'rule_mapper2.txt').readlines()
comp_lines = open(base + 'src2.txt').readlines()
simp_lines = open(base + 'dst2.txt').readlines()

for lid in range(len(rule_lines)):
    items = rule_lines[lid].strip().split('\t')
    for item in items:
        pos = item.split('=>')
        if len(pos) < 4:
            continue
        ori = pos[1]
        tar = pos[2]
        if ' ' in ori or ' ' in tar:
            continue
        ci = len(comp_lines[lid].split(ori))-1
        si = len(simp_lines[lid].split(tar))-1
        if si > 1 or ci > 1:
            print(item)
            print(comp_lines[lid])
            print(simp_lines[lid])
            print('=====')
