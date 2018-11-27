import re

base = '/Users/zhaosanqiang916/git/text_simplification_data/train/dress/wikilarge/train/'
lines_comp = open(base + 'src.txt').readlines()
lines_simp = open(base + 'dst.txt').readlines()
lines_map = open(base + 'rule_mapper.txt').readlines()

assert len(lines_comp) == len(lines_simp) and len(lines_comp) == len(lines_map)

nlines_comp = []
nlines_simp = []
nlines_map = []

for lid in range(len(lines_comp)):
    line_comp = lines_comp[lid].strip().lower().split()
    line_simp = lines_simp[lid].strip().lower().split()
    line_map = lines_map[lid].strip().lower().split()

    # tags_comp = set([w for w in line_comp if '@' in w])
    # tags_simp = set([w for w in line_simp if '@' in w])
    # if len(tags_comp | tags_simp) > 0 and float(len(tags_comp & tags_simp)) / (1.0 + len(tags_comp | tags_simp)) < 0.5:
    #     print('comp=%s\n' % ' '.join(line_comp))
    #     print('simp=%s\n' % ' '.join(line_simp))
    #     print('map=%s\n' % line_map)
    #     print('==========\n')
    #     continue

    if len(line_simp) == 0 or line_simp[-1] != '.' or len(line_simp) < 10:
        continue

    if ''.join(line_simp) == ''.join(line_comp):
        continue

    set_comp = set(line_comp)
    set_simp = set(line_simp)
    if len(set_comp & set_simp) < 5 and len(line_map) == 0:
        continue
    if len(set_comp & set_simp) == 5:
        print('comp=%s\n' % ' '.join(line_comp))
        print('simp=%s\n' % ' '.join(line_simp))
        print('map=%s\n' % line_map)
        print('==========\n')

    nlines_comp.append(lines_comp[lid])
    nlines_simp.append(lines_simp[lid])
    nlines_map.append(lines_map[lid])

f_comp = open(base + 'src2.txt', 'w')
f_simp = open(base + 'dst2.txt', 'w')
f_map = open(base + 'rule_mapper2.txt', 'w')

f_comp.write(''.join(nlines_comp))
f_simp.write(''.join(nlines_simp))
f_map.write(''.join(nlines_map))

f_comp.close()
f_simp.close()
f_map.close()

print('final cnt:%s' % len(nlines_comp))
