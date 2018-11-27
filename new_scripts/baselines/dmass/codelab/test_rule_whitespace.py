for line in open('/Users/zhaosanqiang916/git/text_simplification_data/train/dress/wikilargenew/train/rule_voc.txt'):
    line = line.strip()
    pair = line.split('\t')[0].split('=>')
    ori = pair[0].split()
    tar = pair[1].split()
    if len(ori) > 1 and len(tar) > 1:
        print(line)


# from collections import Counter
# c = Counter()
# for line in open('/Users/zhaosanqiang916/git/text_simplification_data/train/dress/wikilargenew/train/rule_mapper2.txt'):
#     rules = line.strip().split('\t')
#     for rule in rules:
#         items = rule.split('=>')
#         if len(items) != 4:
#             continue
#         r = items[1] + '=>'+ items[2]
#         c.update([r])
#
# rules = []
# for rule, cnt in c.most_common():
#     rules.append('%s\t%s' % (rule, str(cnt)))
#
# f = open('/Users/zhaosanqiang916/git/text_simplification_data/train/dress/wikilargenew/train/' + 'rule_voc2.txt', 'w')
# f.write('\n'.join(rules))
# f.close()