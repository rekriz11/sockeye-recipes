from collections import Counter

c = Counter()
# for line in open('/Users/zhaosanqiang916/git/text_simplification_data/train/dress/wikilargenew/train/dst2.txt'):
# for line in open('/Users/zhaosanqiang916/git/text_simplification_data/train/dress/wikilarge/wiki.full.aner.train.src'):
for line in open('/Users/zhaosanqiang916/git/text_simplification_data/train/dress/wikilargenew/test/dst.txt'):
    cnt = len(line.split())
    c.update([str(cnt)])
    if cnt > 40:
        print(line)

mm = 0
for cnt in c:
    if float(cnt) > mm:
        mm = float(cnt)
print(c.most_common())
print()
print(mm)
