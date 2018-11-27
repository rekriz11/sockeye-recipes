from os import listdir
from data_process.data_preprocess import rui_preprocess

base = '/Users/zhaosanqiang916/Desktop/wiki_data/'
nls_comp, nls_simp = [], []
f_comp = open('/Users/zhaosanqiang916/git/text_simplification_data/train/dress/wikihugenew/src.txt', 'w')
f_simp = open('/Users/zhaosanqiang916/git/text_simplification_data/train/dress/wikihugenew/dst.txt', 'w')
files = listdir(base)
checker = set()
for file in files:
    if file.startswith('ner_comp_'):
        comp_file = file
        id = file[len('ner_comp_'):-len('.txt')]
        if id in checker:
            continue
        checker.add(id)
        simp_file = 'ner_simp_' + id + '.txt'

        ls_comp = open(base + comp_file).readlines()
        ls_simp = open(base + simp_file).readlines()
        if len(ls_comp) == len(ls_simp):
            for lid in range(len(ls_comp)):
                l_comp = ' '.join(rui_preprocess(ls_comp[lid]))
                l_simp = ' '.join(rui_preprocess(ls_simp[lid]))
                if l_comp != l_simp:
                    nls_comp.append(l_comp)
                    nls_simp.append(l_simp)
        else:
            print('Warning!!!\t%s\t%s' % (comp_file, simp_file))

        f_comp.write('\n'.join(nls_comp))
        f_simp.write('\n'.join(nls_simp))
        f_comp.flush()
        f_simp.flush()
        nls_comp.clear()
        nls_simp.clear()
        print('Finished File %s.' % file)

f_comp.write('\n'.join(nls_comp))
f_simp.write('\n'.join(nls_simp))
f_comp.close()
f_simp.close()


