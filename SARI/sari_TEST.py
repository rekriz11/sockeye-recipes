from __future__ import division
from collections import Counter
import numpy as np
from argparse import ArgumentParser

def ReadInFile(filename):
    with open(filename, 'r', encoding='utf8') as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]
    return lines


def SARIngram(sgrams, cgrams, rgramslist, numref):
    rgramsall = [rgram for rgrams in rgramslist for rgram in rgrams]
    rgramcounter = Counter(rgramsall)

    sgramcounter = Counter(sgrams)
    sgramcounter_rep = Counter()
    for sgram, scount in sgramcounter.items():
        sgramcounter_rep[sgram] = scount * numref

    cgramcounter = Counter(cgrams)
    cgramcounter_rep = Counter()
    for cgram, ccount in cgramcounter.items():
        cgramcounter_rep[cgram] = ccount * numref
        
    print("COMPLEX SENTENCE: " + str(sgramcounter_rep))
    print(rgramcounter)
    print(cgramcounter_rep)

    # KEEP
    keepgramcounter_rep = sgramcounter_rep & cgramcounter_rep
    keepgramcountergood_rep = keepgramcounter_rep & rgramcounter
    keepgramcounterall_rep = sgramcounter_rep & rgramcounter

    print("KEEP...")
    print(keepgramcounter_rep)
    print(keepgramcountergood_rep)
    print(keepgramcounterall_rep)

    keeptmpscore1 = 0
    keeptmpscore2 = 0
    for keepgram in keepgramcountergood_rep:
        keeptmpscore1 += keepgramcountergood_rep[keepgram] / keepgramcounter_rep[keepgram]
        keeptmpscore2 += keepgramcountergood_rep[keepgram] / keepgramcounterall_rep[keepgram]
        # print "KEEP", keepgram, keepscore, cgramcounter[keepgram], sgramcounter[keepgram], rgramcounter[keepgram]
    keepscore_precision = 0
    if keeptmpscore1 == 0 and len(keepgramcounter_rep) == 0:
        keepscore_precision = 1
    elif len(keepgramcounter_rep) > 0:
        keepscore_precision = keeptmpscore1 / len(keepgramcounter_rep)
    keepscore_recall = 0 
    if keeptmpscore2 == 0 and len(keepgramcounterall_rep) == 0:
        keepscore_recall = 1
    elif len(keepgramcounterall_rep) > 0:
        keepscore_recall = keeptmpscore2 / len(keepgramcounterall_rep)
    keepscore = 0
    if keepscore_precision > 0 or keepscore_recall > 0:
        keepscore = 2 * keepscore_precision * keepscore_recall / (keepscore_precision + keepscore_recall)

    # DELETION
    delgramcounter_rep = sgramcounter_rep - cgramcounter_rep
    delgramcountergood_rep = delgramcounter_rep - (rgramcounter - cgramcounter_rep)
    delgramcounterall_rep = sgramcounter_rep - rgramcounter

    print("DELETION...")
    print(delgramcounter_rep)
    print(delgramcountergood_rep)
    print(delgramcounterall_rep)
    
    deltmpscore1 = 0
    deltmpscore2 = 0
    for delgram in delgramcountergood_rep:
        deltmpscore1 += delgramcountergood_rep[delgram] / delgramcounter_rep[delgram]
        deltmpscore2 += delgramcountergood_rep[delgram] / delgramcounterall_rep[delgram]
    delscore_precision = 0
    if deltmpscore1 == 0 and len(delgramcounter_rep) == 0:
        delscore_precision = 1
    elif len(delgramcounter_rep) > 0:
        delscore_precision = deltmpscore1 / len(delgramcounter_rep)
        
    delscore_recall = 0
    if deltmpscore2 == 0 and len(delgramcounterall_rep) == 0:
        delscore_recall = 1
    elif len(delgramcounterall_rep) > 0:
        delscore_recall = deltmpscore2 / len(delgramcounterall_rep)
        
    delscore = 0
    if delscore_precision > 0 or delscore_recall > 0:
        delscore = 2 * delscore_precision * delscore_recall / (delscore_precision + delscore_recall)

    print(delscore_precision)
    print(delscore_recall)


    # ADDITION
    addgramcounter = set(cgramcounter) - set(sgramcounter)
    addgramcountergood = set(addgramcounter) & set(rgramcounter)
    addgramcounterall = set(rgramcounter) - set(sgramcounter)

    print("ADDITION...")
    print(addgramcounter)
    print(addgramcountergood)
    print(addgramcounterall)

    addtmpscore = 0
    for addgram in addgramcountergood:
        addtmpscore += 1
    addscore_precision = 0
    if addtmpscore == 0 and len(addgramcounter) == 0:
        addscore_precision = 1
    elif len(addgramcounter) > 0:
        addscore_precision = addtmpscore / len(addgramcounter)
        
    addscore_recall = 0
    if addtmpscore == 0 and len(addgramcounterall) == 0:
        addscore_recall = 1
    elif len(addgramcounterall) > 0:
        addscore_recall = addtmpscore / len(addgramcounterall)
        
    addscore = 0
    if addscore_precision > 0 or addscore_recall > 0:
        addscore = 2 * addscore_precision * addscore_recall / (addscore_precision + addscore_recall)

    return (keepscore, delscore_precision, addscore)


def SARIsent(ssent, csent, rsents):
    numref = len(rsents)

    s1grams = ssent.lower().split(" ")
    c1grams = csent.lower().split(" ")
    s2grams = []
    c2grams = []
    s3grams = []
    c3grams = []
    s4grams = []
    c4grams = []

    r1gramslist = []
    r2gramslist = []
    r3gramslist = []
    r4gramslist = []
    for rsent in rsents:
        r1grams = rsent.lower().split(" ")
        r2grams = []
        r3grams = []
        r4grams = []
        r1gramslist.append(r1grams)
        for i in range(0, len(r1grams) - 1):
            if i < len(r1grams) - 1:
                r2gram = r1grams[i] + " " + r1grams[i + 1]
                r2grams.append(r2gram)
            if i < len(r1grams) - 2:
                r3gram = r1grams[i] + " " + r1grams[i + 1] + " " + r1grams[i + 2]
                r3grams.append(r3gram)
            if i < len(r1grams) - 3:
                r4gram = r1grams[i] + " " + r1grams[i + 1] + " " + r1grams[i + 2] + " " + r1grams[i + 3]
                r4grams.append(r4gram)
        r2gramslist.append(r2grams)
        r3gramslist.append(r3grams)
        r4gramslist.append(r4grams)

    for i in range(0, len(s1grams) - 1):
        if i < len(s1grams) - 1:
            s2gram = s1grams[i] + " " + s1grams[i + 1]
            s2grams.append(s2gram)
        if i < len(s1grams) - 2:
            s3gram = s1grams[i] + " " + s1grams[i + 1] + " " + s1grams[i + 2]
            s3grams.append(s3gram)
        if i < len(s1grams) - 3:
            s4gram = s1grams[i] + " " + s1grams[i + 1] + " " + s1grams[i + 2] + " " + s1grams[i + 3]
            s4grams.append(s4gram)

    for i in range(0, len(c1grams) - 1):
        if i < len(c1grams) - 1:
            c2gram = c1grams[i] + " " + c1grams[i + 1]
            c2grams.append(c2gram)
        if i < len(c1grams) - 2:
            c3gram = c1grams[i] + " " + c1grams[i + 1] + " " + c1grams[i + 2]
            c3grams.append(c3gram)
        if i < len(c1grams) - 3:
            c4gram = c1grams[i] + " " + c1grams[i + 1] + " " + c1grams[i + 2] + " " + c1grams[i + 3]
            c4grams.append(c4gram)

    (keep1score, del1score, add1score) = SARIngram(s1grams, c1grams, r1gramslist, numref)
    print("UNIGRAM SCORES:")
    print(keep1score)
    print(del1score)
    print(add1score)

    (keep2score, del2score, add2score) = SARIngram(s2grams, c2grams, r2gramslist, numref)
    print("BIGRAM SCORES:")
    print(keep2score)
    print(del2score)
    print(add2score)

    (keep3score, del3score, add3score) = SARIngram(s3grams, c3grams, r3gramslist, numref)
    print("TRIGRAM SCORES:")
    print(keep3score)
    print(del3score)
    print(add3score)

    (keep4score, del4score, add4score) = SARIngram(s4grams, c4grams, r4gramslist, numref)
    print("4-gram SCORES:")
    print(keep4score)
    print(del4score)
    print(add4score)

    avgkeepscore = sum([keep1score,keep2score,keep3score,keep4score])/4
    avgdelscore = sum([del1score, del2score, del3score, del4score]) / 4
    avgaddscore = sum([add1score, add2score, add3score, add4score]) / 4
    finalscore = (avgkeepscore + avgdelscore + avgaddscore) / 3
    print("FINAL SCORES:")
    print(avgkeepscore)
    print(avgdelscore)
    print(avgaddscore)
    print(finalscore)

    return finalscore


def main():
    parser = ArgumentParser()
    parser.add_argument("-c", "--complex", dest="complex",
                        help="complex sentences", metavar="FILE")
    parser.add_argument("-r", "--reference", dest="reference",
                        help="reference sentences", metavar="FILE")
    parser.add_argument("-s", "--simplified", dest="simplified",
                        help="simplified sentences", metavar="FILE")

    args = parser.parse_args()

    complex_sentences = ReadInFile(args.complex)[:1]
    reference_sentences = ReadInFile(args.reference)[:1]
    simplified_sentences = ReadInFile(args.simplified)[:1]

    sari_scores = list()
    for i in range(len(simplified_sentences)):
        sari_scores.append(SARIsent(complex_sentences[i], simplified_sentences[i], [reference_sentences[i]]))
        print("\n\n")

    sari_scores = np.array(sari_scores)
    print('SARI score: {}'.format(np.mean(sari_scores)))

if __name__ == '__main__':
    main()
