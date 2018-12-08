import sys
from scipy.stats.stats import pearsonr
import numpy as np

## Reads in data from file
def get_data(file):
	sents = []
	lengths = []
	labels = []

	with open(file, 'r', encoding='utf8') as f:
		for line in f:
			ls = line[:-1].split("\t")
			sents.append(ls[0])
			labels.append(float(ls[1]))

			lengths.append(len(ls[0].split(" ")))
	return sents, labels, lengths

## Gets maximum and minimum values from list
def get_max_min(train_list):
	return max(train_list), min(train_list)

## Gets predictions based on max and min values from training data
def get_preds_lengths(test_list, maxy, miny):
	preds = []
	for i, s in enumerate(test_list):
		pred = 4 * ((s - miny) / (maxy - miny))
		if pred < 0.0:
			preds.append(0.0)
		elif pred > 4.0:
			preds.append(4.0)
		else:
			preds.append(pred)
	return preds

## Calculates pearson correlation
def get_pearson(X, Y):
    corr = pearsonr(np.asarray(X), np.asarray(Y))
    print("CORRELATION: " + str(round(corr[0], 3)))

def calc_stats(labels, preds):
	get_pearson(labels, preds)
	mses = []

	for c in range(5):
		ys = [preds[i] for i in range(len(labels)) if labels[i] == c]
		error = [(y - c) ** 2 for y in ys]
		mses.append(str(round(sum(error)/len(error), 2)))

	all_error = [(preds[i] - labels[i]) ** 2 for i in range(len(labels))]
	all_mse = str(round(sum(all_error)/len(all_error), 2))

	print(all_mse + "\t" + "\t".join(mses))


def main(train_file, test_file):
	train_words, train_labels, train_lengths = get_data(train_file)
	test_words, test_labels, test_lengths = get_data(test_file)

	## Gets predictions based on length
	maxl, minl = get_max_min(train_lengths)
	predsl = get_preds_lengths(test_lengths, maxl, minl)

	print("#### LENGTH STATS ####")
	calc_stats(test_labels, predsl)


if __name__ == '__main__':
	train_file = sys.argv[1]
	test_file = sys.argv[2]

	main(train_file, test_file)


'''
python3 baseline.py \
/data1/reno/newsela_splits/newsela_sents_old_train.txt \
/data1/reno/newsela_splits/newsela_sents_old_test.txt
'''
