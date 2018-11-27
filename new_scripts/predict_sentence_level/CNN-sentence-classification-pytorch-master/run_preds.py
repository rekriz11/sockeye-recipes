from model_new import CNN
import utils

from torch.autograd import Variable
import torch
import torch.optim as optim
import torch.nn as nn

from sklearn.utils import shuffle
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import argparse
import copy
import math
from scipy.stats.stats import pearsonr

def train(data, params):
    if params["MODEL"] != "rand":
        # load word2vec
        print("loading word2vec...")
        word_vectors = KeyedVectors.load_word2vec_format("/data1/reno/embeddings/GoogleNews-vectors-negative300.bin", binary=True)

        wv_matrix = []
        for i in range(len(data["vocab"])):
            word = data["idx_to_word"][i]
            if word in word_vectors.vocab:
                wv_matrix.append(word_vectors.word_vec(word))
            else:
                wv_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype("float32"))

        # one for UNK and one for zero padding
        wv_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype("float32"))
        wv_matrix.append(np.zeros(300).astype("float32"))
        wv_matrix = np.array(wv_matrix)
        params["WV_MATRIX"] = wv_matrix

    model = CNN(**params).cuda(params["GPU"])

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adadelta(parameters, params["LEARNING_RATE"])
    criterion = nn.MSELoss()

    pre_dev_acc = 10000
    max_dev_acc = 10000
    max_test_acc = 10000
    for e in range(params["EPOCH"]):
        data["train_x"], data["train_y"] = shuffle(data["train_x"], data["train_y"])

        for i in range(0, len(data["train_x"]), params["BATCH_SIZE"]):
            batch_range = min(params["BATCH_SIZE"], len(data["train_x"]) - i)

            batch_x = [[data["word_to_idx"][w] for w in sent] +
                       [params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] - len(sent))
                       for sent in data["train_x"][i:i + batch_range]]
#            batch_y = [data["classes"].index(c) for c in data["train_y"][i:i + batch_range]]
            batch_y = data["train_y"][i:i + batch_range]            

            batch_x = Variable(torch.LongTensor(batch_x)).cuda(params["GPU"])
            batch_y = Variable(torch.FloatTensor(batch_y)).cuda(params["GPU"])

            optimizer.zero_grad()
            model.train()
            pred = model(batch_x)

            '''
            for j in range(len(pred)):
                for k in range(len(pred[j])):
                    print(pred[j][k])
                    pred[j][k] = 1/(1+math.exp(-pred[j][k].item()))
            '''
            
            loss = criterion(pred, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, max_norm=params["NORM_LIMIT"])
            optimizer.step()

        dev_acc = test(data, model, params, mode="dev")
        test_acc = test(data, model, params)
        print("epoch:", e + 1, "/ dev_mse:", dev_acc, "/ test_mse:", test_acc)

        if params["EARLY_STOPPING"] and dev_acc >= pre_dev_acc:
            print("early stopping by dev_mse!")
            break
        else:
            pre_dev_acc = dev_acc

        if dev_acc < max_dev_acc:
            max_dev_acc = dev_acc
            max_test_acc = test_acc
            best_model = copy.deepcopy(model)

    print("min dev mse:", max_dev_acc, "test mse:", max_test_acc)
    return best_model


def test(data, model, params, mode="test"):
    model.eval()

    if mode == "dev":
        x, y = data["dev_x"], data["dev_y"]
    elif mode == "test":
        x, y = data["test_x"], data["test_y"]

    x = [[data["word_to_idx"][w] if w in data["vocab"] else params["VOCAB_SIZE"] for w in sent] +
         [params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] - len(sent))
         for sent in x]

    x = Variable(torch.LongTensor(x)).cuda(params["GPU"])
#    y = [data["classes"].index(c) for c in y]

#    pred = np.argmax(model(x).cpu().data.numpy(), axis=1)
    pred = model(x).cpu().data.numpy()
    
#    acc = sum([1 if p == y else 0 for p, y in zip(pred, y)]) / len(pred)
    acc = sum([(p-y) ** 2 for p,y in zip(pred, y)]) / len(pred)

    return acc

## Finds the error of each level in the test set, along with the overall error;
## also calculates pearson correlation
def test_split(data, model, params):
    model.eval()

    x_dict = dict()
    y_dict = dict()
    for i in range(len(data["test_x"])):
        try:
            x_dict[data["test_y"][i]].append(data["test_x"][i])
        except KeyError:
            x_dict[data["test_y"][i]] = [data["test_x"][i]]

    all_errors = []
    all_y = []
    all_preds = []
    all_x = []

    for i in range(5):
        x = x_dict[i]
        all_x += x
        y = [i for sent in x]

        x = [[data["word_to_idx"][w] if w in data["vocab"] else params["VOCAB_SIZE"] for w in sent] +
         [params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] - len(sent))
         for sent in x]
        x = Variable(torch.LongTensor(x)).cuda(params["GPU"])

        pred = model(x).cpu().data.numpy()
        for i in range(len(pred)):
            if pred[i] > 4.0:
                pred[i] = 4.0
            elif pred[i] < 0.0:
                pred[i] = 0.0

        errors = [(p-y) ** 2 for p,y in zip(pred, y)]

        print("Error for level " + str(i) + ": " + str(round(sum(errors)/len(errors), 3)))
        all_errors += errors
        
        all_preds += list(pred)
        all_y += y

    print("Overall error: " + str(round(sum(all_errors)/len(all_errors), 3)))

    corr = pearsonr(np.asarray(all_y), np.asarray(all_preds))
    print("Overall correlation: " + str(round(corr[0], 4)))

    print(len(all_preds))
    c = 0
    if params["OUTFILE"] != "None":
        with open(params["OUTFILE"], 'w', encoding='utf8') as f:
            while min(all_preds) < 100:
                if c % 1000 == 0:
                    print(c)
                c += 1
                
                i = all_preds.index(min(all_preds))
                
                f.write(" ".join(all_x[i]) + "\t" + str(all_preds[i]) + "\t" + str(all_y[i]) + "\n")
                all_preds[i] = 100

## Gets predicted complexity levels for all candidates
def get_preds(candidates_file, data, model, params):
    ## Gets all candidates from file
    candidates = []
    with open(candidates_file, 'r', encoding='utf8') as f:
        for line in f:
            ls = line[:-1].split("\t")
            cands = [l.split(" ") for l in ls]
            candidates.append(cands)

    all_preds = []
    a = 0

    for x in candidates:
        if a % 100 == 0:
            print(a)
        
        x_new = [[data["word_to_idx"][w] if w in data["vocab"] else params["VOCAB_SIZE"] for w in sent] +
         [params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] - len(sent))
         for sent in x]

        if True:

            for b in range(len(x_new)):
                if len(x_new[b]) > 147:
                    print(x[b])
                    print(len(x[b]))
                    print(params["MAX_SENT_LEN"])
                                    
        x_new = Variable(torch.LongTensor(x_new)).cuda(params["GPU"])
        
        pred = model(x_new).cpu().data.numpy()
        for i in range(len(pred)):
            if pred[i] > 4.0:
                pred[i] = 4.0
            elif pred[i] < 0.0:
                pred[i] = 0.0

        all_preds.append(list(pred))
        a += 1
    return all_preds
    
        

def main():
    parser = argparse.ArgumentParser(description="-----[CNN-classifier]-----")
    parser.add_argument("--mode", default="train", help="train: train (with test) a model / test: test saved models")
    parser.add_argument("--model", default="rand", help="available models: rand, static, non-static, multichannel")
    parser.add_argument("--datafile", default="None", help="data file base to read in different datset (needs training, valid, and test files)")
    parser.add_argument("--dataset", default="TREC", help="available datasets: MR, TREC")
    parser.add_argument("--save_model", default=False, action='store_true', help="whether saving model or not")
    parser.add_argument("--early_stopping", default=False, action='store_true', help="whether to apply early stopping")
    parser.add_argument("--epoch", default=100, type=int, help="number of max epoch")
    parser.add_argument("--learning_rate", default=1.0, type=float, help="learning rate")
    parser.add_argument("--gpu", default=-1, type=int, help="the number of gpu to be used")
    parser.add_argument("--cands", default="None", help="candidate outputs file")
    parser.add_argument("--outfile", default="None", help="output file to write complexity predictions to")

    
    options = parser.parse_args()
    if options.datafile == "None":
        data = getattr(utils, f"read_{options.dataset}")()
    else:
        data = utils.read_other(options.datafile)

#    data["vocab"] = sorted(list(set([w for sent in data["train_x"] + data["dev_x"] + data["test_x"] for w in sent])))
    data["vocab"] = sorted(list(set([w for sent in data["train_x"] for w in sent])))
#    data["classes"] = sorted(list(set(data["train_y"])))
    data["word_to_idx"] = {w: i for i, w in enumerate(data["vocab"])}
    data["idx_to_word"] = {i: w for i, w in enumerate(data["vocab"])}

    params = {
        "MODEL": options.model,
        "DATASET": options.dataset,
        "DATAFILE": options.datafile,
        "SAVE_MODEL": options.save_model,
        "EARLY_STOPPING": options.early_stopping,
        "EPOCH": options.epoch,
        "LEARNING_RATE": options.learning_rate,
        "MAX_SENT_LEN": max([len(sent) for sent in data["train_x"] + data["dev_x"] + data["test_x"]]),
        "BATCH_SIZE": 50,
        "WORD_DIM": 300,
        "VOCAB_SIZE": len(data["vocab"]),
#        "CLASS_SIZE": len(data["classes"]),
        "FILTERS": [3, 4, 5],
        "FILTER_NUM": [100, 100, 100],
        "DROPOUT_PROB": 0.5,
        "NORM_LIMIT": 3,
        "GPU": options.gpu,
        "OUTFILE": options.outfile
    }

    print("=" * 20 + "INFORMATION" + "=" * 20)
    print("MODEL:", params["MODEL"])
    if options.datafile == "None":
        print("DATASET:", params["DATASET"])
    else:
        print("DATAFILE:", params["DATAFILE"])
    print("VOCAB_SIZE:", params["VOCAB_SIZE"])
    print("EPOCH:", params["EPOCH"])
    print("LEARNING_RATE:", params["LEARNING_RATE"])
    print("EARLY_STOPPING:", params["EARLY_STOPPING"])
    print("SAVE_MODEL:", params["SAVE_MODEL"])
    print("=" * 20 + "INFORMATION" + "=" * 20)

    if options.mode == "train":
        print("=" * 20 + "TRAINING STARTED" + "=" * 20)
        model = train(data, params)
        if params["SAVE_MODEL"]:
            utils.save_model(model, params)
        print("=" * 20 + "TRAINING FINISHED" + "=" * 20)
    else:
        candidates_file = options.cands
        model = utils.load_model(params).cuda(params["GPU"])
        preds = get_preds(candidates_file, data, model, params)

        print(preds[0])
        print(len(preds))
        print(len(preds[0]))

        with open(options.outfile, 'w', encoding='utf8') as f:
            for ps in preds:
                f.write("\t".join([str(p) for p in ps]) + "\n")


if __name__ == "__main__":
    main()

'''
python3 run_preds.py \
--mode test \
--model non-static \
--datafile /data1/reno/newsela_splits/newsela_sents \
--epoch 20 \
--gpu 0 \
--cands ~/sockeye-recipes/egs/pretrained_embeddings/output/tokens.100beam.20centroids \
--outfile ~/sockeye-recipes/egs/pretrained_embeddings/output/tokens.100beam.20centroids.preds \
--early_stopping
'''
