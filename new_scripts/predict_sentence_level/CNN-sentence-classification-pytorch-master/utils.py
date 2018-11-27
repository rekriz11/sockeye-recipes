from sklearn.utils import shuffle

import pickle


def read_TREC():
    data = {}

    def read(mode):
        x, y = [], []

        with open("data/TREC/TREC_" + mode + ".txt", "r", encoding="utf-8") as f:
            for line in f:
                if line[-1] == "\n":
                    line = line[:-1]
                y.append(line.split()[0].split(":")[0])
                x.append(line.split()[1:])

        x, y = shuffle(x, y)

        if mode == "train":
            dev_idx = len(x) // 10
            data["dev_x"], data["dev_y"] = x[:dev_idx], y[:dev_idx]
            data["train_x"], data["train_y"] = x[dev_idx:], y[dev_idx:]
        else:
            data["test_x"], data["test_y"] = x, y

    read("train")
    read("test")

    return data


def read_MR():
    data = {}
    x, y = [], []

    with open("data/MR/rt-polarity.pos", "r", encoding="utf-8") as f:
        for line in f:
            if line[-1] == "\n":
                line = line[:-1]
            x.append(line.split())
            y.append(1)

    with open("data/MR/rt-polarity.neg", "r", encoding="utf-8") as f:
        for line in f:
            if line[-1] == "\n":
                line = line[:-1]
            x.append(line.split())
            y.append(0)

    x, y = shuffle(x, y)
    dev_idx = len(x) // 10 * 8
    test_idx = len(x) // 10 * 9

    data["train_x"], data["train_y"] = x[:dev_idx], y[:dev_idx]
    data["dev_x"], data["dev_y"] = x[dev_idx:test_idx], y[dev_idx:test_idx]
    data["test_x"], data["test_y"] = x[test_idx:], y[test_idx:]

    return data

## ADDED CODE: Gets data splits for other datasets
## Required: File with each line of the form: <SENTENCE><\t><LEVEL><\n>
def get_data_split(data_file):
    x = []
    y = []
    with open(data_file, 'r', encoding='utf8') as f:
        for line in f:
            ls = line[:-1].split("\t")
            x.append([l.lower() for l in ls[0].split(" ")])
            y.append(float(ls[1]))
    return x, y

## ADDED CODE: Gets data into the required dictionary format
def read_other(data_base):
    trainx, trainy = get_data_split(data_base + "_train.txt")
    trainx, trainy = shuffle(trainx, trainy)
    
    devx, devy = get_data_split(data_base + "_valid.txt")
    devx, devy = shuffle(devx, devy)
    
    testx, testy = get_data_split(data_base + "_test.txt")
    testx, testy = shuffle(testx, testy)

    data = dict()
    data['train_x'] = trainx
    data['train_y'] = trainy
    data['dev_x'] = devx
    data['dev_y'] = devy
    data['test_x'] = testx
    data['test_y'] = testy

    return data

def save_model(model, params):
    if params['DATAFILE'] == "None":
        path = f"saved_models/{params['DATASET']}_{params['MODEL']}_{params['EPOCH']}.pkl"
        pickle.dump(model, open(path, "wb"))
        print(f"A model is saved successfully as {path}!")
    else:
        data_base = params["DATAFILE"].split("/")[-1]
        path = f"saved_models/{data_base}_{params['MODEL']}_{params['EPOCH']}.pkl"
        pickle.dump(model, open(path, "wb"))
        print(f"A model is saved successfully as {path}!")


def load_model(params):
    if params['DATAFILE'] == "None":
        path = f"saved_models/{params['DATASET']}_{params['MODEL']}_{params['EPOCH']}.pkl"
    else:
        data_base = params["DATAFILE"].split("/")[-1]
        path = f"saved_models/{data_base}_{params['MODEL']}_{params['EPOCH']}.pkl"

    try:
        model = pickle.load(open(path, "rb"))
        print(f"Model in {path} loaded successfully!")

        return model
    except:
        print(f"No available model such as {path}.")
        exit()
