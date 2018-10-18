import sys
from os import listdir
from os.path import isfile, join, exists
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
import pickle
from scipy.stats.stats import pearsonr

def load_features(file):
    words = []
    y = []
    X = []

    with open(file, 'r', encoding='utf8') as f:
        for line in f:
            ls = line[:-1].split('\t')
            words.append(ls[0])
            y.append(float(ls[1]))
            X.append([float(x) for x in ls[2:]])

    X_np = np.asarray(X)
    means = np.mean(X_np, axis=0)
    stds = np.std(X_np, axis=0)


    X_scaled = []
    for i in range(len(X)):
        Xs = []
        for j in range(len(X[i])):
            Xs.append((X[i][j] - means[j])/stds[j])
        X_scaled.append(Xs)
    X_scaled_np = np.asarray(X_scaled)

    print(X_scaled_np.shape)
    Y = np.asarray(y)
    
    return words, X_scaled_np, Y, means, stds

def train(X, Y, model_type, a):
    if model_type == "Logistic":
        clf = LogisticRegression()
    elif model_type == "Ridge":
        clf = Ridge(alpha=a)
    else:
        clf = LinearRegression()
    clf.fit(X, Y)
    return clf

def save_classifier(clf, means, stds, regression_file):
    with open(regression_file, 'wb') as f:
        pickle.dump([clf, means, stds], f)
        

def main(feature_file, model_type, regression_file, alpha):
    words, X, Y, means, stds = load_features(feature_file)
    clf = train(X, Y, model_type, alpha)
    save_classifier(clf, means, stds, regression_file)
        
if __name__ == '__main__':
    feature_file = sys.argv[1]
    model_type = sys.argv[2]
    regression_file = sys.argv[3]

    alpha = 0.0
    if len(sys.argv) > 4:
        alpha = float(sys.argv[4])
    
    main(feature_file, model_type, regression_file, alpha)

## Linear Regression on weighted updated data
'''
python3 train_linear_regression.py \
/scratch-shared/users/reno/text_simplification/Text_Simplification_CLEAN/Simple_for_Whom/Predict_Word_Level/Intermediate_Data/all_features_we3_weight.txt \
Linear \
/scratch-shared/users/reno/text_simplification/Text_Simplification_CLEAN/Simple_for_Whom/Predict_Word_Level/Models/lin_reg_test_we3_weight_NEW.pkl
'''
