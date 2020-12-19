from dataloader import load_embedding, load_data_news_test, load_index
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
import numpy as np
import time

fasttext = 'fasttext'
glove = 'glove'
random = 'random'
transformer = 'transformer'
embedding_list = [random, fasttext, glove, transformer]

knn = 'knn'
svm = 'svm'
random_forest = 'random_forest'
clf_list = [knn, svm, random_forest]

origin = 'origin'
small = 'small'
unbalance = 'unbalance'
balance = 'balance'
data_type_list = [origin, unbalance, balance, small]


def run_clf(ebd_type, clf_type, index=None, result_suffix=''):
    start = time.time()
    print('start {} with model {}:'.format(ebd_type, clf_type))
    data, label = load_embedding(ebd_type, False, index)
    test_data, test_label = load_embedding(ebd_type, True)
    print('data loaded {}'.format(time.strftime('%Hh %Mm %Ss', time.gmtime(time.time() - start))))
    if clf_type == knn:
        clf = KNeighborsClassifier()
    elif clf_type == svm:
        clf = SVC()
    elif clf_type == random_forest:
        clf = RandomForestClassifier()
    else:
        print('undefined clf: {}'.format(clf_type))
    clf.fit(data, label)
    print('model fitted {}'.format(time.strftime('%Hh %Mm %Ss', time.gmtime(time.time() - start))))
    result = clf.predict(test_data)
    print('model predicted {}'.format(time.strftime('%Hh %Mm %Ss', time.gmtime(time.time() - start))))
    with open(ebd_type + '_embedding/' + clf_type + '_result' + result_suffix + '.txt', 'w') as f:
        f.write(str(list(result)))
    print('end {} with model {}:'.format(ebd_type, clf_type))


def load_result(ebd_type, clf_type, result_suffix=''):
    dir = ebd_type + '_embedding/' + clf_type + '_result' + result_suffix + '.txt'
    with open(dir) as f:
        line = f.readline()
    return list(map(int, line.strip().strip('[').strip(']').split(',')))


def evaluate_f1_score(dtype=''):
    _, label = load_embedding(fasttext, True)
    for clf in clf_list:
        for ebd in embedding_list:
            if dtype != origin:
                predict = load_result(ebd, clf, '_' + dtype)
            else:
                predict = load_result(ebd, clf, '')
            print('type:\t{}\t clf: \t{}\t ebd: \t{}\t f1-score: \t{:.2f}'.format(dtype, clf, ebd,
                                                                                  f1_score(label, predict,
                                                                                           average='macro')))


if __name__ == '__main__':
    # for dtype in data_type_list:
    # dtype = small
    # for clf in clf_list:
    #     for ebd in embedding_list:
    #         run_clf(ebd, clf, load_index(dtype), '_' + dtype)
    # for dt in data_type_list:
    #     evaluate_f1_score(dt)
    ebd, _ = load_embedding(transformer)
    print(len(ebd[0]))
