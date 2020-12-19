import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import torch
import time
from transformers import AutoTokenizer, AutoModel

CATEGORY = 'CATEGORY'
TEXT = 'TEXT'
TITLE = 'TITLE'


def load_data_news_train(nrows=None) -> pd.DataFrame:
    data_file = 'data/ag_news_csv/train.csv'
    columns_name = [CATEGORY, TITLE, TEXT]
    file_data = pd.read_csv(data_file, engine='python', header=None, names=columns_name,
                            nrows=nrows)
    file_data[TEXT] = file_data[TITLE] + ' ' + file_data[TEXT]
    file_data = file_data.drop(columns=[TITLE, ])
    file_data[TEXT] = file_data[TEXT].str.replace('[^a-zA-Z0-9]', ' ').str.lower()
    return file_data


def token_to_cuda(tokenizer_output):
    tokens_tensor = tokenizer_output['input_ids'].cuda()
    token_type_ids = tokenizer_output['token_type_ids'].cuda()
    attention_mask = tokenizer_output['attention_mask'].cuda()

    output = {'input_ids': tokens_tensor,
              'token_type_ids': token_type_ids,
              'attention_mask': attention_mask}

    return output


def generate_bert_embedding(data, batchsize=10, log_interval=500):
    start = time.time()
    data_size = data.shape[0]
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained('bert-base-uncased').cuda()
    result = []
    for i in range((data.shape[0] // batchsize) + 1):
        data_batch = data[(i * batchsize):((i + 1) * batchsize)]
        inputs = token_to_cuda(tokenizer(data_batch.tolist(), padding=True, truncation=True, return_tensors='pt'))
        batch_result = model(**inputs).pooler_output.detach().tolist()
        result += batch_result
        if (i + 1) % log_interval == 0:
            with open('bert_embedding\\' + str((i + 1) // log_interval) + '.txt', 'w') as f:
                for line in result:
                    line_format = [round(i, 5) for i in line]
                    f.write(str(line_format))
            print('TIME {}  num timesteps {} processed {:.2f}%'
                  .format(time.strftime('%Hh %Mm %Ss', time.gmtime(time.time() - start)),
                          i + 1,
                          ((i + 1) * 100) / data_size))
            result = []
    return result


def get_distribution(data):
    distribution = {
        'len < 10': 0,
        '10 <= len < 20': 0,
        '20 <= len < 30': 0,
        '30 <= len < 40': 0,
        '40 <= len < 50': 0,
        '50 <= len < 60': 0,
        '60 <= len < 70': 0,
        '70 <= len < 80': 0,
        '80 <= len < 90': 0,
        '90 <= len <100': 0,
        '100<= len': 0
    }
    len_list = []
    for text in data[TEXT]:
        l = len(text.split())
        len_list.append(l)
        if l < 10:
            distribution['len < 10'] += 1
        elif l < 20:
            distribution['10 <= len < 20'] += 1
        elif l < 30:
            distribution['20 <= len < 30'] += 1
        elif l < 40:
            distribution['30 <= len < 40'] += 1
        elif l < 50:
            distribution['40 <= len < 50'] += 1
        elif l < 60:
            distribution['50 <= len < 60'] += 1
        elif l < 70:
            distribution['60 <= len < 70'] += 1
        elif l < 80:
            distribution['70 <= len < 80'] += 1
        elif l < 90:
            distribution['80 <= len < 90'] += 1
        elif l < 100:
            distribution['90 <= len <100'] += 1
        else:
            distribution['100<= len'] += 1
    len_list = np.array(len_list)
    return len_list, distribution


def to_tf_idf(data):
    vectorizer = CountVectorizer()
    x = vectorizer.fit_transform(data[TEXT])
    tf_idf_transformer = TfidfTransformer()
    tf_idf = tf_idf_transformer.fit_transform(x)
    truncated_svd = TruncatedSVD(256)
    svd = truncated_svd.fit_transform(tf_idf)
    standard_scaler = StandardScaler()
    scaled = standard_scaler.fit_transform(svd)
    train_vector = pd.DataFrame(scaled)
    # train_vector.to_csv('data/ag_news_csv/train_vector_tfidf.csv')
    return train_vector


def get_statistics(data):
    len_list, distribution = get_distribution(data)
    total = 0
    for k, v in distribution.items():
        total += v
    # vectorizer = CountVectorizer()
    # vectorizer.fit_transform(data[TEXT])
    print('Total Num: ' + str(total))
    print('Category Num: ')
    print(data[CATEGORY].value_counts())
    print('Max Len: ' + str(len_list.max()))
    print('Min Len: ' + str(len_list.min()))
    print('Average Len: ' + str(len_list.mean()))
    print('Variance: ' + str(len_list.var()))
    print('Standard Deviation: ' + str(len_list.std()))
    print('Distribution: ' + str(distribution))
    # print('Vocabulary Size: ' + str(len(vectorizer.get_feature_names())))

    for i in range(1, 5):
        print('\n')
        print('Category ' + str(i))
        len_list, distribution = get_distribution(data[data[CATEGORY] == i])
        print('Max Len: ' + str(len_list.max()))
        print('Min Len: ' + str(len_list.min()))
        print('Average Len: ' + str(len_list.mean()))
        print('Variance: ' + str(len_list.var()))
        print('Standard Deviation: ' + str(len_list.std()))
        print('Distribution: ' + str(distribution))


data = load_data_news_train()
generate_bert_embedding(data[TEXT])
