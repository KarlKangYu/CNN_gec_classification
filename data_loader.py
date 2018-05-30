import numpy as np
import codecs
import re

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def data_loader(edited_data, right_data):
    x_text = list()
    y_text = list()
    y = list()
    l = 0
    with codecs.open(edited_data, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            wrong, right = line.split('\t')
            # wrong = clean_str(wrong)
            # right = clean_str(right)
            x_text.append(wrong)
            y_text.append(right)
            y.append(0)
            if len(wrong.split(' ')) > l:
                l = len(wrong.split(' '))

    with codecs.open(right_data, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            wrong = line
            right = line
            x_text.append(wrong)
            y_text.append(right)
            y.append(1)

    return x_text, y_text, y, l


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            batch_data = shuffled_data[start_index:end_index]
            x_text, y, y_text = zip(*batch_data)
            mask_y_text = np.ones([len(y_text), len(y_text[0])])
            mask_y_text[[i == 1 for i in y], :] = 0
            mask_y_text[y_text == 0] = 0
            batch_data = zip(x_text, y, y_text, mask_y_text)
            yield batch_data









