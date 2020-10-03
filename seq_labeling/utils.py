import numpy as np
from seqeval.metrics import classification_report

def load_dataset(filename, encoding='utf-8'):
    sents, labels = [], []
    words, tags = [], []
    with open(filename, encoding=encoding) as f:
        for line in f:
            line = line.rstrip()
            if line:
                word, tag = line.split('\t')
                words.append(word)
                tags.append(tag)
            else:
                sents.append(words)
                labels.append(tags)
                words, tags = [], []
        if words:
            sents.append(words)
            labels.append(tags)

    return sents, labels

def evaluate(model, target_vocab, features, labels):
    label_ids = model.predict(features)
    label_ids = np.argmax(label_ids, axis=-1)
    y_pred = [[] for _ in range(label_ids.shape[0])]
    y_true = [[] for _ in range(label_ids.shape[0])]
    for i in range(label_ids.shape[0]):
        for j in range(label_ids.shape[1]):
            if labels[i][j] == 0:
                continue
            y_pred[i].append(label_ids[i][j])
            y_true[i].append(labels[i][j])
    y_pred = target_vocab.decode(y_pred)
    y_true = target_vocab.decode(y_true)
    print(classification_report(y_true, y_pred, digits=4))