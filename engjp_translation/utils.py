from collections import defaultdict
from nltk.translate.bleu_score import corpus_bleu

def load_dataset(filename):
    en_texts = []
    ja_texts = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            en_text, ja_text = line.strip().split('\t')[:2]
            en_texts.append(en_text)
            ja_texts.append(ja_text)
        
    return en_texts, ja_texts

def evaluate_bleu(X, y, api):
    d = defaultdict(list)
    for source, target in zip(X, y):
        d[source].append(target)
    hypothesis = []
    references = []
    for source, targets in d.items():
        pred = api.predict(source)
        hypothesis.append(pred)
        references.append(targets)
    blue_score = corpus_bleu(references, hypothesis)
    return blue_score