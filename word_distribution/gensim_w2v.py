import logging
from gensim.models.word2vec import Word2Vec, Text8Corpus

logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)

sentences = Text8Corpus('data/ja.text8')

model = Word2Vec(sentences, size=100, window=5, sg=1)

model.save('models/model.bin')