from gensim.models.word2vec import Word2Vec, Text8Corpus

model = Word2Vec.load('models/model.bin')
print('-'*50)
print(model['猫'])
print(model['猫'].shape)
print('-'*50)
print(model.most_similar('猫', topn=10))
print('-'*50)
print(model.most_similar(positive=['ロンドン', '日本'], negative=['東京'], topn=10))
print('-'*50)
print(model.similarity('猫', '犬'))
print('-'*50)
print(model.similarity('猫', '車'))
print('-'*50)
print(model.similarity('車', 'セダン'))

