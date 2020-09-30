from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from inference import InferenceAPI
from models import RNNModel, LSTMModel, CNNModel
from preprocessing import preprocess_dataset, build_vocabulary
from utils import load_dataset, load_fasttext, filter_embeddings

def main():
    # ハイパーパラメータの背一定
    batch_size = 128
    epochs = 100
    maxlen = 300
    # model_path = 'models/rnn_model.h5'
    # model_path = 'models/lstm_model.h5'
    # model_path = 'models/CNN_model.h5'
    model_path = 'models/latm_iniemb_model.h5'
    num_words = 4000
    num_label = 2

    # データ・セットの読み込み
    x, y = load_dataset('data/amazon_reviews_multilingual_JP_v1_00.tsv')

    # データセットの前処理
    x = preprocess_dataset(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    vocab = build_vocabulary(x_train, num_words)
    x_train = vocab.texts_to_sequences(x_train)
    x_test  = vocab.texts_to_sequences(x_test)
    x_train = pad_sequences(x_train, maxlen=maxlen, truncating='post')
    x_test = pad_sequences(x_test, maxlen=maxlen, truncating='post')

    # 単語分散表現
    wv = load_fasttext('data/cc.ja.300.vec')
    wv = filter_embeddings(wv, vocab.word_index, num_words)

    # モデルの構築
    # model = RNNModel(num_words, num_label, embeddings=None).build()
    model = LSTMModel(num_words, num_label, embeddings=wv).build()
    # model = CNNModel(num_words, num_label, embeddings=None).build()
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['acc'])

    # コールバックの用意
    callbacks = [
        EarlyStopping(patience=3),
        ModelCheckpoint(model_path, save_best_only=True)
    ]

    # モデルの学習
    model.fit(x=x_train,
            y=y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.2,
            callbacks=callbacks,
            shuffle=True)
    
    # 予測
    model = load_model(model_path)
    api = InferenceAPI(model, vocab, preprocess_dataset)
    y_pred = api.predict_from_sequences(x_test)
    print('precision: {:.4f}'.format(precision_score(y_test, y_pred, average='binary')))
    print('recall: {:.4f}'.format(recall_score(y_test, y_pred, average='binary')))
    print('f1: {:.4f}'.format(f1_score(y_test, y_pred, average='binary')))

if __name__=='__main__':
    main()