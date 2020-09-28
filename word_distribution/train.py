from pprint import pprint

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

from inference import InferenceAPI
from models import EmbeddingModel
from preprocessing import build_vocabulary, create_dataset
from utils import load_data

if __name__ == '__main__':
    # ハイパーパラメータの設定
    emb_dim = 50
    epochs = 10
    model_path = 'model.h5'
    negative_samples = 1
    num_words = 10000
    window_size = 1

    # コーパスの読み込み
    text = load_data(filepath='data/ja.text8')

    # ボキャブラリの構築
    vocab = build_vocabulary(text, num_words)

    # データ・セットの作成
    x, y = create_dataset(text, vocab, num_words, window_size, negative_samples)

    # モデルの構築
    model = EmbeddingModel(num_words, emb_dim)
    model = model.build()
    model.compile(optimizer='adam', loss='binary_crossentropy')

    # コールバックの用意
    callbacks = [
        EarlyStopping(patience=1),
        ModelCheckpoint(model_path, save_best_only=True)
    ]

    # モデルの学習
    model.fit(x=x,
            y=y,
            batch_size=128,
            epochs=epochs,
            validation_split=0.2,
            callbacks=callbacks)

    # 予測
    model = load_model(model_path)
    api = InferenceAPI(model, vocab)
    pprint(api.most_similar(word='日本'))