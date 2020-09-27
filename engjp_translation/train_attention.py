from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from inference import InferenceAPIforAttention
from models_attention import Seq2seq, Encoder, Decoder, AttentionDecoder
from preprocessing import build_vocabulary, preprocess_ja, preprocess_dataset, create_dataset
from utils import load_dataset, evaluate_bleu

def main():
    # ハイパーパラメータの設定
    batch_size = 32
    epochs = 100
    model_path = 'models/attention_model.h5'
    enc_arch = 'models/encoder.json'
    dec_arch = 'models/decoder.json'
    data_path = 'data/jpn.txt'
    num_words = 10000
    num_data = 20000

    # データ・セット読み込み
    en_texts, ja_texts = load_dataset(data_path)
    en_texts, ja_texts = en_texts[:num_data], ja_texts[:num_data]

    # データ・セットの前処理
    ja_texts = preprocess_ja(ja_texts)
    ja_texts = preprocess_dataset(ja_texts)
    en_texts = preprocess_dataset(en_texts)
    x_train, x_test, y_train, y_test = train_test_split(en_texts, ja_texts, test_size=0.2, random_state=42)

    en_vocab = build_vocabulary(x_train, num_words)
    ja_vocab = build_vocabulary(y_train, num_words)
    x_train, y_train = create_dataset(x_train, y_train, en_vocab, ja_vocab)

    # モデルの構築
    encoder = Encoder(num_words, return_sequences=True)
    decoder = AttentionDecoder(num_words)
    seq2seq = Seq2seq(encoder, decoder)
    model = seq2seq.build()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    # コールバックの用意
    callbacks = [
        EarlyStopping(patience=3),
        ModelCheckpoint(model_path, save_best_only=True, save_weights_only=True)
    ]

    # モデルの学習
    model.fit(x=x_train,
            y=y_train,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            validation_split=0.1)
    encoder.save_as_json(enc_arch)
    decoder.save_as_json(dec_arch)

    # 予測
    encoder = Encoder.load(enc_arch, model_path)
    decoder = Decoder.load(dec_arch, model_path)
    api = InferenceAPIforAttention(encoder, decoder, en_vocab, ja_vocab)
    texts = sorted(set(en_texts[:50]), key=len)
    for text in texts:
        decoded = api.predict(text=text)
        print('English : {}'.format(text))
        print('Japanese: {}'.format(decoded))

    # 性能評価
    y_test = [y.split(' ')[1:-1] for y in y_test]
    bleu_score = evaluate_bleu(x_test, y_test, api)
    print('BLEU: {}'.format(bleu_score))
    
if __name__ == '__main__':
    main()