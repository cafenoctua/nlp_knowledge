from pprint import pprint

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

from inference import InferenceAPI
from models import EmbeddingModel
from preprocessing import build_vocabulary, create_dataset
from utils import load_data

if __name__ == '__main__':
    num_words = 10000
    model_path = 'model.h5'

    # コーパスの読み込み
    text = load_data(filepath='data/ja.text8')

    # ボキャブラリの構築
    vocab = build_vocabulary(text, num_words)

    # 予測
    model = load_model(model_path)
    api = InferenceAPI(model, vocab)
    pprint(api.most_similar(word='日本'))