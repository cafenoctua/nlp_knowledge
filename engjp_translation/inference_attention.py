import numpy as np

class InferenceAPI:

    def __init__(self, encoder_model, decoder_model, en_vocab, ja_vocab):
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model
        self.en_vocab = en_vocab
        self.ja_vocab = ja_vocab

    def predict(self, text):
        output, state = self._compute_encoder_output(text)
        sequence = self._generate_sequence(output, state)
        decoded = self._decode(sequence)
        return decoded
    
    def _compute_encoder_output(self, text):
        x = self.en_vocab.text_to_sequences([text])
        output, state = self.encoder_model.predict(x)
        return output, state

    def _compute_decoder_ouput(self, target_seq, state, enc_output=None):
        output, state = self.decoder_model.predict([target_seq, enc_output, state])
        return output, state

    def _generate_sequence(self, enc_output, state, max_seq_len=50):
        target_seq = np.array([self.ja_vocab.word_index['<start>']])
        sequence = []
        for i in range(max_seq_len):
            output, state = self._compute_decoder_ouput(target_seq, state, enc_output)
            sampled_token_index = np.argmax(output[0, 0])
            if sampled_token_index == self.ja_vocab.word_index['<end>']:
                break
            sequence.append(sampled_token_index)
            target_seq = np.array([sampled_token_index])

        return sequence

    def _decode(self, sequence):
        decoded = self.ja_vocab.sequence_to_texts([sequence])
        decode = decoded[0].split(' ')

if __name__ == '__main__':
    test = InferenceAPI()
    test.predict('<state> Hello! <end>')