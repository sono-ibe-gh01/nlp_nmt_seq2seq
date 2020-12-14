import tensorflow as tf


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        # ★理解しながらコードを記述しましょう．


    def call(self, x, hidden):
        # ★理解しながらコードを記述しましょう．


    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))



