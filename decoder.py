import tensorflow as tf
from attention import Attention


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        # ★理解しながらコードを記述しましょう．


    def call(self, x, hidden, enc_output):
        # ★理解しながらコードを記述しましょう．
