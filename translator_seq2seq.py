import tensorflow as tf


from sklearn.model_selection import train_test_split

import os
import time

from encoder import Encoder
from decoder import Decoder
from attention import Attention
from dataset import *

############################################################
# パラメータ設定
NUM_EXAMPLES = 500          # 利用するサンプル数を設定（～62000）
BATCH_SIZE = 64             # ミニバッチのサイズ
DIM_EMBEDDING = 256         # 埋込み表現の次元数
DIM_HIDDEN = 1024           # 隠れ状態の次元数（ユニット数）
EPOCHS = 15                 # 訓練時の繰返し回数


############################################################
# 訓練関係の関数

# 損失関数を定義
def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


# 訓練を行う関数train_stepをオーバーライド
@tf.function
def train_step(src, tgt, enc_hidden):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(src, enc_hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([lang_tgt.word_index['<bos>']] * BATCH_SIZE, 1)

        for t in range(1, tgt.shape[1]):
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
            loss += loss_function(tgt[:, t], predictions)
            # Teacher Forcing ... 正解値を次の入力として供給
            dec_input = tf.expand_dims(tgt[:, t], 1)

        batch_loss = (loss / int(tgt.shape[1]))

        variables = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss


# 訓練
def train():
    for epoch in range(EPOCHS):
        start = time.time()
        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0

        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(inp, targ, enc_hidden)
            total_loss += batch_loss

            if batch % 100 == 0:
                print('Epoch: {}, Batch: {}, Loss: {:.4f}'.format(
                    epoch + 1, batch, batch_loss.numpy()))

        # 2 epoch毎にモデルを保存
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix=ckeckpoint_prefix)

        print('Epoch: {}, Loss: {:.4f}'.format(
            epoch + 1, total_loss / steps_per_epoch))
        print('Time taken for 1 epoch: {:.2f} sec\n'.format(
            time.time() - start))


############################################################
# テスト関係の関数

# 推論
def evaluate(sentence):
    # ★理解しながらコードを記述しましょう．


# 翻訳
def translate(sentence):
    result, sentence = evaluate(sentence)

    print('Input: %s' % (sentence))
    print('Predicted translation: {}'.format(result))


############################################################
# Main:

# データセットをファイルから読み込み
path_to_file = "./eng-jpn_seg.txt"
eng, jpn = create_dataset(path_to_file, None)

# データセットを作成
tensor_src, tensor_tgt, lang_src, lang_tgt \
    = load_dataset(path_to_file, NUM_EXAMPLES)

# ターゲットテンソルの最大長を計算
max_length_tgt, max_length_src \
    = max_length(tensor_tgt), max_length(tensor_src)

# 80%:20%で分割を行い，訓練用と検証用のデータセットを作成
tensor_src_train, tensor_src_valid, tensor_tgt_train, tensor_tgt_valid \
    = train_test_split(tensor_src, tensor_tgt, test_size=0.2)

# パラメータ設定
buffer_size = len(tensor_src_train)
steps_per_epoch = len(tensor_src_train) // BATCH_SIZE
vocab_size_src = len(lang_src.word_index) + 1
vocab_size_tgt = len(lang_tgt.word_index) + 1

# サンプルをシャッフル
dataset = tf.data.Dataset.from_tensor_slices(
            (tensor_src_train, tensor_tgt_train)).shuffle(buffer_size)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

# モデルの構成要素を定義
encoder = Encoder(vocab_size_src, DIM_EMBEDDING, DIM_HIDDEN, BATCH_SIZE)
attention_layer = Attention(10)
decoder = Decoder(vocab_size_tgt, DIM_EMBEDDING, DIM_HIDDEN, BATCH_SIZE)

# オプティマイザを定義
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True, reduction='none')

# チェックポイント（モデルを保存するファイル）を定義
checkpoint_dir = './training_checkpoints'
ckeckpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(
                optimizer=optimizer, encoder=encoder, decoder=decoder)

# チェックポイントが存在しない場合は訓練を実行
if not os.path.exists(checkpoint_dir):
    train()

# 最後のチェックポイントを復元し，テストを実施
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# いろいろ翻訳してみよう
translate('this is a pen .')
translate('hello .')
translate('thank you .')
translate('welcome to my home .')
