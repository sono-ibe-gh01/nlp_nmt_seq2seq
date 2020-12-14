import tensorflow as tf


import unicodedata
import re
import io


# ユニコードファイルをasciiに変換
def _unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def preprocess_sentence_eng(w):
    w = _unicode_to_ascii(w.lower().strip())
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    w = w.rstrip().strip()

    # 文の開始と終了のトークンを付加
    w = '<bos> ' + w + ' <eos>'
    return w


def preprocess_sentence_jpn(w):
    # 文の開始と終了のトークンを付加
    w = '<bos> ' + w + ' <eos>'
    return w


def _tokenize(lang):
    # インデックス割り当て
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)        # ボキャブラリを更新

    tensor = lang_tokenizer.texts_to_sequences(lang)     # インデックスの系列に変換
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')  # 長さを揃える

    return tensor, lang_tokenizer


# アクセント記号を除去，文をクリーニング，単語ペアを作成
def create_dataset(path, num_examples):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')

    word_pairs = []
    for line in lines[:num_examples]:
        snt_eng, snt_jpn = line.split('\t')
        snt_eng = preprocess_sentence_eng(snt_eng)
        snt_jpn = preprocess_sentence_jpn(snt_jpn)
        word_pairs.append([snt_eng, snt_jpn])
    return zip(*word_pairs)


def load_dataset(path, num_examples=None):
    # クリーニングされた入力と出力のペアを生成
    lang_src, lang_tgt = create_dataset(path, num_examples)

    tensor_src, tokenizer_src = _tokenize(lang_src)
    tensor_tgt, tokenizer_tgt = _tokenize(lang_tgt)

    return tensor_src, tensor_tgt, tokenizer_src, tokenizer_tgt


def max_length(tensor):
    return max(len(t) for t in tensor)



