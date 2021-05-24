from __future__ import print_function
import json
import numpy as np
from tqdm import tqdm
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import SpTokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, open
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder
from keras.models import Model
from rouge import Rouge  # pip install rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# 基本参数
max_c_len = 256

# 模型路径
config_path = './mt5/mt5_small/mt5_small_config.json'
checkpoint_path = './mt5/mt5_small/model.ckpt-1000000'
spm_path = './mt5/sentencepiece_cn.model'
keep_tokens_path = './mt5/sentencepiece_cn_keep_tokens.json'


def load_data(source, target):
    D = []
    f1 = open(source, encoding='utf-8')
    f2 = open(target, encoding='utf-8')
    for content, title in zip(f1, f2):
        D.append((title, content))
    return D


class AutoTitle(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        c_encoded = inputs[0]
        return decoder.predict([c_encoded, output_ids])[:, -1]

    def generate(self, text, topk=1):
        c_token_ids, _ = tokenizer.encode(text, maxlen=max_c_len)
        c_encoded = encoder.predict(np.array([c_token_ids]))[0]
        output_ids = self.beam_search([c_encoded], topk)  # 基于beam search
        return tokenizer.decode([int(i) for i in output_ids])


class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉输入部分
    """
    def compute_loss(self, inputs, mask=None):
        y_true, y_pred = inputs
        y_true = y_true[:, 1:]  # 目标token_ids
        y_mask = K.cast(mask[1], K.floatx())[:, :-1]  # 解码器自带mask
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss


test_data = load_data("dataset/test1.source", "dataset/test1.target")


if __name__ == '__main__':
    # 加载分词器
    tokenizer = SpTokenizer(spm_path, token_start=None, token_end='</s>')
    keep_tokens = json.load(open(keep_tokens_path))

    t5 = build_transformer_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        keep_tokens=keep_tokens,
        model='t5.1.1',
        return_keras_model=False,
        name='T5',
    )

    encoder = t5.encoder
    decoder = t5.decoder
    model = t5.model
    model.summary()

    # output = CrossEntropy(1)([model.inputs[1], model.outputs[0]])
    #
    # model = Model(model.inputs, output)
    # model.compile(optimizer=Adam(2e-4))

    model.load_weights('./save/best_model_39.weights')

    # 注：T5有一个很让人不解的设置，它的<bos>标记id是0，即<bos>和<pad>其实都是0
    autotitle = AutoTitle(start_id=0, end_id=tokenizer._token_end_id, maxlen=128)
    topk = 1

    ans = open("./result/summary_39.txt", "a+", encoding='utf-8')
    for title, content in tqdm(test_data):
        title = ' '.join(title).lower()
        pred_title = ' '.join(autotitle.generate(content, topk)).lower()
        content = content.replace(" ", "")
        content = content.replace("\n", "")
        title = title.replace(" ", "")
        title = title.replace("\n", "")
        pred_title = pred_title.replace(" ", "")
        pred_title = pred_title.replace("\n", "")
        ans.write(content+"\n"+title+"\n"+pred_title+"\n")