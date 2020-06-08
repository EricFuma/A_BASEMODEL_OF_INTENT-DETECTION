import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import os
import sys
import time
import json
import re
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from pprint import pprint
from collections import Counter
os.environ['VISIBLE_CUDA_DEVICES'] = '2'
import argparse



# utils
from get_dataset import get_dataset
from model import Chara_CNN_BGRU
from utils import get_confusion_matrix


parser = argparse.ArgumentParser()
parser.add_argument('--init_data', default='./SMP_2019/train.json', help='file path which store the init dataset')
# 这三个 size 都应该是在构造数据集的时候，人工锁定下来的，这里我是为了统一，直接人工 check 后当默认值了，如果是其他数据集需要进行有针对性的修改哦
parser.add_argument('--vocab_size', default=1548, help='The size of the vocabulary, including padding element')
parser.add_argument('--batch_size', default=32, help='batch size during training and testing')
parser.add_argument('--seq_length', default=27, help='max length of the sentence length (token number)')
parser.add_argument('--embed_size', default=64, help='embedding size of the character vector')

parser.add_argument('--epochs', default=10, help='training epochs')
parser.add_argument('--log_dir', default='./logDir/', help='The directionary which store TensorBoard log')
parser.add_argument('--model_path', default='./savedModel/', help='The directionary which store the best model')
parser.add_argument('--split_path', default='./SMP_2019/split_data/', help='The direcctionary which store the traing & testing data')
args = parser.parse_args()  # jupyter notebook 中使用 argparse 会有一些问题

print(args.init_data)


with open(os.path.join(args.split_path, 'train.json'), 'r', encoding='utf-8') as f:
    train_data = json.load(f)
    X_train, y_train = train_data['X_train'], train_data['y_train']
with open(os.path.join(args.split_path, 'test.json'), 'r', encoding='utf-8') as f:
    test_data = json.load(f)
    X_test, y_test = test_data['X_test'], test_data['y_test']

train_dataset, test_dataset, my_tokenizer, label2id = get_dataset(X_train, y_train, X_test, y_test, args)
print("Get Dataset!")
print('-'*40)

model = Chara_CNN_BGRU(
    args.vocab_size, args.seq_length, args.embed_size)
print("Get Model!")
print(model.summary())
print('-*40')

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy', 'mse'])
callbacks = [
    keras.callbacks.EarlyStopping(patience=3, min_delta=1e-3),
    keras.callbacks.TensorBoard(args.log_dir),
    keras.callbacks.ModelCheckpoint(filepath=args.model_path, save_best_only=True)]
history = model.fit(train_dataset, epochs=args.epochs,
                    validation_data=test_dataset,
                    callbacks=callbacks)

best_model = keras.models.load_model(args.model_path)

# Testing
with open(os.path.join(args.split_path, 'test.json'), 'r', encoding='utf-8') as f:
    test_init_data = json.load(f)
    input_tensors = my_tokenizer.texts_to_sequences(test_init_data['X_test'])
    input_tensors = keras.preprocessing.sequence.pad_sequences(input_tensors, maxlen=args.seq_length)
    y_true = test_init_data['y_test']
    id2label = dict(zip(label2id.values(), label2id.keys()))
    y_true = [label2id[label] for label in y_true]
    res = best_model.predict(input_tensors)
    y_pred = tf.argmax(res, axis=-1).numpy().tolist()
    labels = sorted(list(set(y_true) | set(y_pred)))

    best_model.evaluate(test_dataset)
    # y_eval = tf.argmax(res_eval, axis=-1).numpy().tolist()


label_true = [id2label[idx] for idx in y_true]
label_pred = [id2label[idx] for idx in y_pred]
#label_eval = [id2label[idx] for idx in y_eval]
#real_labels = sorted(list(set(label_true) | set(label_pred)))
print('-'*40)
get_confusion_matrix(label_true, label_pred)
print("Plot confusion matrix Over!")
pred_acc = sum([1 if t == p else 0 for t,p in zip(label_true, label_pred) ]) / len(label_true)
#eval_acc = sum([1 if t == p else 0 for t,p in zip(label_true, label_eval) ]) / len(label_true)
print("PRED ACC:", pred_acc)


 
