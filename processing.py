

from main import args
import re
import json
import os
from collections import Counter
from sklearn.model_selection import train_test_split

# 文本清洗模块
def clean_text(text):
    pass

# 中文分字
def seg_char(sent):
    """
    把句子按字分开，不破坏英文结构
    """
    # 首先分割 英文 以及英文和标点
    #pattern_char_1 = re.compile(r'([\W])')
    #parts = pattern_char_1.split(sent)
    #parts = [p for p in parts if len(p.strip())>0]
    #print(parts)
    # 分割中文
    pattern = re.compile(r'([\u4e00-\u9fa5,\s,0-9,\W])')
    chars = pattern.split(sent)
    chars = [w for w in chars if len(w.strip())>0]
    return chars
print(seg_char('oh2333,你知道Batman吗，这是个很有名的角色！'))


# 统计句子长度，并画分布直方图
def length_static_and_plot():
    pass

# 统计类别数目，并画分布直方图
def type_static_and_plot():
    pass

# 划分训练数据和测试数据
def parse_data(init_data):
    text2label_pairs = [(' '.join(seg_char(data['text'])), data['intent']) for data in init_data]
    return zip(*text2label_pairs)


def process_and_split_data():
    with open(args.init_data, 'r', encoding='utf-8') as f:
        init_data = json.load(f)
    texts, labels = map(list, parse_data(init_data))
    print("init data size:", len(texts), len(labels))
    print("sample:", texts[0], '\t', labels[0])
    # 基本的数据扩充
    unique_labels = set(labels)
    for lab in unique_labels:
        if labels.count(lab) < 5:
            txts = [txt for txt,cla in zip(texts, labels) if cla == lab]
            labs = [cla for cla in labels if cla == lab]
            while labels.count(lab) < 5:
                labels += labs
                texts += txts
    print("Argmentation data size: ", len(labels), len(texts))
    # 保留label数据分布的情况下，划分数据集
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2,
                                                        stratify=labels)  # 保证训练集和测试集的 label 不存在 Mistach 问题
    print("training data size: ", len(X_train), len(X_test))
    print("testing data size: ", len(y_train), len(y_test))
    assert set(Counter(y_train).keys()) == set(Counter(y_test).keys())


    # 在测试集中，对小样本数据再进行部分扩充
    X_test += ['我 想 知 道 蚂 蚁 金 服 昨 日 收 盘 价 十 多 少 ？',
            '猎 心 者 这 部 剧 什 么 时 候 能 看 啊',
            '帮 我 下 载 这 个 app',
            '我 想 查 看 短 信']
    y_test += ['CLOSEPRICE_QUERY', 'DATE_QUERY', 'DOWNLOAD', 'VIEW']

    assert not set(Counter(y_train).keys()) - set(Counter(y_test).keys()), print("Mismatch between train dataset and test dataset")
    print("final training size: ", len(X_train), len(X_test))
    print("final testing size: ", len(y_train), len(y_test))

    # 保存划分后的原始数据集
    train_dict = {'X_train':X_train, 'y_train':y_train}
    test_dict = {'X_test':X_test, 'y_test':y_test}
    with open(os.path.join(args.split_path, 'train.json'), 'w', encoding='utf-8') as f:
        json.dump(train_dict, f, ensure_ascii=False, indent=4)

    with open(os.path.join(args.split_path, 'test.json'), 'w', encoding='utf-8') as f:
        json.dump(test_dict, f, ensure_ascii=False, indent=4)

# 数据预处理、划分训练数据、测试数据并写入文件中
process_and_split_data()
