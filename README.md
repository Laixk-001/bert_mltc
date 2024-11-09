# bert_mltc
Multi-Label Tweets Classification：多标签文本分类模型，本项目使用bert-base-uncased模型实现多标签文本分类任务。

## 数据集
使用的数据集为英语论文数据集，该数据集给出论文的标题（TITLE）和摘要（ABSTRACT），来预测论文属于哪个主题。该数据集有20972个训练样本，有六个主题，分别为：Computer Science, Physics, Mathematics, Statistics, Quantitative Biology, Quantitative Finance。

## 文件结构
- scr
    - bert-base-uncased
        - config.json
    - data
        - test.csv
        - train.csv
    - load_data.py
    - model.py
    - model_train.py
    - params.py
