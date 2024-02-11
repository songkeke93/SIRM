import os
import pandas as pd
import numpy as np
from sklearn import metrics

def get_max_stage(input_file, output_file):
    stage_list = ["t1d", "t1j", "t1p", "t1y", "t2", "echo1", "echo2", "adc", "b=500"]
    max_stage = []
    max_stage_index = []
    df = pd.read_csv(input_file, index_col=0)
    for i in range(df.shape[0]):
        row = df.iloc[i, :]
        max_stage_index.append(np.argmax(row))
        max_stage.append(stage_list[np.argmax(row)])
    df['stage'] = max_stage
    df['index'] = max_stage_index
    print(df.head())
    df.to_csv(output_file, encoding='utf_8_sig')


def roc_compare():
    model_prob_file_1 = "/workspace/medical/code/data/results/task2/features/label1_prob_.csv"
    model_prob_file_2 = "/workspace/medical/code/data/results/task2/features/label2_prob_.csv"
    output_file = "/workspace/medical/code/data/results/task2/features/label1_prob_subset_2.csv"
    df_model_prob_1 = pd.read_csv(model_prob_file_2, index_col=0)
    df_model_prob_2 = pd.read_csv(model_prob_file_1, index_col=0)
    subset_index = []
    for i in df_model_prob_2.index.values:
        if i in df_model_prob_1.index.values:
            subset_index.append(i)
    print(len(subset_index))
    df_model_prob_1 = df_model_prob_1.loc[subset_index]
    y = df_model_prob_1.loc[:, 'label']
    prob = df_model_prob_1.loc[:, '0':'8'].max(axis=1)
    auc = metrics.roc_auc_score(y, prob)
    fpr, tpr, thresholds = metrics.roc_curve(y, prob)
    # tmp = round(0.683168316831683, 8)
    # print(tmp)
    print(auc)
    print(thresholds)
    index = 5
    sensitivity = tpr[index]
    specificity = 1 - fpr[index]
    print(sensitivity)
    print(specificity)
    df_model_prob_1.to_csv(output_file, encoding='utf_8_sig')


if __name__ == "__main__":
    input_file = "/workspace/medical/code/data/results/task2/features/label2_prob.csv"
    output_file = "/workspace/medical/code/data/results/task2/features/label2_prob_test_with_stage.csv"
    get_max_stage(input_file, output_file)
    roc_compare()