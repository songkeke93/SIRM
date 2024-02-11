import os
import numpy as np
import pandas as pd
import joblib
from utils.data_load_util import data_load
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import matplotlib.pyplot as plt
import pdb
plt.style.use('science')


def get_model_prob(root_path, output_file, target_label_type, f_select=False):
    save_path = "/workspace/medical/code/data/results/task2/models"
    features_path = "/workspace/medical/code/data/results/task2/features"
    train_cls_file_path = "/workspace/medical/code/data/cls/train_task2.csv"
    test_cls_file_path = "/workspace/medical/code/data/cls/test_task2.csv"
    features_selected_save_path = "/workspace/medical/code/data/results/task2/features_selected"
    stage_dict = {"t1d": 0, "t1j": 1, "t1p": 2, "t1y": 3, "t2": 4, "echo1": 5, "echo2": 6, "adc":7, "b=500": 8}

    results = {}
    path_quene = [root_path]
    while len(path_quene) > 0:
        current_path = path_quene.pop(0)
        if 'img' in os.listdir(current_path):
            stage_name = current_path.split('/')[-1]
            rois_path = os.path.join(current_path, 'rois')
            for item in os.listdir(rois_path):

                label_type = item[:item.find('.nii.gz')]
                if int(label_type) != target_label_type:
                    continue
                else:
                    model_name = "{}_{}.m".format(stage_name, label_type)
                    model_path = os.path.join(save_path, model_name)
                    model = joblib.load(model_path)
                    features_file_name = "{}_{}.csv".format(stage_name, label_type)
                    # features_file_name = "{}_{}.csv".format('fusion_features', label_type)
                    features_file_path = os.path.join(features_path, features_file_name)
                    x_train, df_train, x_test, df_test = data_load(train_cls_file_path, test_cls_file_path,
                                                                   features_file_path, stage_name,
                                                                   'group')
                    scaler = StandardScaler()
                    scaler.fit(x_train)
                    if f_select:
                        features_selected_file_path = os.path.join(features_selected_save_path, "{}_{}.npy".format(stage_name, label_type))
                        with open(features_selected_file_path, 'rb') as f:
                            f_mask = np.load(f)
                        # x_test = x_test.loc[:, f_mask]
                    tmp_path = current_path[:current_path.rfind('/')]
                    featureVector_name = tmp_path.replace(root_path, '').replace('/', '-')
                    if featureVector_name + "-{}".format(stage_name) in x_test.index.values.ravel():
                        x_vector = scaler.transform([x_test.loc[featureVector_name + "-{}".format(stage_name)].values])
                        if f_select:
                            x_vector = x_vector[:, f_mask]
                        if featureVector_name in results.keys():
                            results[featureVector_name][stage_dict[stage_name]] = model.predict_proba(x_vector)[:, 1]
                        else:
                            featureVector = np.zeros(9)
                            results[featureVector_name] = featureVector
                            results[featureVector_name][stage_dict[stage_name]] = model.predict_proba(x_vector)[:, 1]
        else:
            for name in os.listdir(current_path):
                path_quene.append(os.path.join(current_path, name))

    lab = df_test.loc[["{}-{}".format(k, stage_name) for k in results.keys()], 'group'].values
    print(lab)
    results = pd.DataFrame(results)
    results = results.T
    results['label'] = lab
    auc = metrics.roc_auc_score(lab, results.loc[:, 0])
    print(auc)
    if results.shape[0] > 0:
        results.to_csv(output_file, encoding='utf_8_sig')


def get_model_prob_train(root_path, output_file, target_label_type):
    save_path = "/workspace/medical/code/data/results/task2/models"
    features_path = "/workspace/medical/code/data/results/task2/features"
    train_cls_file_path = "/workspace/medical/code/data/cls/train_task2.csv"
    test_cls_file_path = "/workspace/medical/code/data/cls/test_task2.csv"
    stage_dict = {"t1d": 0, "t1j": 1, "t1p": 2, "t1y": 3, "t2": 4, "echo1": 5, "echo2": 6, "adc":7, "b=500": 8}

    results = {}
    path_quene = [root_path]
    while len(path_quene) > 0:
        current_path = path_quene.pop(0)
        if 'img' in os.listdir(current_path):
            stage_name = current_path.split('/')[-1]
            rois_path = os.path.join(current_path, 'rois')
            for item in os.listdir(rois_path):
                label_type = item[:item.find('.nii.gz')]
                if int(label_type) != target_label_type:
                    continue
                else:
                    model_name = "{}_{}.m".format(stage_name, label_type)
                    model_path = os.path.join(save_path, model_name)
                    model = joblib.load(model_path)
                    features_file_name = "{}_{}.csv".format(stage_name, label_type)
                    features_file_path = os.path.join(features_path, features_file_name)
                    x_train, df_train, x_test, df_test = data_load(train_cls_file_path, test_cls_file_path,
                                                                   features_file_path, stage_name,
                                                                   'group')
                    scaler = StandardScaler()
                    scaler.fit(x_train)
                    tmp_path = current_path[:current_path.rfind('/')]
                    featureVector_name = tmp_path.replace(root_path, '').replace('/', '-')
                    if featureVector_name + "-{}".format(stage_name) in x_train.index.values.ravel():
                        if featureVector_name in results.keys():
                            results[featureVector_name][stage_dict[stage_name]] = model.predict_proba(
                                scaler.transform([x_train.loc[featureVector_name + "-{}".format(stage_name)].values]))[:, 1]
                        else:
                            featureVector = np.zeros(9)
                            results[featureVector_name] = featureVector
                            results[featureVector_name][stage_dict[stage_name]] = model.predict_proba(
                                scaler.transform([x_train.loc[featureVector_name + "-{}".format(stage_name)].values]))[:, 1]
        else:
            for name in os.listdir(current_path):
                path_quene.append(os.path.join(current_path, name))
    results = pd.DataFrame(results)
    results = results.T
    if results.shape[0] > 0:
        results.to_csv(output_file, encoding='utf_8_sig')


def integration_max(model_prob_file, test_cls_file_path):
    df_test_cls = pd.read_csv(test_cls_file_path, index_col=0)
    df_model_prob = pd.read_csv(model_prob_file, index_col=0)
    index = []
    for item in df_model_prob.index:
        if item in df_test_cls.index:
            index.append(item)
    df_test_cls = df_test_cls.loc[index]
    for i in range(9):
        auc = metrics.roc_auc_score(df_test_cls, df_model_prob.iloc[:, i])
        print(auc)
    max_prob = df_model_prob.max(axis=1)
    auc = metrics.roc_auc_score(df_test_cls, max_prob)
    print(auc)


if __name__ == "__main__":
    root_path = "storage/workspace/work/code/data/translate/task2/"
    output_file = "/workspace/medical/code/data/results/task2/features/label2_prob_fusion.csv"
    test_cls_file_path = "/workspace/medical/code/data/cls/test_task2.csv"
    get_model_prob(root_path, output_file, 2, True)
    # integration_max(output_file, test_cls_file_path)
    # plot_roc(output_file, test_cls_file_path)
    # get_model_prob_train(root_path, output_file, 2)