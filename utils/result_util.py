import os
import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from utils.data_load_util import data_load_task1
from sklearn.preprocessing import StandardScaler
# plt.style.use('science')

def _get_importance_of_selected_features(model_file_path, features_file_path, save_path, selected_file_path=None):
    name = model_file_path.split('/')[-1].split('.')[0]
    model = joblib.load(model_file_path)
    df_feature = pd.read_csv(features_file_path, index_col=0)
    if not selected_file_path is None:
        with open(selected_file_path, 'rb') as f:
            f_mask = np.load(f)
        model_features = df_feature.columns.values[f_mask]
    else:
        model_features = df_feature.columns.values
    model_features_importances = model.feature_importances_
    top_10_features_index = model_features_importances.argsort()[::-1][:10]
    top_10_features_importances = model_features_importances[top_10_features_index]
    top_10_features_name = model_features[top_10_features_index]
    # print(top_10_features_importances)
    # print(top_10_features_name)
    df_importance = pd.DataFrame({'name': top_10_features_name,
                                  "importance": top_10_features_importances})
    output_file = os.path.join(save_path, "{}.csv".format(name))
    df_importance.to_csv(output_file, encoding='utf_8_sig', index=False)

    top_10_features_importances = top_10_features_importances / max(top_10_features_importances) * 100
    fig, ax = plt.subplots()
    fig.set_figwidth(8)
    fig.set_figheight(12)
    fig.subplots_adjust(left=0.2, right=0.9, top=0.99, bottom=0.8)
    # plt.tight_layout()
    plt.bar(range(len(top_10_features_importances)), top_10_features_importances)
    plt.xticks(range(len(top_10_features_importances)), top_10_features_name, rotation=90)
    plt.gcf().subplots_adjust(bottom=0.45)
    plt.ylabel(u'Relative variable importance(%)')
    output_file = os.path.join(save_path, "{}.png".format(name))
    plt.savefig(output_file)
    plt.close()


def get_importance_of_selected_features(model_root_path, feature_root_path, save_path, selected_root_path=None):
    stage_list = ["t1d", "t1j", "t1p", "t1y", "t2", "echo1", "echo2", "adc", "b=500"]
    label_type_list = ['1', '2']
    for stage_name in stage_list:
        for label_type in label_type_list:
            model_file_path = os.path.join(model_root_path, "{}_{}.m".format(stage_name, label_type))
            features_file_path = os.path.join(feature_root_path, "{}_{}.csv".format(stage_name, label_type))
            if not selected_root_path is None:
                selected_file_path = os.path.join(selected_root_path, "{}_{}.npy".format(stage_name, label_type))
            else:
                selected_file_path = None
            _get_importance_of_selected_features(model_file_path, features_file_path, save_path, selected_file_path)


def plot_roc(model_prob_file, test_cls_file_path):
    stage_list = ["t1d", "t1j", "t1p", "t1y", "t2", "echo1", "echo2", "adc", "b=500"]
    df_test_cls = pd.read_csv(test_cls_file_path, index_col=0)
    df_model_prob = pd.read_csv(model_prob_file, index_col=0)
    index = []
    for item in df_model_prob.index:
        if item in df_test_cls.index:
            index.append(item)
    df_test_cls = df_test_cls.loc[index]
    fig, axs = plt.subplots(nrows=3, ncols=3, constrained_layout=True)
    fig.set_figheight(12)
    fig.set_figwidth(12)
    for i in range(9):
        fpr, tpr, thresholds = metrics.roc_curve(df_test_cls, df_model_prob.iloc[:, i])
        auc = metrics.roc_auc_score(df_test_cls, df_model_prob.iloc[:, i])
        ax = axs[int(i / 3), i % 3]
        ax.plot(fpr, tpr, 'g--', label='AUC=%0.3f' % auc, lw=2)
        ax.legend(loc="lower right")
        ax.plot([-0.05, 1.05], [-0.05, 1.05], '--')

        ax.set_xlabel(u'1 - Specificity')
        ax.set_ylabel(u'Sensitivity')
        ax.set_title('ROC of {}'.format(stage_list[i].upper()))
    plt.savefig("stage.png")
    plt.close()

    max_prob = df_model_prob.max(axis=1)
    auc = metrics.roc_auc_score(df_test_cls, max_prob)
    print(auc)
    fpr, tpr, thresholds = metrics.roc_curve(df_test_cls, max_prob)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, 'g--', label='AUC=%0.3f' % auc, lw=2)
    plt.legend(loc="lower right")
    plt.plot([-0.05, 1.05], [-0.05, 1.05], '--')

    plt.xlabel(u'1 - Specificity')
    plt.ylabel(u'Sensitivity')
    plt.title('ROC')
    plt.savefig("max.png")


def get_roc_threshold(model_prob_file):
    df_model_prob = pd.read_csv(model_prob_file, index_col=0)
    y = df_model_prob.loc[:, 'label']
    prob = df_model_prob.loc[:, '0':'8'].max(axis=1)
    auc = metrics.roc_auc_score(y, prob)
    print(auc)
    fpr, tpr, thresholds = metrics.roc_curve(y, prob)
    # calculate youden index
    youden_list = []
    for i, j in zip(fpr, tpr):
        youden = j - i
        youden_list.append(youden)
    youden_max = max(youden_list)
    index = youden_list.index(youden_max)
    print(f'the best threshold is {thresholds[index]} where sensitivity = {tpr[index]} specificity = {1 - fpr[index]}')


def get_prob_task1():
    cls_file_path = "/workspace/medical/code/data/cls/task1.csv"
    features_file_path = "/workspace/medical/code/data/results/task1/features/fusion_features_1.csv"
    model_path = "/workspace/medical/code/data/results/task1/models"
    f_selected_path = "/workspace/medical/code/data/results/task1/features_selected"

    model_0 = joblib.load(os.path.join(model_path, '0.m'))
    model_1 = joblib.load(os.path.join(model_path, '1.m'))

    x_train, df_train, x_test, df_test_0 = data_load_task1(cls_file_path, features_file_path, '0')
    _, _, x_test_bak, df_test_1 = data_load_task1(cls_file_path, features_file_path, '1')
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    f_selected_file_name = "{}.npy".format('0')
    f_selected_file_path = os.path.join(f_selected_path, f_selected_file_name)
    with open(f_selected_file_path, 'rb') as f:
        f_mask = np.load(f)
        x_train_0 = x_train[:, f_mask]
        x_test_0 = x_test[:, f_mask]
    f_selected_file_name = "{}.npy".format('1')
    f_selected_file_path = os.path.join(f_selected_path, f_selected_file_name)
    with open(f_selected_file_path, 'rb') as f:
        f_mask = np.load(f)
        x_train_1 = x_train[:, f_mask]
        x_test_1 = x_test[:, f_mask]
    pred_proba_0 = model_0.predict_proba(x_test_0)[:, 1]
    auc = metrics.roc_auc_score(df_test_0, pred_proba_0)
    print(auc)

    pred_proba_1 = model_1.predict_proba(x_test_1)[:, 1]
    auc = metrics.roc_auc_score(df_test_1, pred_proba_1)
    print(auc)
    result = pd.DataFrame({'a_prob': pred_proba_0,
                           'a_label': df_test_0,
                           'b_prob': pred_proba_1,
                           'b_label': df_test_1,
                           })
    out = "/workspace/medical/code/data/results/task1/features/result.csv"
    result.to_csv(out, encoding='utf_8_sig')


def roc_task1():
    result_path = "/workspace/medical/code/data/results/task1/features/result.csv"
    save_path = "/workspace/medical/code/data/results/task1/roc"
    output_file_a = os.path.join(save_path, 'a.png')
    output_file_b = os.path.join(save_path, 'b.png')
    df_result = pd.read_csv(result_path, index_col=0)

    # metric a
    fpr, tpr, thresholds = metrics.roc_curve(df_result.loc[:, 'a_label'], df_result.loc[:, 'a_prob'])
    auc = metrics.roc_auc_score(df_result.loc[:, 'a_label'], df_result.loc[:, 'a_prob'])
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, 'g--', label='AUC=%0.3f' % auc, lw=2)
    plt.legend(loc="lower right")
    plt.plot([-0.05, 1.05], [-0.05, 1.05], '--')

    plt.xlabel(u'1 - Specificity')
    plt.ylabel(u'Sensitivity')
    plt.title('ROC')
    plt.savefig(output_file_a)
    plt.close()
    # youden
    youden_list = []
    for i, j in zip(fpr, tpr):
        youden = j - i
        youden_list.append(youden)
    youden_max = max(youden_list)
    index = youden_list.index(youden_max)
    print(f'the best threshold is {thresholds[index]} where sensitivity = {tpr[index]} specificity = {1 - fpr[index]}')
    thresholds_a = thresholds[index]

    # metric b
    fpr, tpr, thresholds = metrics.roc_curve(df_result.loc[:, 'b_label'], df_result.loc[:, 'b_prob'])
    auc = metrics.roc_auc_score(df_result.loc[:, 'b_label'], df_result.loc[:, 'b_prob'])
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, 'g--', label='AUC=%0.3f' % auc, lw=2)
    plt.legend(loc="lower right")
    plt.plot([-0.05, 1.05], [-0.05, 1.05], '--')

    plt.xlabel(u'1 - Specificity')
    plt.ylabel(u'Sensitivity')
    plt.title('ROC')
    plt.savefig(output_file_b)
    plt.close()

    # youden
    youden_list = []
    for i, j in zip(fpr, tpr):
        youden = j - i
        youden_list.append(youden)
    youden_max = max(youden_list)
    index = youden_list.index(youden_max)
    index = 8
    print(f'the best threshold is {thresholds[index]} where sensitivity = {tpr[index]} specificity = {1 - fpr[index]}')
    thresholds_b = thresholds[index]

    # final
    TN = 0
    TP = 0
    FN = 0
    FP = 0
    for i in df_result.index:
        prob_a = df_result.loc[i, 'a_prob']
        prob_b = df_result.loc[i, 'b_prob']
        pred_a = 1 if prob_a >= thresholds_a else 0
        pred_b = 1 if prob_b >= thresholds_b else 0
        if pred_a == 0 and pred_b == 1:
            pred_final = 1
        else:
            pred_final = 0
        label_a = df_result.loc[i, 'a_label']
        label_b = df_result.loc[i, 'b_label']
        if label_a == 0 and label_b == 1:
            label_final = 1
        else:
            label_final = 0
        if label_final == 0 and pred_final == 0:
            TN += 1
        elif label_final == 0 and pred_final == 1:
            FP += 1
        elif label_final == 1 and pred_final == 1:
            TP += 1
        else:
            FN += 1
    print("final sensitivity {}".format(TP /(TP + FN)))
    print("final specificity {}".format(TN / (TN + FP)))

if __name__ == "__main__":
    # ROC
    output_file = "/workspace/medical/code/data/results/task2/features/label1_prob.csv"
    test_cls_file_path = "/workspace/medical/code/data/cls/test_task2.csv"
    plot_roc(output_file, test_cls_file_path)

    # Feature importance
    # model_file_path = "/workspace/medical/code/data/results/task2/models"
    # selected_file_path = "/workspace/medical/code/data/results/task2/features_selected"
    # features_file_path = "/workspace/medical/code/data/results/task2/features"
    # save_path = "/workspace/medical/code/data/results/task2/features_importance"
    # get_importance_of_selected_features(model_file_path, features_file_path, save_path)

    # single feature importance
    # model_file_path = "/workspace/medical/code/data/results/task2/models/t1d_1.m"
    # selected_file_path = "/workspace/medical/code/data/results/task2/features_selected/t1d_1.npy"
    # features_file_path = "/workspace/medical/code/data/results/task2/features/fusion_features_1.csv"
    # save_path = "/workspace/medical/code/data/results/task2/features_importance"
    # _get_importance_of_selected_features(model_file_path, features_file_path, save_path, selected_file_path=selected_file_path)

    # youden index
    # model_prob_file = "/workspace/medical/code/data/results/task2/features/label1_prob_.csv"
    # get_roc_threshold(model_prob_file)

    # task1
    # get_prob_task1()
    # single feature importance
    # model_file_path = "/workspace/medical/code/data/results/task1/models/0.m"
    # selected_file_path = "/workspace/medical/code/data/results/task1/features_selected/0.npy"
    # features_file_path = "/workspace/medical/code/data/results/task1/features/fusion_features_1.csv"
    # save_path = "/workspace/medical/code/data/results/task1/features_importance"
    # _get_importance_of_selected_features(model_file_path, features_file_path, save_path, selected_file_path=selected_file_path)
    # roc_task1()