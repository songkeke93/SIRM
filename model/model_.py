import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from utils.data_load_util import data_load
from sklearn import metrics
import joblib
import os


def _model_for_type(
        x_train,
        y_train,
        x_test,
        y_test,
        stage,
        label_type,
        save_path,
        standard=True,
        f_select=True,
        f_selected_path=None):
    if standard:
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
    if f_select:
        f_selected_file_name = "{}_{}.npy".format(stage, label_type)
        f_selected_file_path = os.path.join(f_selected_path, f_selected_file_name)
        if os.path.exists(f_selected_file_path):
            with open(f_selected_file_path, 'rb') as f:
                f_mask = np.load(f)
                x_train = x_train[:, f_mask]
                x_test = x_test[:, f_mask]
        else:
            estimator = LinearSVC(max_iter=10000)
            selector = RFE(estimator=estimator, n_features_to_select=90)
            selector = selector.fit(x_train, y_train.values.ravel())
            x_train = selector.transform(x_train)
            x_test = selector.transform(x_test)
            with open(f_selected_file_path, 'wb') as f:
                np.save(f, selector.support_)
    # 交叉验证
    best_auc = 0
    kf = KFold(n_splits=6)
    for train_index, val_index in kf.split(x_train):
        x_train_kf = x_train[train_index]
        x_val_kf = x_train[val_index]
        y_train_kf = y_train.iloc[train_index]
        y_val_kf = y_train.iloc[val_index]
        clf = RandomForestClassifier(n_estimators=101)
        clf.fit(x_train_kf, y_train_kf.values.ravel())

        pred_proba = clf.predict_proba(x_val_kf)[:, 1]
        val_auc = metrics.roc_auc_score(y_val_kf, pred_proba)
        val_auc = np.abs(0.5 - val_auc)
        if val_auc > best_auc:
            best_auc = val_auc
            pred_proba = clf.predict_proba(x_test)[:, 1]
            test_auc = metrics.roc_auc_score(y_test, pred_proba)
            # print(y_test.values)
            # print(pred_proba)
            print("Auc_Test:{}".format(test_auc))
            model_name = "{}_{}.m".format(stage, label_type)
            model_path = os.path.join(save_path, model_name)
            joblib.dump(clf, model_path)


if __name__ == "__main__":
    train_cls_file_path = "/workspace/medical/code/data/cls/train_task2.csv"
    test_cls_file_path = "/workspace/medical/code/data/cls/test_task2.csv"
    features_file_path = "/workspace/medical/code/data/results/task2/features/fusion_features_2.csv"
    save_path = "/workspace/medical/code/data/results/task2/models"
    f_selected_path = "/workspace/medical/code/data/results/task2/features_selected"
    x_train, df_train, x_test, df_test = data_load(train_cls_file_path, test_cls_file_path, features_file_path, 't1d',
                                                   'group')

    _model_for_type(x_train, df_train, x_test, df_test, 't1d', '2', save_path, True, True, f_selected_path)