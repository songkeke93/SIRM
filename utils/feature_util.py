import os
import numpy as np
import pandas as pd
from radiomics import featureextractor
import logging
import joblib
from utils.data_load_util import data_load
from sklearn.preprocessing import StandardScaler


logger = logging.getLogger("radiomics.glcm")
logger.setLevel(logging.ERROR)


def extract_feature(root_path, config_path, output_file, stage_name, label_type):
    params_path = os.path.abspath(config_path)
    extractor = featureextractor.RadiomicsFeatureExtractor(params_path)
    results = pd.DataFrame()
    path_quene = [root_path]
    while len(path_quene) > 0:
        current_path = path_quene.pop(0)
        if 'img' in os.listdir(current_path):
            if stage_name not in current_path:
                continue
            img_path = os.path.join(current_path, 'img')
            img_path = os.path.join(img_path, 'img.nii.gz')
            label_path = os.path.join(current_path, 'rois')
            label_path = os.path.join(label_path, '{}.nii.gz'.format(label_type))
            if os.path.exists(label_path):
                try:
                    featureVector = pd.Series(extractor.execute(img_path, label_path))
                    featureVector.name = current_path.replace(root_path, '').replace('/', '-')
                    results = results.join(featureVector, how='outer')
                except Exception:
                    print(label_path)
        else:
            for name in os.listdir(current_path):
                path_quene.append(os.path.join(current_path, name))
    results = results.T
    drop_dp = results.filter(regex=('diagnostics.*'))
    results = results.drop(drop_dp.columns, axis=1)
    if results.shape[0] > 0:
        results.to_csv(output_file, encoding='utf_8_sig')


def selena_extract_feature():
    stage_list = ["t1d", "t1j", "t1p", "t1y", "t2", "echo1", "echo2", "adc", "b=500"]
    # stage_list = ["echo1", "echo2", "adc", "b=500"]
    task1_label_type = ["1"]
    # task2_label_type = ["1", "2", "3", "4", "5", "6", "7", "8"]
    task2_label_type = ["1", "2"]
    root_path = "D:/freelance/medical/20230701-aftersale/code/data/image/2023_75MRI_translate"
    config_path = "D:/freelance/medical/20230701-aftersale/code/configs/Params.yaml"
    out_path = "D:/freelance/medical/20230701-aftersale/code/data/results/task1/features_230722"
    label_type_list = task1_label_type
    for stage_name in stage_list:
        for label_type in label_type_list:
            print("Extract feature for Stage: {}, Label_type: {}".format(stage_name, label_type))
            out_file_name = "{}_{}.csv".format(stage_name, label_type)
            out_file_path = os.path.join(out_path, out_file_name)
            extract_feature(root_path, config_path, out_file_path, stage_name, label_type)


def _label_count(root_path, stage_name, label_type):
    count = 0
    path_quene = [root_path]
    while len(path_quene) > 0:
        current_path = path_quene.pop(0)
        if 'img' in os.listdir(current_path):
            if stage_name not in current_path:
                continue
            img_path = os.path.join(current_path, 'img')
            img_path = os.path.join(img_path, 'img.nii.gz')
            label_path = os.path.join(current_path, 'rois')
            label_path = os.path.join(label_path, '{}.nii.gz'.format(label_type))
            if os.path.exists(label_path):
                count += 1
        else:
            for name in os.listdir(current_path):
                path_quene.append(os.path.join(current_path, name))
    return count


def selena_label_count():
    result = {}
    stage_list = ["t1d", "t1j", "t1p", "t1y", "t2", "echo1", "echo2", "adc", "b=500"]
    task1_label_type = ["1"]
    task2_label_type = ["1", "2", "3", "4", "5", "6", "7", "8"]
    root_path = "/storage/workspace/work/code/data/translate/task2"
    config_path = "/workspace/medical/code/configs/Params.yaml"
    out_path = "/workspace/medical/code/data/results/task2/features"
    label_type_list = task2_label_type
    for stage_name in stage_list:
        for label_type in label_type_list:
            count = _label_count(root_path, stage_name, label_type)
            result["{}_{}".format(stage_name, label_type)] = [count]

    df_result = pd.DataFrame(result)
    df_result = df_result.T
    df_result.to_csv("/workspace/medical/code/data/results/task2/features_count.csv", encoding='utf_8_sig')


def extract_model_feature(root_path, output_file, stage_name):
    save_path = "/workspace/medical/code/data/results/task2/models"
    features_path = "/workspace/medical/code/data/results/task2/features"

    train_cls_file_path = "/workspace/medical/code/data/cls/train_task2.csv"
    test_cls_file_path = "/workspace/medical/code/data/cls/test_task2.csv"

    results = pd.DataFrame()
    path_quene = [root_path]
    while len(path_quene) > 0:
        current_path = path_quene.pop(0)
        if 'img' in os.listdir(current_path):
            if stage_name not in current_path:
                continue
            rois_path = os.path.join(current_path, 'rois')
            featureVector = np.zeros(8)
            for item in os.listdir(rois_path):
                label_type = item[:item.find('.nii.gz')]
                if int(label_type) == 9:
                    featureVector[6] = 1
                # elif int(label_type) == 1 or int(label_type) == 5 or int(label_type) == 6:
                elif int(label_type) == 1:
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
                    # x_train_s = scaler.transform(x_train)
                    # x_test_s = scaler.transform(x_test)
                    featureVector_name = current_path.replace(root_path, '').replace('/', '-')
                    # print(x_train.index.values.ravel())

                    if featureVector_name in x_train.index.values.ravel():
                        # print([x_train.loc[featureVector_name].values])
                        featureVector[int(label_type) - 1] = model.predict_proba(scaler.transform([x_train.loc[featureVector_name].values]))[:, 1]
                    elif featureVector_name in x_test.index.values.ravel():
                        featureVector[int(label_type) - 1] = model.predict_proba(scaler.transform([x_test.loc[featureVector_name].values]))[:, 1]
                else:
                    pass
                    # featureVector[int(label_type) - 1] = 1
            featureVector = pd.Series(featureVector)
            featureVector.name = current_path.replace(root_path, '').replace('/', '-')
            results = results.join(featureVector, how='outer')
        else:
            for name in os.listdir(current_path):
                path_quene.append(os.path.join(current_path, name))
    results = results.T
    results['label'] = x_test.loc[results.index.values]
    if results.shape[0] > 0:
        results.to_csv(output_file, encoding='utf_8_sig')


def extract_cls_feature(root_path, output_file, stage_name):
    results = pd.DataFrame()
    path_quene = [root_path]
    while len(path_quene) > 0:
        current_path = path_quene.pop(0)
        if 'img' in os.listdir(current_path):
            if stage_name not in current_path:
                continue
            rois_path = os.path.join(current_path, 'rois')
            featureVector = np.zeros(9)
            for item in os.listdir(rois_path):
                label_type = item[:item.find('.nii.gz')]
                featureVector[int(label_type) - 1] = 1
            featureVector = pd.Series(featureVector)
            tmp_path = current_path[:current_path.rfind('/')]
            featureVector.name = tmp_path.replace(root_path, '').replace('/', '-')
            results = results.join(featureVector, how='outer')
        else:
            for name in os.listdir(current_path):
                path_quene.append(os.path.join(current_path, name))
    results = results.T
    if results.shape[0] > 0:
        results.to_csv(output_file, encoding='utf_8_sig')

def label_count(root_path, output_file):
    results = {}
    path_quene = [root_path]
    while len(path_quene) > 0:
        current_path = path_quene.pop(0)
        if 'img' in os.listdir(current_path):
            rois_path = os.path.join(current_path, 'rois')
            tmp_path = current_path[:current_path.rfind('/')]
            featureVector_name = tmp_path.replace(root_path, '').replace('/', '-')
            if featureVector_name in results.keys():
                for item in os.listdir(rois_path):
                    label_type = item[:item.find('.nii.gz')]
                    results[featureVector_name][int(label_type) - 1] = 1
            else:
                featureVector = np.zeros(9)
                for item in os.listdir(rois_path):
                    label_type = item[:item.find('.nii.gz')]
                    featureVector[int(label_type) - 1] = 1
                results[featureVector_name] = featureVector
        else:
            for name in os.listdir(current_path):
                path_quene.append(os.path.join(current_path, name))
    results = pd.DataFrame(results)
    results = results.T
    if results.shape[0] > 0:
        results.to_csv(output_file, encoding='utf_8_sig')


def features_fusion(features_root_path, label_type):
    stage_list = ["t1d", "t1j", "t1p", "t1y", "t2", "echo1", "echo2", "adc", "b=500"]
    # task1
    stage_list = ["t1d", "t1j", "t1p", "t1y", "t2", "echo1", "echo2"]
    file_path = os.path.join(features_root_path, "{}_{}.csv".format(stage_list[0], label_type))
    df = pd.read_csv(file_path, index_col=0)
    patient_index_list = df.index.values
    single_columns = df.columns.values
    final_columns = []
    for stage_name in stage_list:
        for name in single_columns:
            final_columns.append("{}_{}".format(stage_name, name))
    df_result = {}
    for patient_index in patient_index_list:
        feature_vector = []
        lack = False
        for stage_name in stage_list:
            current_patient_index = patient_index.replace(stage_list[0], stage_name)
            feature_file_path = "{}_{}.csv".format(stage_name, label_type)
            feature_file_path = os.path.join(features_root_path, feature_file_path)
            df_feature = pd.read_csv(feature_file_path, index_col=0)
            if current_patient_index not in df_feature.index.values:
                lack = True
                break
            feature_vector.extend(df_feature.loc[current_patient_index].values)
        if not lack:
            df_result[patient_index] = feature_vector
    results = pd.DataFrame(df_result)
    results = results.T
    results.columns = final_columns
    out_file_name = "fusion_features_{}.csv".format(label_type)
    output_file = os.path.join(features_root_path, out_file_name)
    results.to_csv(output_file, encoding='utf_8_sig')



if __name__ == "__main__":
    selena_extract_feature()
    # stage_list = ["t1d", "t1j", "t1p", "t1y", "t2", "echo1", "echo2", "adc", "b=500"]
    # for stage_name in stage_list:
    #     save_path = "/workspace/medical/code/data/results/task2/features"
    #     file_name = "cls_{}.csv".format(stage_name)
    #     file_path = os.path.join(save_path, file_name)
    #     extract_model_feature("/storage/workspace/work/code/data/translate/task2/",
    #                         file_path,
    #                         stage_name)
    # label_count("/storage/workspace/work/code/data/translate/task2/",
    #             "/workspace/medical/code/data/results/task2/features/cls.csv")