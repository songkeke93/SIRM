import pandas as pd


def data_load(train_cls_file_path, test_cls_file_path, features_file_path, stage, cls_header_name):
    df_train_cls = pd.read_csv(train_cls_file_path, index_col=0)
    df_test_cls = pd.read_csv(test_cls_file_path, index_col=0)
    df_feature = pd.read_csv(features_file_path, index_col=0)
    index_train = []
    cls_train = []
    for item in df_train_cls.index:
        feature_id = item + '-{}'.format(stage)
        if feature_id in df_feature.index:
            index_train.append(feature_id)
            cls_train.append(df_train_cls.loc[item, cls_header_name])
    df_feature_train = df_feature.loc[index_train]
    df_train_cls = pd.DataFrame(cls_train, index=index_train, columns=['group'])

    index_test = []
    cls_test = []
    for item in df_test_cls.index:
        feature_id = item + '-{}'.format(stage)
        if feature_id in df_feature.index:
            index_test.append(feature_id)
            cls_test.append(df_test_cls.loc[item, cls_header_name])
    df_feature_test = df_feature.loc[index_test]
    df_test_cls = pd.DataFrame(cls_test, index=index_test, columns=['group'])

    return df_feature_train, df_train_cls, df_feature_test, df_test_cls


if __name__ == "__main__":
    train_cls_file_path = "/workspace/medical/code/data/cls/train_task2.csv"
    test_cls_file_path = "/workspace/medical/code/data/cls/test_task2.csv"
    features_file_path = "/workspace/medical/code/data/results/task2/features/adc_1.csv"
    data_load(train_cls_file_path, test_cls_file_path, features_file_path, 'adc', 'group')