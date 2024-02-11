import os
import pandas as pd
import numpy as np


def dataset_split(root_path, train_radio, cls_path):
    path_quene = [root_path]
    name_list = []
    while len(path_quene) > 0:
        current_path = path_quene.pop(0)
        if 'img' in os.listdir(current_path):

            name = current_path.replace(root_path, '').replace('/', '-')
            name = name[:name.rfind('-')]
            name_list.append(name)
        else:
            for name in os.listdir(current_path):
                path_quene.append(os.path.join(current_path, name))
    name_list = list(set(name_list))
    name_array = np.asarray(name_list)
    # name_array.sort()
    patient_num = len(name_list)
    print("共有{}个病人".format(patient_num))
    np.random.seed(20210531)
    patient_index = np.random.permutation(patient_num)
    train_node = int(patient_num * train_radio)
    train_index = name_array[patient_index[:train_node]]
    test_index = name_array[patient_index[train_node:]]
    # read cls
    cls = read_cls_2(cls_path)
    train_cls = []
    for pos, inx in enumerate(train_index):
        i = inx.split('-')[0]
        train_cls.append(cls[i])
    test_cls = []
    for pos, inx in enumerate(test_index):
        i = inx.split('-')[0]
        test_cls.append(cls[i])
    train_cls = np.asarray(train_cls)
    test_cls = np.asarray(test_cls)
    assert train_index.shape[0] == train_cls.shape[0], "wrong"
    assert test_index.shape[0] == test_cls.shape[0], "wrong"

    df_train = pd.DataFrame(train_cls, index=train_index, columns=['group'])
    df_test = pd.DataFrame(test_cls, index=test_index, columns=['group'])
    df_train.to_csv("/workspace/medical/code/data/cls/train_task2.csv")
    df_test.to_csv("/workspace/medical/code/data/cls/test_task2.csv")

def read_cls_2(file_path):
    cls_dict = {}
    with open(file_path, 'r') as f:
        for line in f.readlines():
            cls_dict[line.strip().split('\t')[0]] = int(line.strip().split('\t')[1]) - 1
    return cls_dict


if __name__ == "__main__":
    root_path = "/storage/workspace/work/code/data/translate/task2/"
    cls_task2 = "/workspace/medical/code/data/cls/task2.txt"
    dataset_split(root_path, 0.8, cls_task2)