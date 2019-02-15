# -*- coding: utf-7 -*-
"""
Created on Sun May 27 13:22:30 2018

@author: lihui
"""

import tensorflow as tf
import numpy as np
from tensorflow.python.framework import graph_util
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import operator
import json
import glob

model_dir = "...//inception-2015-12-05"
label_file = "...//"
INPUT_DATA = "...//test_dataset"  # the path of an independent mineral dataset
model_pb_dir = "...//classify_image_graph_def"
TransferModel_dir = ""


def create_inception_graph():
    with tf.gfile.FastGFile(os.path.join(model_dir, model_pb_dir), "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name="")


def bubble_sort(lists):
    # 冒泡排序
    count = len(lists)
    for i in range(0, count):
        for j in range(i + 1, count):
            if lists[i] < lists[j]:
                lists[i], lists[j] = lists[j], lists[i]
    return lists


def create_layer_graph():
    with tf.gfile.FastGFile(TransferModel_dir, "rb") as f:
        graph_def_layer = tf.GraphDef()
        graph_def_layer.ParseFromString(f.read())
        tf.import_graph_def(graph_def_layer, name="")


def read_label_text():
    label_list = []
    with open(label_file, encoding="gb18030", errors="ignore") as labels:
        for label in labels:
            label_list.append(label)
    return label_list


def predict(image):
    image_data = tf.gfile.FastGFile(image, "rb").read()
    g1 = tf.Graph()
    g2 = tf.Graph()
    with g1.as_default():
        create_inception_graph()
        sess = tf.Session(graph=g1)
        # Inception-v3模型的最后池化层的输出
        bottleneck_tensor = sess.graph.get_tensor_by_name("pool_3/_reshape:0")
        # 输入图像数据，得到最后一个池化层输出张量
        bottleneck_tensor_value = sess.run(
            bottleneck_tensor, {"DecodeJpeg/contents:0": image_data}
        )
        sess.close()
    with tf.Session(graph=g2) as sess1:
        create_layer_graph()
        c = tf.placeholder(tf.float32, [1, 2048], name="bottleneck_tensor")
        c = bottleneck_tensor_value
        a = tf.placeholder(tf.float32, [None, None], name="weights_1")
        b = tf.placeholder(tf.float32, [None, None], name="biases_1")
        d = tf.placeholder(tf.float32, [None, None], name="weights_2")
        e = tf.placeholder(tf.float32, [None, None], name="biases_2")
        a = sess1.graph.get_tensor_by_name("Weights_1:0")
        b = sess1.graph.get_tensor_by_name("Biases_1:0")
        d = sess1.graph.get_tensor_by_name("Weights_2:0")
        e = sess1.graph.get_tensor_by_name("Biases_2:0")
        logits_1 = tf.add(tf.matmul(c, a), b)
        output_1 = tf.nn.relu(logits_1)
        logits_2 = tf.add(tf.matmul(output_1, d), e)
        output_2 = tf.nn.relu(logits_2)
        final_tensor = tf.nn.softmax(output_2)
        final = tf.squeeze(final_tensor)
        final_tensor = final.eval()
    return final_tensor


def json_write(filename, list_name):
    with open(filename, "w") as file:
        js = json.dumps(list_name)
        file.write(js)


def create_image_lists(INPUT_DATA):
    # 得到的所有图片都存在result这个字典里，字典的key为类别名称，value也是一个字典，字典存储所有图片的名称
    result = {}
    label_names = []
    #    获取当前目录下所有的子目录
    #    root为根目录，dirs为目录名，files为文件
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    #    得到的第一个目录是当前目录，不要考虑
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        #        获取当前目录所有的有效文件
        extensions = ["jpg", "jpeg"]
        file_list = []
        #        获取目录最后的文件名
        dir_name = os.path.basename(sub_dir)
        label_name = dir_name.lower()
        print(dir_name)
        label_names.append(label_name)
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, "*." + extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list:
            continue
        #    通过目录名获取类别的名称
        test = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            test.append(base_name)
        result[label_name] = {"test": test}
    json_write("test.txt", result)
    return result


def main():
    i = 0
    test_index = []
    result = create_image_lists(INPUT_DATA)
    file = open("final_tensor.txt", "w")
    for key, value in result.items():
        label_name = key
        test_list = value["test"]
        print(label_name)
        for label in test_list:
            test_index.append(i)
            print(label)
            image = os.path.join(INPUT_DATA, label_name, label)
            final_tensor = predict(image)
            for f in final_tensor:
                file.write(str(f) + " ")
            file.write("\n")
        i = i + 1
    np.savetxt("test_index.txt", test_index)
    file.close()


if __name__ == "__main__":
    main()
