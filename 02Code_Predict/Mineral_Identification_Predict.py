# -*- coding: utf-8 -*-
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


model_dir = ""  # the path of inceptionv3 model
image_path = ""
label_file = ""
model_pb_dir = ""
TransferModel_dir = ""

# 读取训练好的Inception-v3模型来创建graph
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
    with open(label_file) as labels:
        for label in labels:
            label_list.append(label)
    return label_list


# 读取图片
image_data = tf.gfile.FastGFile(image_path, "rb").read()
# 创建graph
g1 = tf.Graph()
g2 = tf.Graph()
with g1.as_default():
    create_inception_graph()
    sess = tf.Session(graph=g1)
    bottleneck_tensor = sess.graph.get_tensor_by_name("pool_3/_reshape:0")
    bottleneck_tensor_value = sess.run(
        bottleneck_tensor, {"DecodeJpeg/contents:0": image_data}
    )
    sess.close()

with tf.Session(graph=g2) as sess1:
    create_layer_graph()
    label_list = read_label_text()
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
    predict = tf.argmax(final)
    lena = mpimg.imread(image_path)
    plt.imshow(lena)
    plt.show()
    print(predict.eval())
    print(label_list[predict.eval()])
    confidence = np.max(final.eval())
    print(confidence)
    label_name_list = read_label_text()
    final_1 = bubble_sort(final.eval())
    for index, value in enumerate(final_1):
        if index < 3:
            for index1, value1 in enumerate(final.eval()):
                if operator.eq(value, value1):
                    number = index1
                    print("图片识别结果为 " "%s" "%.2f" % (label_name_list[number], value))
