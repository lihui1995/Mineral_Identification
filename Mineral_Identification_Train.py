# -*- coding: utf-8 -*-
"""
Created on Fri May 11 16:01:05 2018

@author: lihui
All rights reserved by Wuhan University
"""

import glob
import os.path
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import operator
import math
import json
BOTTLENECK_TENSOR_SIZE=2048

BOTTLENECK_TENSOR_NAME='pool_3/_reshape:0'

JPEG_DATA_TENSOR_NAME='DecodeJpeg/contents:0'

MODEL_DIR='...\\inception-2015-12-05'

MODEL_FILE='...\\classify_image_graph_def.pb'

CACHE_DIR='...\\Train_dataset_vector'

INPUT_DATA='...\\Train_dataset'

VALIDATION_PERCENTAGE=20

TEST_PERCENTAGE=20

LEARNING_RATE=0.2

STEPS=1000

BATCH=256

testing_index=[]
label_names=[]

def create_image_lists(testing_percentage,validation_percentage):
#得到的所有图片都存在result这个字典里，字典的key为类别名称，value也是一个字典，字典存储所有图片的名称
    result={}
    label_names=[]
#    获取当前目录下所有的子目录
#    root为根目录，dirs为目录名，files为文件
    sub_dirs=[x[0] for x in os.walk(INPUT_DATA)]
#    得到的第一个目录是当前目录，不要考虑
    is_root_dir=True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir=False
            continue
#        获取当前目录所有的有效文件
        extensions=['jpg','jpeg']
        file_list=[]
#        获取目录最后的文件名
        dir_name=os.path.basename(sub_dir)
        label_name=dir_name.lower()
        label_names.append(label_name)
        for extension in extensions:
            file_glob=os.path.join(INPUT_DATA,dir_name,'*.'+extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list: continue
#    通过目录名获取类别的名称
        training_images=[]
        testing_images=[]
        validation_images=[]
        length=len(file_list)
        number=length*0.2
        random.shuffle(file_list)
        validation=random.sample(file_list,int(number))
        for file_name in validation:
            base_name=os.path.basename(file_name)
            validation_images.append(base_name)
#        testing_images=validation_images
        file_list = [item for item in file_list if item not in set(validation)]
        number=length*0.2
        print(number)
        test=random.sample(file_list,int(number))
        for file_name in test:
            base_name=os.path.basename(file_name)
            testing_images.append(base_name)
        file_list = [item for item in file_list if item not in set(test)]
        number=length*0.6
        train=random.sample(file_list,int(number))
        for file_name in train:
            base_name=os.path.basename(file_name)
            training_images.append(base_name)
        
        result[label_name]={
                'dir':dir_name,
                'training':training_images,
                'testing':testing_images,
                'validation':validation_images
                }
#        dir_txt=dir_name+'.txt'
#        json_write(dir_txt,testing_images)
    json_write('dataset_mineral.txt',result)
    return result

def json_write(filename,list_name):
    with open(filename,'w') as file:
        js = json.dumps(list_name)   
        file.write(js)  

def json_read(list_file):
    label_list={}
    file = open(list_file, 'r') 
    js = file.read()
    label_list = json.loads(js) 
    file.close()
    return label_list


#这个函数通过类别名称、所属数据集和图片编号获取一张图片的地址
#image_lists参数给出所有图片信息
#        image_dir参数给出根目录。存放图片数据的根目录与存放图片特征向量的根目录地址不同
#        label_name参数给定类别名称
#        index参数指定需要获取的图片编号
#        category参数指定获取的图片是在训练数据集、测试数据集还是验证数据集
def get_image_path(image_lists,image_dir,label_name,index,category):
#    获取给定类别中的所有信息
    label_lists=image_lists[label_name]
    
#    获取所属数据集名称获取集合中所有图片信息
    category_list=label_lists[category]
    mod_index=index%len(category_list)
    #获取文件名
    base_name=category_list[mod_index]
    sub_dir=label_lists['dir']
#    最终地址为数据根目录地址加上类别的文件夹加上图片的名称
    full_path=os.path.join(image_dir,sub_dir,base_name)
    if category=="testing":
        testing_index.append(sub_dir)
    return full_path

#这个函数通过类别名称、所属数据集和图片编号获取经过模型处理后的特征向量
def get_bottleneck_path(image_lists,label_name,index,category):
    return get_image_path(image_lists,CACHE_DIR,label_name,index,category)+'.txt'


#这个函数加载训练好的模型处理一张图片得到这个图片的特征向量
def run_bottleneck_on_image(sess,image_data,image_data_tensor,bottleneck_tensor):
#    这个过程就是将当前图片作为输入计算瓶颈张量的值，这值就是这图片的新的特征向量
    bottleneck_values=sess.run(bottleneck_tensor,{image_data_tensor:image_data})
#    卷积神经网络处理最后得到一个四维数组，需要将这个结果压缩成一个特征向量
    bottleneck_values=np.squeeze(bottleneck_values)
    return bottleneck_values

#这个函数获取一张图片经过模型处理后的特征向量。这个函数会先寻找已经计算且保存下来的特征向量
#如果找不到则计算这个向量，然后保存
def get_or_create_bottleneck(sess,image_lists,label_name,index,category,jpeg_data_tensor,bottleneck_tensor):
#    获取一张图片对应的特征向量文件的路径
    label_lists=image_lists[label_name]
    sub_dir=label_lists['dir']
    
    sub_dir_path=os.path.join(CACHE_DIR,sub_dir)
  
    if not os.path.exists(sub_dir_path):os.makedirs(sub_dir_path)
    bottleneck_path=get_bottleneck_path(image_lists,label_name,index,category)
#    如果这个特征向量文件不存在，则通过模型计算特征向量，并将结果保存
    if not os.path.exists(bottleneck_path):
#        获取图片原始路径
        image_path=get_image_path(
                image_lists,INPUT_DATA,label_name,index,category)
#        获取图片内容
        image_data=gfile.FastGFile(image_path,'rb').read()
#        通过模型计算特征向量
        bottleneck_values=run_bottleneck_on_image(sess,image_data,jpeg_data_tensor,bottleneck_tensor)
#        保存计算结果
        bottleneck_string=','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_path,'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)
    else:
        with open(bottleneck_path,'r') as bottleneck_file:
            bottleneck_string =bottleneck_file.read()
        bottleneck_values=[float(x) for x in bottleneck_string.split(',')]
    return bottleneck_values
    
#这个函数随机获取一个batch的图片作为训练数据
def get_random_cached_bottlenecks(sess,n_classes,image_lists,how_many,category,jpeg_data_tensor,bottleneck_tensor):
    bottlenecks=[]
    ground_truths=[]
    for _ in range (how_many):
#        随机一个类别和图片的编号加入当前训练数据
        label_index=random.randrange(n_classes)
        label_name=list(image_lists.keys())[label_index]
        image_index=random.randrange(3537)
        bottleneck=get_or_create_bottleneck(sess,image_lists,label_name,image_index,category,jpeg_data_tensor,bottleneck_tensor)
        ground_truth=np.zeros(n_classes,dtype=np.float32)
        ground_truth[label_index]=1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)
    return bottlenecks,ground_truths
        

#这个函数获取全部的测试数据。在最终测试的需要在所有的测试数据上计算正确率
def get_test_bottlenecks(sess,image_lists,n_classes,
                         jpeg_data_tensor,bottleneck_tensor):
    bottlenecks=[]
    ground_truths=[]
    label_name_list=list((image_lists.keys()))
    #枚举所有的类别和每个类别中的测试图片
    #enumerate为枚举函数，简化遍历字典代码
    for label_index,label_name in enumerate(label_name_list):
        category='testing'
        for index,unused_base_name in enumerate(
                image_lists[label_name][category]):
            bottleneck=get_or_create_bottleneck(sess,
                                                image_lists,label_name,index,category,
                                                jpeg_data_tensor,bottleneck_tensor)
            ground_truth=np.zeros(n_classes,dtype=np.float32)
            ground_truth[label_index]=1.0
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
    return bottlenecks,ground_truths

def bubble_sort(lists):
    # 冒泡排序
    count = len(lists)
    for i in range(0, count):
        for j in range(i + 1, count):
            if lists[i] < lists[j]:
                lists[i], lists[j] = lists[j], lists[i]
    return lists

def saveVariables(filename,variable):
     with open(filename,'w') as file:
          for x in variable:
              file.write(str(x)+'\n')
          file.close()
     return

def main(_):
    #绘制曲线
    x_range=[]
    x2_range=[]
    y_accuracy=[]
    y2_accuracy=[]
    x_train=[]
    y_train=[]
    #读取所有图片
    image_lists=create_image_lists(TEST_PERCENTAGE,VALIDATION_PERCENTAGE)
#    image_lists=json_read('dataset_mineral.txt')
    
    #类别种类
    n_classes=len(image_lists.keys())
    
    label_name_list=list((image_lists.keys()))
    print(label_name_list)
#    生成标签文件
    filename="label_name_list_mineral.txt"
    with open(filename,'w') as file:
        for label in label_name_list:
            file.write(label+"\n")
        
         #读取已经训练好的模型。谷歌训练好的模型保存在GraphDefProtoolBuffer,
#    里面保存了每个节点的取值的计算方法以及变量的取值。
    with gfile.FastGFile(os.path.join(MODEL_DIR,MODEL_FILE),'rb') as f:
             graph_def1=tf.GraphDef()
             graph_def1.ParseFromString(f.read())
#        读取模型，返回数据输入对应的张量以及计算瓶颈层结构对应的张量
    bottleneck_tensor,jpeg_data_tensor=tf.import_graph_def(graph_def1,return_elements=[BOTTLENECK_TENSOR_NAME,JPEG_DATA_TENSOR_NAME])     
  #定义输入
    bottleneck_input=tf.placeholder(tf.float32,[None,BOTTLENECK_TENSOR_SIZE],name='BottleneckPlaceholder')
        #定义输出
    ground_truth_input=tf.placeholder(tf.float32,[None,n_classes],name='GroundTruthnput')
         
    #定义第一层全连接
    weights_1=tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE,2048],stddev=0.01),name='Weights_1')
    biases_1=tf.Variable(tf.zeros([2048],name='Biases_1'))
    logits_1=tf.add(tf.matmul(bottleneck_input,weights_1),biases_1,name='Logits_1')
    keep_prob = tf.placeholder(tf.float32,name='keep_prob')
    output_1=tf.nn.relu(logits_1,name='Output_1')
    output_1 = tf.nn.dropout(output_1, keep_prob,name='Dropout_1')
    weights_2=tf.Variable(tf.truncated_normal([2048,2048],stddev=0.01),name='Weights_2')
    biases_2=tf.Variable(tf.zeros([2048],name='Biases_2'))
    logits_2=tf.add(tf.matmul(output_1,weights_2),biases_2,name='Logits_2')
    output_2=tf.nn.relu(logits_2,name='Output_2')
    output_2 = tf.nn.dropout(output_2, keep_prob,name='Dropout_2')
    #定义最后一层全连接及softmax层操作
    weights_3=tf.Variable(tf.truncated_normal([2048,n_classes],stddev=0.01),name='Weights_3')
    biases_3=tf.Variable(tf.zeros([n_classes]),name='Biases_3')
    logits_3=tf.add(tf.matmul(output_2,weights_3),biases_3,name='Logits_3')
    final_tensor=tf.nn.softmax(logits_3,name='Final_tensor')
    cross_entropy=tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_3,labels=ground_truth_input)
    cross_entropy_mean=tf.reduce_mean(cross_entropy)
    global_step=tf.Variable(0)
    learning_rate=tf.train.exponential_decay(LEARNING_RATE,global_step,100,0.9995,staircase=True)
    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy_mean,global_step=global_step)
    correct_prediction=tf.equal(tf.argmax(final_tensor,1),
                                    tf.argmax(ground_truth_input,1),name='Correct_prediction')
    evaluation_step=tf.reduce_mean(
                tf.cast(correct_prediction,tf.float32),name='Evaluation_step')
    with tf.Session() as sess:
                 init=tf.global_variables_initializer()
                 sess.run(init)
        #训练过程
                 for i in range (STEPS):
            #每次获取一个batch数据
                     train_bottlenecks,train_ground_truth=get_random_cached_bottlenecks(sess,n_classes,image_lists,BATCH,'training',jpeg_data_tensor,bottleneck_tensor)
                     _,cross,training_accuracy=sess.run([train_step,cross_entropy_mean,evaluation_step],                              feed_dict={bottleneck_input: train_bottlenecks,
                                ground_truth_input: train_ground_truth,keep_prob: 1.0})
    #验证数据测试正确率
                     if i%50==0:
                        validation_bottlenecks,validation_ground_truth=\
                        get_random_cached_bottlenecks(
                         sess,n_classes,image_lists,BATCH,'validation',
                         jpeg_data_tensor,bottleneck_tensor)
                        validation_accuracy,validation_loss=sess.run([evaluation_step,cross_entropy_mean],feed_dict={
                                bottleneck_input:validation_bottlenecks,ground_truth_input: validation_ground_truth,keep_prob: 1.0})
                        x2_range.append(i)
                        y2_accuracy.append(cross)
                        print('Step %d:Validation accuracy on random sampled''%d examples=%.1f%%'%(i,BATCH,validation_accuracy*100))
                        x_range.append(i)
                        y_accuracy.append(validation_accuracy*100)
                        x_train.append(i)
                        y_train.append(training_accuracy*100)

                 test_bottlenecks,test_ground_truth=get_test_bottlenecks(
                        sess,image_lists,n_classes,jpeg_data_tensor,bottleneck_tensor)
                 test_accuracy,final_tensor=sess.run([evaluation_step,final_tensor],feed_dict={
                 bottleneck_input:test_bottlenecks,ground_truth_input:test_ground_truth,keep_prob: 1.0})
                 print('Step %d:Test accuracy on random sampled''%d examples=%.1f%%'%(i,BATCH,test_accuracy*100))
                 plt.plot(x_range,y_accuracy,x_train,y_train)
                 plt.show()
                 plt.plot(x2_range,y2_accuracy)
                 plt.show()
                 constant_graph = graph_util.convert_variables_to_constants(sess,sess.graph_def,['Weights_1','Weights_2','Biases_1','Biases_2','Weights_3','Biases_3','Final_tensor'])
                 with tf.gfile.FastGFile('Model_mineral.pb', mode='wb') as f:
                      f.write(constant_graph.SerializeToString())
if __name__=='__main__':
    tf.app.run()
    
    
    

    

            
        
        
        
        
        
        
        
        
        
        
        
        

        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    




















