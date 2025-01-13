from typing import Dict, Tuple
from flwr.common import NDArrays, Scalar
import tensorflow as tf
# print(tf.config.list_physical_devices('GPU'))
# print('Num GPUs Available:',len(tf.config.list_physical_devices('GPU')))
import flwr as fl

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from keras.layers import Input,Dense
from keras.models import Model

path=r"D:/test_flower_tf/data/ALFA_ENG_ELE_RU_AIL_imu-data_imu-mag_Pitch-Roll-Yaw.csv" # 5分类
data=pd.read_csv(path,header=0)
data=data.drop("%time",axis=1)
data=data.sample(frac=1)

#----------------------------------------#
X_data=data.drop(['labels'],axis=1).values
Y_data=data['labels'].values

max_min=StandardScaler()
X_data=max_min.fit_transform(X_data)

#----------------------------------------#
train_data,test_data,train_labels,test_labels=train_test_split(X_data,Y_data,test_size=0.2)

#-------------------模型构建---------------------#
input_data=tf.keras.layers.Input(shape=(19,))
model1=tf.keras.layers.Dense(64,activation='relu')(input_data)
model1=tf.keras.layers.Dense(128,activation='relu')(model1)

model2=tf.keras.layers.Reshape((19,1))(input_data)
model2=tf.keras.layers.Conv1D(filters=3,kernel_size=3,activation='relu')(model2)
model2=tf.keras.layers.MaxPool1D(pool_size=2)(model2)
model2=tf.keras.layers.Flatten()(model2)
model2=tf.keras.layers.Dense(128,activation='relu')(model2)

merge=tf.keras.layers.concatenate([model1,model2])
out=Dense(128,activation='relu')(merge)
des=Dense(5,activation='softmax')(out)
model=Model(inputs=input_data,outputs=des)

# (X_train,y_train),(X_test,y_test)=tf.keras.datasets.cifar10.load_data()
# model=tf.keras.applications.MobileNetV2((32,32,3),classes=10,weights=None)
model.compile("adam","sparse_categorical_crossentropy",metrics=["accuracy"])

class MyNet(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()
    
    def fit(self,parameters,config):
        model.set_weights(parameters)
        # model.fit(X_train,y_train,epochs=200,batch_size=32)
        model.fit(train_data,train_labels,epochs=200)

        return model.get_weights(),len(train_data),{}
    
    def evaluate(self, parameters,config):
        model.set_weights(parameters)
        loss,accuracy=model.evaluate(test_data,test_labels)
        return loss,len(test_data),{"accuracy":float(accuracy)}
    

fl.client.start_numpy_client(server_address="127.0.0.1:8080",client=MyNet())