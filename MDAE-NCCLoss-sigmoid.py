from sklearn.calibration import partial
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Dense
from keras.layers import Input,Dense
from keras.models import Model
import pandas as pd
import json
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
import math
# path_noise=r"./data/ALFA_ENG_ELE_RU_AIL_imu-data_imu-mag_Pitch-Roll-Yaw_Noise_Abnormal_andNormal.csv" # 5分类，噪声数据
# path_noise=r"D:/test_flower_tf/data/UAVGPSAttack_imu-0_mag_Pitch-Roll_Yaw_Jamming_Spoofing_DownSample_Noise.csv" # 3分类，噪声数据 # 5分类，噪声数据
# path_noise=r"D:/test_flower_tf/data/TLM_imu-0_mag_Pitch-Roll_Yaw_Noise.csv" # 5分类，噪声数据 # 5分类，噪声数据
# path_noise=r"D:/test_flower_tf/data/3class-Dataset_T-Cyber_Encoded_Noise_Time.csv" # 3分类
# path_noise=r"./data/ITS-Physical-3_noise.csv" # 3分类
# path_noise=r"D:/test_flower_tf/data/processed_friday_dataset_CIC_Noise.csv" # 3分类
# path_noise=r"./data/IoT2023_7_noise.csv" # 7分类
# path_noise=r"./data/BAT-Data_Train_noise.csv" # 11分类
# path_noise=r"data/UNSW_NB15_training-set_OK_Balance_noise.csv" # 11分类
# path_noise=r"./BAT_combined_data_noise_X3.csv" # 11分类
# path_noise=r"./BAT_combined_data_noise_X4.csv" # 11分类
path_noise=r"data/awid-dataset_processed_noise.csv" # 11分类

data_noise=pd.read_csv(path_noise)
# data_noise=data_noise.sample(frac=1)
# data_noise=data_noise.drop("timestamp_c",axis=1) # ITS_Cyber正常数据里面有时间戳

#----------------------------------------#
X_data_noise=data_noise.drop(['labels'],axis=1).values
Y_data_noise=data_noise['labels'].values

##############################################Physical扩展##############################################
# print("之前：",pd.Series(Y_data_noise).value_counts())

# from imblearn.over_sampling import SMOTE
# smote=SMOTE(n_jobs=-1,sampling_strategy={1:4290,2:4290})
# X_data_noise, Y_data_noise = smote.fit_resample(X_data_noise, Y_data_noise)

# print("之后：",pd.Series(Y_data_noise).value_counts())
# print("Shape of X_data",X_data_noise.shape)
##############################################Physical扩展##############################################

max_min=StandardScaler()
X_data_noise=max_min.fit_transform(X_data_noise)

#----------------------------------------#
# train_data_noise,test_data_noise,train_labels_noise,test_labels_noise=train_test_split(X_data_noise,Y_data_noise,test_size=0.2)


# path=r"D:./data/ALFA_ENG_ELE_RU_AIL_imu-data_imu-mag_Pitch-Roll-Yaw.csv" # 5分类，正常数据
# path=r"D:/test_flower_tf/data/UAVGPSAttack_imu-0_mag_Pitch-Roll_Yaw_Jamming_Spoofing_DownSample.csv" # 5分类，正常数据
# path=r"D:/test_flower_tf/data/TLM_imu-0_mag_Pitch-Roll_Yaw.csv"# 5分类，正常数据
# path=r"D:/test_flower_tf/data/3class-Dataset_T-Cyber_Encoded_Time.csv" # 3分类
# path=r"./data/ITS-Physical-3.csv" # 3分类
# path=r"./data/IoT2023_7.csv" # 7分类

# path=r"D:/test_flower_tf/data/processed_friday_dataset_CIC.csv" # 3分类
# path=r"./data/BAT-Data_Train.csv" # 11分类
# path=r"./BAT_combined_data.csv" # 11分类
path=r"data/awid-dataset_processed.csv" # 11分类
# path=r"./BAT_combined_data_X4.csv" # 11分类

data=pd.read_csv(path)
# data=data.sample(frac=1)
# data=data.drop("%time",axis=1) # ALFA正常数据里面有时间戳

# data=data.drop("timestamp",axis=1) # UAVGPSAttack正常数据里面有时间戳
# data=data.drop("timestamp",axis=1) # TLM正常数据里面有时间戳
# data=data.drop("timestamp_c",axis=1) # ITS_Cyber正常数据里面有时间戳

#----------------------------------------#
X_data=data.drop(['labels'],axis=1).values
Y_data=data['labels'].values

##############################################Physical扩展##############################################
# print("之前：",pd.Series(Y_data).value_counts())

# from imblearn.over_sampling import SMOTE
# smote=SMOTE(n_jobs=-1,sampling_strategy={1:4290,2:4290})
# X_data, Y_data = smote.fit_resample(X_data, Y_data)

# print("之后：",pd.Series(Y_data).value_counts())
# print("Shape of X_data",X_data.shape)
##############################################Physical扩展##############################################


# print("Shape of X_data",X_data.shape)

max_min=StandardScaler()
X_data=max_min.fit_transform(X_data)

#----------------------------------------#
# train_data,test_data,train_labels,test_labels=train_test_split(X_data,Y_data,test_size=0.2)


encoding_dim=512 #512

input_data_encoder=tf.keras.layers.Input(shape=(X_data.shape[1],),name='input')
#1
model1_encoder=tf.keras.layers.Dense(encoding_dim,activation='relu',name='enD1')(input_data_encoder)
model1_encoder=tf.keras.layers.Dense(encoding_dim+64,activation='relu',name='enD2')(model1_encoder)

#2
model2_encoder=tf.keras.layers.Reshape((X_data.shape[1],1),name='reshape1')(input_data_encoder)
model2_encoder=tf.keras.layers.Conv1D(filters=4,kernel_size=6,activation='relu',name='enCon1')(model2_encoder)
model2_encoder=tf.keras.layers.MaxPool1D(pool_size=2,name='Mp1')(model2_encoder)
model2_encoder=tf.keras.layers.Flatten(name='flatten1')(model2_encoder)
model2_encoder=tf.keras.layers.Dense(encoding_dim+64,activation='relu',name='enD3')(model2_encoder)

merge_encoder=tf.keras.layers.concatenate([model1_encoder,model2_encoder])
out_encoder=Dense(128,activation='relu',name='concatenateD1')(merge_encoder)

# encoder_1=Model(inputs=input_data_encoder,outputs=out_encoder)

#-------------------------------------------------------------------
decoding_dim=512

# input_data_decoder=tf.keras.layers.Input(shape=(128,))
#1
model1_decoder=tf.keras.layers.Dense(decoding_dim+64,activation='relu',name='DeD1')(out_encoder)
model1_decoder=tf.keras.layers.Dense(decoding_dim,activation='relu',name='DeD2')(model1_decoder)
out_decoder=Dense(X_data.shape[1],activation='relu',name='DeD3')(model1_decoder)
#2

model2_decoder=tf.keras.layers.Reshape((128,1),name='reshape2')(out_encoder)

# model2_decoder=tf.keras.layers.Input(shape=(19,))(out_encoder)
model2_decoder=tf.keras.layers.Conv1D(filters=4,kernel_size=6,activation='relu',name='deCon1')(model2_decoder)

model2_decoder=tf.keras.layers.MaxPool1D(pool_size=2,name='Mp2')(model2_decoder)
model2_decoder=tf.keras.layers.Flatten(name='flatten2')(model2_decoder)
model2_decoder=tf.keras.layers.Dense(128,activation='relu',name='DeD3')(model2_decoder)

merge_decoder=tf.keras.layers.concatenate([model1_decoder,model2_decoder])
out_decoder=Dense(X_data.shape[1],activation='relu',name='concatenateD2')(merge_decoder)

# decoder_1=Model(inputs=out_encoder,outputs=out_decoder)

autoencoder=Model(input_data_encoder,out_decoder)
autoencoder.summary()

# def mysigmoid(x):
#     return 1 / (1 + math.exp(-x))
def mysigmoid(x):
    return 1-(1 / (1 + tf.exp(-x)))
def norm_data(data):
    """
    normalize data to have mean=0 and standard_deviation=1
    """
    # mean_data=np.mean(data)
    # std_data=np.std(data, ddof=1)
    #return (data-mean_data)/(std_data*np.sqrt(data.size-1))
    return (data - tf.reduce_mean(data)) / tf.math.reduce_std(data)

def calncc(data0, data1):
    """
    normalized cross-correlation coefficient between two data sets

    Parameters
    ----------
    data0, data1 :  numpy arrays of same size

    """
    ncc = (1.0 / (tf.cast(tf.size(data0), tf.float32) - tf.constant(1.0))) * tf.reduce_sum(norm_data(data0) * norm_data(data1))
    # return 1-((ncc + 1) / 2)  # Normalize to [0, 1]
    return ncc  # Normalize to [0, 1]

# 自定义损失函数
def custom_loss(y_true, y_pred):
    # 均方误差
    mse_loss = tf.keras.losses.mean_squared_error(y_true, y_pred)

    # 计算
    ncc= calncc(y_true, y_pred)
    
    ncc=mysigmoid(ncc)
    # 联合损失函数
    lambda_value =1  # 调节参数
    combined_loss = mse_loss + lambda_value * ncc

    return combined_loss


# tf.keras.utils.plot_model(autoencoder,show_shapes=True,show_layer_names=True)
optimizers = tf.keras.optimizers.Adam(learning_rate=0.1)
autoencoder.compile(optimizer="adam",loss=custom_loss,metrics=["accuracy"])

# 定义 ModelCheckpoint 回调
checkpoint_filepath = './models/best/AWID_Sig_DAE_MultiModel_CheckPoint.h5'
model_checkpoint = ModelCheckpoint(filepath=checkpoint_filepath,
                                   save_best_only=True,
                                   monitor='loss',
                                   mode='min',
                                   verbose=1)
# history=autoencoder.fit(X_data_noise,X_data,epochs=100,validation_split=0.2)
print("GPU is available:",tf.test.is_gpu_available())
history=autoencoder.fit(X_data_noise,X_data,epochs=100,validation_split=0.2, callbacks=[model_checkpoint],batch_size=128)

# autoencoder.save('./models/save/NCCLoss_ITS_Physical_Sig_DAE_MultiModel_CheckPoint.h5')

# with open('./correct/TLM_acc_loss_DAE_MultiModel.json','w') as f:
#     json.dump(history.history,f)