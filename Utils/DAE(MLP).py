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
from tensorflow.keras.callbacks import ModelCheckpoint



# path_noise=r"D:/test_flower_tf/data/ALFA_ENG_ELE_RU_AIL_imu-data_imu-mag_Pitch-Roll-Yaw_Noise_Abnormal_andNormal.csv" # 5分类，噪声数据
# path_noise=r"D:/test_flower_tf/data/UAVGPSAttack_imu-0_mag_Pitch-Roll_Yaw_Jamming_Spoofing_DownSample_Noise.csv" # 3分类，噪声数据 # 5分类，噪声数据
# path_noise=r"D:/test_flower_tf/data/TLM_imu-0_mag_Pitch-Roll_Yaw_Noise.csv" # 5分类，噪声数据 # 5分类，噪声数据
# path_noise=r"D:/test_flower_tf/data/3class-Dataset_T-Cyber_Encoded_Noise_Time.csv" # 3分类
# path_noise=r"./data/ITS-Physical-3_noise.csv" # 3分类
# path_noise=r"./data/BAT-Data_Train_noise.csv" # 11分类
path_noise=r"./BAT_combined_data_noise_X4.csv" # 11分类

data_noise=pd.read_csv(path_noise,header=0)
# data_noise=data_noise.sample(frac=1)
# data_noise=data_noise.drop("timestamp_c",axis=1) # ITS_Cyber正常数据里面有时间戳

#----------------------------------------#
X_data_noise=data_noise.drop(['labels'],axis=1).values
Y_data_noise=data_noise['labels'].values

# 扩展
# from imblearn.over_sampling import SMOTE
# smote=SMOTE(n_jobs=-1,sampling_strategy={4:3500,5:3500,6:3500,7:3500,8:3500,9:3500,10:3500})
# X_data_noise, Y_data_noise = smote.fit_resample(X_data_noise, Y_data_noise)

# print("Shape of X_data",X_data.shape)

max_min=StandardScaler()
X_data_noise=max_min.fit_transform(X_data_noise)

#----------------------------------------#
# train_data_noise,test_data_noise,train_labels_noise,test_labels_noise=train_test_split(X_data_noise,Y_data_noise,test_size=0.2)


# path=r"D:/test_flower_tf/data/ALFA_ENG_ELE_RU_AIL_imu-data_imu-mag_Pitch-Roll-Yaw.csv" # 5分类，正常数据
# path=r"D:/test_flower_tf/data/UAVGPSAttack_imu-0_mag_Pitch-Roll_Yaw_Jamming_Spoofing_DownSample.csv" # 5分类，正常数据
# path=r"D:/test_flower_tf/data/TLM_imu-0_mag_Pitch-Roll_Yaw.csv"# 5分类，正常数据
# path=r"D:/test_flower_tf/data/3class-Dataset_T-Cyber_Encoded_Time.csv" # 3分类
# path=r"./data/ITS-Physical-3.csv" # 7分类
# path=r"./data/BAT-Data_Train.csv" # 11分类
path=r"./BAT_combined_data_X4.csv" # 11分类


data=pd.read_csv(path,header=0)
# data=data.sample(frac=1)
# data=data.drop("timestamp",axis=1) # 正常数据里面有时间戳
# data=data.drop("%time",axis=1) # 正常数据里面有时间戳
# data=data.drop("timestamp_c",axis=1) # ITS_Cyber正常数据里面有时间戳

#----------------------------------------#
X_data=data.drop(['labels'],axis=1).values
Y_data=data['labels'].values

#扩展
# from imblearn.over_sampling import SMOTE
# smote=SMOTE(n_jobs=-1,sampling_strategy={4:3500,5:3500,6:3500,7:3500,8:3500,9:3500,10:3500,})
# X_data, Y_data = smote.fit_resample(X_data, Y_data)
# print("Shape of X_data",X_data.shape)

max_min=StandardScaler()
X_data=max_min.fit_transform(X_data)

#----------------------------------------#
# train_data,test_data,train_labels,test_labels=train_test_split(X_data,Y_data,test_size=0.2)


encoding_dim=512

input_data_encoder=tf.keras.layers.Input(shape=(X_data.shape[1],))
#1 encoder
encoder=tf.keras.layers.Dense(encoding_dim,activation='relu')(input_data_encoder)

en_1=tf.keras.layers.Dense(encoding_dim+16,activation='relu')(encoder)
#2 decoder
decoder=tf.keras.layers.Dense(encoding_dim,activation='relu')(en_1)
decoderout=tf.keras.layers.Dense(X_data.shape[1],activation='relu')(decoder)


autoencoder=tf.keras.Model(input_data_encoder,decoderout)

autoencoder.summary()
# tf.keras.utils.plot_model(autoencoder,show_shapes=True,show_layer_names=True)
autoencoder.compile(optimizer="adam",loss="mean_squared_error",metrics=["accuracy"])

checkpoint_filepath = './models/baseModel/best/BAT_balance_X4_DAE_CheckPoint.h5'
model_checkpoint = ModelCheckpoint(filepath=checkpoint_filepath,
                                   save_best_only=True,
                                   monitor='loss',
                                   mode='min',
                                   verbose=1)

history=autoencoder.fit(X_data_noise,X_data,epochs=200,validation_split=0.2, callbacks=[model_checkpoint],batch_size=128)

# autoencoder.save('./models/baseModel/save/BAT_DAE(MLP).h5')

# with open('./correct/ITS_acc_loss_DAE(MLP)-13.json','w') as f:
#     json.dump(history.history,f)