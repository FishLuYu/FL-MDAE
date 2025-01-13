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
from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score,confusion_matrix,classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint


# path=r"D:/test_flower_tf/data/ALFA_ENG_ELE_RU_AIL_imu-data_imu-mag_Pitch-Roll-Yaw_Noise_Abnormal_andNormal.csv" # 5分类
# path=r"D:/test_flower_tf/data/UAVGPSAttack_imu-0_mag_Pitch-Roll_Yaw_Jamming_Spoofing_DownSample_Noise.csv"# 3分类
# path=r"D:/test_flower_tf/data/TLM_imu-0_mag_Pitch-Roll_Yaw_Noise.csv"# 5分类
# path=r"D:/test_flower_tf/data/3class-Dataset_T-Cyber_Encoded_Noise_Time.csv" # 3分类
# path=r"./data/ITS-Physical-3_noise.csv" # 3分类
# path=r"./data/BAT-Data_Train_noise.csv" # 11分类
path=r"./BAT_combined_data_noise_X4.csv" # 11分类


data=pd.read_csv(path,header=0)
# data=data.drop("timestamp_c",axis=1) # ITS_Cyber正常数据里面有时间戳

data=data.sample(frac=1)

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
train_data,test_data,train_labels,test_labels=train_test_split(X_data,Y_data,test_size=0.2)

orgAEModel=tf.keras.models.load_model('./models/baseModel/best/BAT_balance_X4_DAE_CheckPoint.h5')

# orgAEModel.summary()

# 冻结所有参数(0-13)/全部
for layer in orgAEModel.layers[0:12]:
    layer._name = layer._name + str("2")
    layer.trainable=False
    # tf.keras.backend.clear_session()

# 添加全连接层分类器
x=orgAEModel.output
x=tf.keras.layers.Dense(128,activation='relu',name='fineDense')(x)

x=tf.keras.layers.Dense(256,activation='relu',name='fineDense2')(x)

out=tf.keras.layers.Dense(11,activation='softmax')(x)

fineTuningModel=Model(inputs=orgAEModel.input,outputs=out)

fineTuningModel.summary()

fineTuningModel.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])

checkpoint_filepath = './models/baseModel/testmodel/fineTuning_BAT_balance_X4_DAE_CheckPoint.h5'
model_checkpoint = ModelCheckpoint(filepath=checkpoint_filepath,
                                   save_best_only=True,
                                   monitor='accuracy',
                                   mode='max',
                                   verbose=1)

history=fineTuningModel.fit(train_data,train_labels,epochs=100,validation_split=0.1,callbacks=[model_checkpoint],batch_size=128)
# fineTuningModel.save('./Comparision_Model/baseModel/ALFA_fineTuning_DAE(MLP)_13.h5')


# 测试
testmodel=tf.keras.models.load_model('./models/baseModel/testmodel/fineTuning_BAT_balance_X4_DAE_CheckPoint.h5')
test_predictions = testmodel.predict(test_data)
test_prediction=np.argmax(test_predictions,axis=1)
accuracy = accuracy_score(test_labels,test_prediction)
print('Test Accuracy:', accuracy)

classificationReport=classification_report(test_labels,test_prediction)
print('Classification Report:\n', classificationReport)


cm=confusion_matrix(test_labels,test_prediction)

precision = precision_score(test_labels, test_prediction, average='macro')
print('precision:', precision)

f1=f1_score(test_labels,test_prediction,average='micro')
print("F1:",f1)

recall = recall_score(test_labels, test_prediction, average='macro')
print('recall:', recall)

# # 混淆矩阵
# cm=confusion_matrix(test_labels,test_prediction)

# plt.rcParams['font.family'] = 'Times New Roman'
# plt.title('Confusion Matrix',fontsize=12)
# # cm_1 = cm/cm.sum(axis=0) #转换成小数
# cm_1 = cm #转换成小数
# # cm_2 = pd.DataFrame(cm_1, index=['Normal','Engine','Elevator','Rudder','Aileron'], columns=['Normal','Engine','Elevator','Rudder','Aileron'])
# # cm_2 = pd.DataFrame(cm_1, index=['1','2','3','4','5'], 
# #                     columns=['1','2','3','4','5'])
# cm_2 = pd.DataFrame(cm_1, index=['Normal','GPS Jamming','GPS Spoofing'], 
#                     columns=['Normal','GPS Jamming','GPS Spoofing'])
# sns.heatmap(cm_2, annot=True, annot_kws={"size": 10}, cmap="Blues",fmt='.20g')
# plt.ylabel('True label', fontsize=12)
# plt.xlabel('Predicted label', fontsize=12)
# plt.xticks(fontsize=8)
# plt.yticks(fontsize=8)

# plt.show()
# with open('./correct/ITS_acc_loss_fineTuningDAE_13.json','w') as f:
#     json.dump(history.history,f)

