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
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp
from itertools import cycle
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'Times New Roman'

# path=r"D:/test_flower_tf/data/ALFA_ENG_ELE_RU_AIL_imu-data_imu-mag_Pitch-Roll-Yaw_Noise_Abnormal_andNormal.csv" # 5分类
# path=r"D:/test_flower_tf/data/UAVGPSAttack_imu-0_mag_Pitch-Roll_Yaw_Jamming_Spoofing_DownSample_Noise.csv"# 3分类
# path=r"D:/test_flower_tf/data/TLM_imu-0_mag_Pitch-Roll_Yaw_Noise.csv"# 5分类
# path=r"D:/test_flower_tf/data/3class-Dataset_T-Cyber_Encoded_Noise_Time.csv" # 3分类
# path=r"./data/processed_friday_dataset_CIC_Noise.csv" # 3分类
# path=r"data/IoT2023_7_noise.csv" # 7分类
# path=r"data/BAT-Data_Train_noise.csv" # 11分类
# path=r"./data/ITS-Physical-3_noise.csv" # 3分类
# path=r"data/UNSW_NB15_training-set_OK_Balance_noise.csv" # 11分类
# path=r"./BAT_combined_data_noise_X3.csv" # 11分类
# path=r"./BAT_combined_data_noise_X4.csv" # 11分类
path=r"data/awid-dataset_processed_noise.csv" # 11分类

data=pd.read_csv(path)
print(data['labels'].value_counts())
# data=data.sample(frac=1)
# data=data.drop("timestamp_c",axis=1) # ITS_Cyber正常数据里面有时间戳

#----------------------------------------#
X_data=data.drop(['labels'],axis=1).values
Y_data=data['labels'].values


max_min=StandardScaler()
X_data=max_min.fit_transform(X_data)

#----------------------------------------#
# NB数据集不划分训练集和测试集
train_data,test_data,train_labels,test_labels=train_test_split(X_data,Y_data,test_size=0.2)

# train_data=X_data
# train_labels=Y_data

# testDD=pd.read_csv("data/UNSW_NB15_testing-set_OK_Balance_noise.csv")
# test_data=testDD.drop(['labels'],axis=1).values
# test_data=max_min.transform(test_data)
# test_labels=testDD['labels'].values

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
    return ncc   # Normalize to [0, 1]

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

# 加载模型
# orgAEModel=tf.keras.models.load_model('./models/correct/ITS_DAE_MultiModel.h5')
orgAEModel = tf.keras.models.load_model('./models/best/AWID_Sig_DAE_MultiModel_CheckPoint.h5', custom_objects={'custom_loss': custom_loss})


# orgAEModel.summary()

# 冻结所有参数(0-13)/全部
i=0
for layer in orgAEModel.layers[0:12]:
    # layer._name = layer.name + str("Hellotest2")
    # print("第%s层的名字是：%s"%(i,layer.name))
    # i=i+1
    layer.trainable=False
    # tf.keras.backend.clear_session()

# 添加全连接层分类器
x=orgAEModel.output
x=tf.keras.layers.Dense(256,activation='relu',name='fineDense')(x) #128

x=tf.keras.layers.Dense(128,activation='relu',name='fineDense2')(x)#265

# x=tf.keras.layers.Dense(64,activation='relu',name='fineDense3')(x)
# x=tf.keras.layers.Dense(32,activation='relu',name='fineDense4')(x)


out=tf.keras.layers.Dense(10,activation='softmax')(x)

fineTuningModel=Model(inputs=orgAEModel.input,outputs=out)

fineTuningModel.summary()

optimizers = tf.keras.optimizers.Adam(learning_rate=0.001)

fineTuningModel.compile(optimizer=optimizers,loss="sparse_categorical_crossentropy",metrics=["accuracy"])

checkpoint_filepath = './models/testmodel/testFineTuning_AWID_Sig_DAE_MultiModel_CheckPoint.h5'
model_checkpoint = ModelCheckpoint(filepath=checkpoint_filepath,
                                   save_best_only=True,
                                   monitor='accuracy',
                                   mode='max',
                                   verbose=1)
print("GPU is available:",tf.test.is_gpu_available())
history=fineTuningModel.fit(train_data,train_labels,epochs=100,validation_split=0.1,callbacks=[model_checkpoint],batch_size=128)
# fineTuningModel.save('./models/correct/TLM.h5')

testmodel=tf.keras.models.load_model('./models/testmodel/testFineTuning_AWID_Sig_DAE_MultiModel_CheckPoint.h5')

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




'''# 将标签二值化
y_test = label_binarize(test_labels, classes=[0, 1, 2,3,4,5,6,7,8,9,10])
n_classes = y_test.shape[1]

# 计算每个类别的ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], test_predictions[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# 计算微平均ROC曲线和ROC面积
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), test_predictions.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure()
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve of Multi-class')
plt.legend(loc="lower right")
plt.show()

# 混淆矩阵
cm=confusion_matrix(test_labels,test_prediction)

plt.rcParams['font.family'] = 'Times New Roman'
plt.title('Confusion Matrix',fontsize=12)
# cm_1 = cm/cm.sum(axis=0) #转换成小数
cm_1 = cm #转换成小数
# cm_2 = pd.DataFrame(cm_1, index=['Normal','Engine','Elevator','Rudder','Aileron'], columns=['Normal','Engine','Elevator','Rudder','Aileron'])
# cm_2 = pd.DataFrame(cm_1, index=['Normal','GPS','Accelerometer','Engine','RC'], 
#                     columns=['Normal','GPS','Accelerometer','Engine','RC'])
# cm_2 = pd.DataFrame(cm_1, index=['Normal','Dos Attack','Replay Attack'], 
#                     columns=['Normal','Dos Attack','Replay Attack'])
cm_2 = pd.DataFrame(cm_1, index=['benign_traffic','gafgyt_attacks combo','gafgyt_attacks junk','gafgyt_attacks scan','gafgyt_attacks tcp','gafgyt_attacks udp','mirai_attacks ack','mirai_attacks scan','mirai_attacks syn','mirai_attacks udp','mirai_attacks udpplain'], 
                    columns=['benign_traffic','gafgyt_attacks combo','gafgyt_attacks junk','gafgyt_attacks scan','gafgyt_attacks tcp','gafgyt_attacks udp','mirai_attacks ack','mirai_attacks scan','mirai_attacks syn','mirai_attacks udp','mirai_attacks udpplain'])
# cm_2 = pd.DataFrame(cm_1, index=['Normal','Engine','Elevator','Rudder','Aileron'], 
#                     columns=['Normal','Engine','Elevator','Rudder','Aileron'])
# cm_2 = pd.DataFrame(cm_1, index=['Normal','GPS Jamming','GPS Spoofing'], 
#                     columns=['Normal','GPS Jamming','GPS Spoofing'])
# cm_2 = pd.DataFrame(cm_1, index=['Normal','GPS','Accelerometer','Engine','RC'], 
#                     columns=['Normal','GPS','Accelerometer','Engine','RC'])
# cm_2 = pd.DataFrame(cm_1, index=['Normal','Dos Attack','Replay Attack'], 
#                     columns=['Normal','Dos Attack','Replay Attack'])
sns.heatmap(cm_2, annot=True, annot_kws={"size": 10}, cmap="Blues",fmt='.20g')
plt.ylabel('True label', fontsize=12)
plt.xlabel('Predicted label', fontsize=12)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

plt.show()'''
# with open('./correct/ITS_acc_loss_13_fineTuningDAE.json','w') as f:
#     json.dump(history.history,f)
