import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def add_gaussian_noise(data, mean=0.1, std=0.2):
    noise = np.random.normal(mean, std, data.shape)
    noisy_data = data + noise
    return noisy_data

# path=r"D:/test_flower_tf/data/ALFA_ENG_ELE_RU_AIL_imu-data_imu-mag_Pitch-Roll-Yaw.csv" # 5分类
# path=r"D:/test_flower_tf/data/UAVGPSAttack_imu-0_mag_Pitch-Roll_Yaw_Jamming_Spoofing_DownSample.csv" # 3分类
# path=r"D:/test_flower_tf/data/TLM_imu-0_mag_Pitch-Roll_Yaw.csv" # 3分类
# path=r"D:/test_flower_tf/data/3class-Dataset_T-Cyber_Encoded.csv" # 3分类
# path=r"D:/test_flower_tf/data/processed_friday_dataset_CIC.csv" # 3分类

# path=r'D:/FL-MDAE/data/IoT2023_7.csv' #7分类
# data=pd.read_csv(path,header=0)
# path=r'data/BAT-Data_Train.csv' # 3分类
# path=r'D:/FL-MDAE/data/UNSW_NB15_training-set_OK.csv' # 10分类
path=r'data/awid-dataset_processed_30.csv' # 10分类


data=pd.read_csv(path)

data_labels=data['labels'] #取出标签
data=data.drop(["labels"],axis=1) # 标签和时间列不用添加噪声

print("之前--------------------------")
print(data)
print("Addition of noise......")

noise_df=pd.DataFrame(columns=data.columns)
labels=pd.DataFrame(data=data_labels,columns=['labels'])
# print(labels)
# print(noise_df)
for col in data.columns:
    # print(col) # 列名字
    # print(data[col])# 取出列名为col的列
    noise_data=add_gaussian_noise(data)
    noise_df=data+noise_data

print("之后--------------------------")
print(noise_df)

# 将噪声数据与标签拼接
result = pd.concat([noise_df, labels], axis=1)

print(result.shape)
print("Save file......")
result.to_csv("data/awid-dataset_processed_30_noise.csv",index=False)











# ---------------------------------------------------------------------------------
'''# 准备数据
x = [i for i in range(data.shape[0])]
# x = data['%time']

y = data['field.orientation.x']
y_noise=noise_df['field.orientation.x']

yy = data['field.orientation.y']
yy_noise=add_gaussian_noise(yy)
# 绘制图像
plt.plot(x, y_noise, color='#1e592b', linestyle='-', linewidth=2, markersize=5)

plt.plot(x, y, color='#ee6a5b', linestyle='-', linewidth=2, markersize=5)

plt.xlabel('TimeStamp', fontsize=14)
plt.ylabel('Value', fontsize=14)
plt.title('TimeStamp vs Orientation', fontsize=16, fontweight='bold')
plt.xticks(np.arange(0,data.shape[0],5000))
# plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9])
# plt.set_xticks(np.arange(1, data.shape[0], 1000))
plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
plt.tick_params(axis='both', which='major', labelsize=12)

# 添加图例
plt.legend(['field.orientation.x_noise','field.orientation.x'], loc='lower right', fontsize=12)
# plt.savefig('./Experiment_Result_Acc/添加噪声前后_大数据集.svg',format='svg')

plt.show()'''