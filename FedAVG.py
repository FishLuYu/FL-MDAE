import math
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
import numpy as np

# path=r"D:/test_flower_tf/data/ALFA_ENG_ELE_RU_AIL_imu-data_imu-mag_Pitch-Roll-Yaw.csv" # 5分类,正常数据
# path=r"D:/test_flower_tf/data/UAVGPSAttack_imu-0_mag_Pitch-Roll_Yaw_Jamming_Spoofing_DownSample.csv"
# path=r"D:/test_flower_tf/data/TLM_imu-0_mag_Pitch-Roll_Yaw.csv"# 5 分类


path=r"D:/test_flower_tf/data/ALFA_ENG_ELE_RU_AIL_imu-data_imu-mag_Pitch-Roll-Yaw_Noise_Abnormal_andNormal.csv" # 5分类，带噪声
# path=r"D:/test_flower_tf/data/ALFA_ENG_ELE_RU_AIL_imu-data_imu-mag_Pitch-Roll-Yaw.csv" # 5分类,正常数据
# path=r"D:/test_flower_tf/data/UAVGPSAttack_imu-0_mag_Pitch-Roll_Yaw_Jamming_Spoofing_DownSample_Noise.csv"# 3分类
# path=r"D:/test_flower_tf/data/UAVGPSAttack_imu-0_mag_Pitch-Roll_Yaw_Jamming_Spoofing_DownSample.csv"

# path=r"D:/test_flower_tf/data/TLM_imu-0_mag_Pitch-Roll_Yaw_Noise.csv"# 3分类

# data=pd.read_csv(path,header=0)
# data=data.drop("timestamp",axis=1)
# data=data.sample(frac=1)

# #----------------------------------------#
# X_data=data.drop(['labels'],axis=1).values
# Y_data=data['labels'].values

# 加载CSV数据集
def load_csv_data(file_path):
    df = pd.read_csv(file_path)
    X = df.drop('labels', axis=1).values
    y = df['labels'].values
    return X, y

# 创建模型
def create_model():
    model = keras.Sequential([
        layers.Input(shape=(19,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(5, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# 模拟分布式数据
def get_federated_data(X, y, clients=10):
    client_data = []
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for i in range(clients):
        start_idx = i * len(X_train) // clients
        end_idx = (i + 1) * len(X_train) // clients
        client_data.append({'x': X_train[start_idx:end_idx], 'y': y_train[start_idx:end_idx]})

    return client_data, {'x': X_test, 'y': y_test}

# 联邦平均算法
def federated_average(model, client_models):
    global_weights = model.get_weights()
    for client_model in client_models:
        client_weights = client_model.get_weights()
        global_weights = [global_w + client_w for global_w, client_w in zip(global_weights, client_weights)]
    global_weights = [global_w / len(client_models) for global_w in global_weights]
    model.set_weights(global_weights)

# 主训练循环
def federated_learning(num_rounds, clients_per_round, file_path):
    X, y = load_csv_data(file_path)
    global_model = create_model()

    for round in range(num_rounds):
        client_data, test_data = get_federated_data(X, y, clients=clients_per_round)

        client_models = []
        for data in client_data:
            client_model = create_model()
            client_model.set_weights(global_model.get_weights())
            client_model.fit(data['x'], data['y'], epochs=1, verbose=0)
            client_models.append(client_model)

        federated_average(global_model, client_models)

        loss, accuracy = global_model.evaluate(test_data['x'], test_data['y'])
        print(f"Round {round + 1}, Accuracy: {accuracy}")

# 调用主训练循环
federated_learning(num_rounds=10, clients_per_round=5, file_path=path)