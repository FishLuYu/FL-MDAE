from sklearn.calibration import partial
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Dense
from keras.layers import Input,Dense
from keras.models import Model
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np

import os
import math
import argparse
from typing import Dict, List, Tuple

import tensorflow as tf
from imblearn.over_sampling import SMOTE

import flwr as fl
from flwr.common import Metrics, NDArrays, Scalar
from flwr.simulation.ray_transport.utils import enable_tf_gpu_growth
import json
from typing import Callable, Dict, List, Optional, Tuple, Union
# tf.compat.v1.disable_eager_execution()
tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()


#----------------------------start FL-------------------------------------------------------------------

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

parser = argparse.ArgumentParser(description="Flower Simulation with Tensorflow/Keras")

parser.add_argument(
    "--num_cpus",
    type=int,
    default=16,
    help="Number of CPUs to assign to a virtual client",
)
parser.add_argument(
    "--num_gpus",
    type=float,
    default=0.89,
    help="Ratio of GPU memory to assign to a virtual client",
)
parser.add_argument("--num_rounds", type=int, default=40, help="Number of FL rounds.")

NUM_CLIENTS = 10
# NUM_CLIENTS = 5

VERBOSE = 0
EPOCH=20
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, x_train, y_train, x_val, y_val) -> None:
        # Create model
        self.model = get_model()
        self.x_train, self.y_train = x_train, y_train
        self.x_val, self.y_val = x_val, y_val

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(
            self.x_train, self.y_train, epochs=5, batch_size=32, verbose=VERBOSE
        )
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(
            self.x_val, self.y_val, batch_size=64, verbose=VERBOSE
        )
        return loss, len(self.x_val), {"accuracy": acc}
    
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
def get_model():
    
    print("开始获取模型...")
    orgAEModel=tf.keras.models.load_model('./models/best/AWID_Sig_DAE_MultiModel_CheckPoint.h5', custom_objects={'custom_loss': custom_loss})
    # orgAEModel.summary()
    # 冻结所有参数(0-13)/全部
    for layer in orgAEModel.layers[0:12]:
        layer._name = layer.name + str("testhello")
        layer.trainable=False
    # 添加全连接层分类器
    x=orgAEModel.output
    x=tf.keras.layers.Dense(256,activation='relu',name='fineDense')(x)

    x=tf.keras.layers.Dense(128,activation='relu',name='fineDense2')(x)

    out=tf.keras.layers.Dense(11,activation='softmax')(x)

    fineTuningModel=Model(inputs=orgAEModel.input,outputs=out)

    # fineTuningModel.summary()
    fineTuningModel.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"],run_eagerly=True)
    print("获取模型成功...")
    return fineTuningModel

def get_client_fn(dataset_partitions): # Return a function to construc a client
    def client_fn(cid:str)-> fl.client.Client:
        x_train,y_train=dataset_partitions[int(cid)]
        split_idx=math.floor(len(x_train)*0.9)

        x_train_cid,y_train_cid=(x_train[:split_idx],y_train[:split_idx],)
        x_val_cid,y_val_cid=x_train[split_idx:],y_train[split_idx:]

        return FlowerClient(x_train_cid,y_train_cid,x_val_cid,y_val_cid)
    
    return client_fn

# path_noise=r"D:/test_flower_tf/data/ALFA_ENG_ELE_RU_AIL_imu-data_imu-mag_Pitch-Roll-Yaw_Noise_Abnormal_andNormal.csv" # 5分类，噪声数据
# path_noise=r"D:/test_flower_tf/data/UAVGPSAttack_imu-0_mag_Pitch-Roll_Yaw_Jamming_Spoofing_DownSample_Noise.csv"# 3分类
# path_noise=r"D:/test_flower_tf/data/TLM_imu-0_mag_Pitch-Roll_Yaw_Noise.csv"# 5 分类
# path_noise=r"D:/test_flower_tf/data/3class-Dataset_T-Cyber_Encoded_Noise_Time.csv" # 3分类
# path_noise=r"./data/ITS-Physical-3_noise.csv" # 3分类
# path_noise=r"./data/BAT-Data_Train_noise.csv" # 11分类
# path_noise=r"./BAT_combined_data_noise.csv" # 11分类
path_noise=r"data/awid-dataset_processed_30_noise.csv" # 11分类


def partition_UAV_Dataset(path=path_noise):
    print("开始划分数据集...")

    data=pd.read_csv(path,header=0)
    data=data.sample(frac=1)
    # data=data.drop("timestamp_c",axis=1) # ITS_Cyber正常数据里面有时间戳
    # data=data.drop("timestamp_p",axis=1) # ITS_Cyber正常数据里面有时间戳

#----------------------------------------#
    X_data=data.drop(['labels'],axis=1).values
    Y_data=data['labels'].values
    
    # # 扩展
    # from imblearn.over_sampling import SMOTE
    # smote=SMOTE(n_jobs=-1,sampling_strategy={4:3500,5:3500,6:3500,7:3500,8:3500,9:3500,10:3500,})
    # X_data, Y_data = smote.fit_resample(X_data, Y_data)

   ##############################################Physical扩展##############################################
    '''print("之前：",pd.Series(Y_data).value_counts())

    smote=SMOTE(n_jobs=-1,sampling_strategy={1:4290,2:4290})
    X_data, Y_data = smote.fit_resample(X_data, Y_data)

    print("之后：",pd.Series(Y_data).value_counts())'''

    #############################################Physical扩展##############################################

    max_min=StandardScaler()
    X_data=max_min.fit_transform(X_data)
#----------------------------------------#
    train_data,test_data,train_labels,test_labels=train_test_split(X_data,Y_data,test_size=0.2)

    partitions=[]
    NUM_CLIENTS=10
    partition_size=math.floor(len(X_data)/NUM_CLIENTS)
    # print(partition_size)

    for cid in range(NUM_CLIENTS):
        idx_from,idx_to=int(cid)*partition_size,(int(cid)+1)*partition_size
    # print(idx_from,idx_to)
        partitions.append((train_data[idx_from:idx_to] , train_labels[idx_from:idx_to]))
    print("数据集划分成功...")
    return partitions,test_data,test_labels

def weighted_average(metrics: List[Tuple[int, Metrics]])-> Metrics:
    '''
    Aggregation function for (federated) evaluation metrics.

    It ill aggregate those metrics returned by the client's evaluate() method.
    '''
    # Multiply accuracy of each client by number of examples used
    accuracies=[num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

def get_evaluate_fn(test_data,test_labels):
    """Return an evaluation function for server-side (i.e. centralised) evaluation."""
    def evaluate(server_round:int,parameters:fl.common.NDArrays,config:Dict[str,fl.common.Scalar]):
        model=get_model()
        model.set_weights(parameters)

        loss,accuracy=model.evaluate(test_data,test_labels,verbose=VERBOSE)
        #-------------------------------------------
        test_predictions = model.predict(test_data)
        test_prediction=np.argmax(test_predictions,axis=1)
        # accuracy = accuracy_score(test_labels,test_prediction)
        # print('Test Accuracy:', accuracy)
        precision = precision_score(test_labels, test_prediction, average='macro')
        print('-----------------------Precision----------------------------:', precision)

        f1=f1_score(test_labels,test_prediction,average='micro')
        print("-----------------------F1-----------------------------------:",f1)

        recall = recall_score(test_labels, test_prediction, average='macro')
        print("-----------------------Recall-------------------------------:",recall)

        return loss,{"accuracy_test":accuracy}
        # return {"loss":loss},{"accuracy_test":accuracy},{"precision":precision},{"f1":f1},{"recall":recall}
    
    
    return evaluate

# 没用
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[
            Union[
                Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes],
                BaseException,
            ]
        ],
    ) -> Optional[fl.common.NDArrays]:
        weights = super().aggregate_fit(server_round, results, failures)
        if weights is not None:
            # Save weights
            print(f"Saving round {server_round} weights...")
            np.savez(f"round-{server_round}-weights.npz", *weights)
        return weights


def Main()->None:
    args=parser.parse_args()

    partitions,test_data,test_labels=partition_UAV_Dataset()
    strategy=fl.server.strategy.FedAvg(
        fraction_fit=0.1,# Sample 10% of available clients for training
        fraction_evaluate=0.05,# Sample 5% of available clients for evaluation
        min_fit_clients=6,  # Never sample less than 10 clients for training
        min_evaluate_clients=5,  # Never sample less than 5 clients for evaluation
        min_available_clients=int(NUM_CLIENTS*0.75),# Wait until at least 75 clients are available
        evaluate_metrics_aggregation_fn=weighted_average,  # aggregates federated metrics
        evaluate_fn=get_evaluate_fn(test_data,test_labels),  # global evaluation function

    )

    client_resources={"num_cpus":args.num_cpus,"num_gpus":args.num_gpus}

    # Start Simulation
    history=fl.simulation.start_simulation(
        client_fn=get_client_fn(partitions),
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
        client_resources=client_resources,
        actor_kwargs={
            "on_actor_init_fn":enable_tf_gpu_growth
        },
    )

    # with open('./correct/FL_40_test.json','w') as f:
    #     #  json.dump(history.metrics_centralized['accuracy_test'],f)
    #      json.dump(history,f)


if __name__=="__main__":
    enable_tf_gpu_growth()
    Main()