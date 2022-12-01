#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 08:12:04 2021

@author: amms227
"""


import os
os.environ["TF_KERAS"]="1"  # when tensorflow<2.0
os.environ["CUDA_VISIBLE_DEVICES"]="1" # 使用编号为1，2号的GPU
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for device in gpus:
    tf.config.experimental.set_memory_growth(device, enable=True)





import pandas as pd
# import time
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from keras.utils import np_utils,to_categorical


##############################################################################################################
##############################################################################################################
database='eicu'
timesteps=4
labelname=['hemo_shock_label']
fore_len=4  # 在模型训练过程中的步长
fore_len_in_labelcsv=4  #csv 文件中实际的预测步长

param_name=['icustay_id','age','gender_0','gender_1','bmi','chart_time','gcs','gcseyes','gcsmotor','gcsverbal','tidalvolume',\
  'peep','fio2','ph','pco2','po2','urineoutput','sysbp','diasbp','heartrate','spo2','resprate','tempc']
comtest = pd.read_csv('comtest_'+database+'_timesteps_4_gap_4_forelen_4_param.csv',usecols=param_name)
comtest_age=comtest.age.copy()

# comtest = pd.read_csv('comtest_'+database+'_timesteps_4_gap_4_forelen_4_param.csv')
param_name=list(comtest)

icustay_id=np.array(list(set(comtest.icustay_id)));data_dim=comtest.shape[1]
param_name=list(comtest)
scaler = StandardScaler()   #对病例数据进行标准化处理
comtest.iloc[:,1:comtest.shape[1]] = scaler.fit_transform(comtest.iloc[:,1:comtest.shape[1]])

 
label_name=[]
for j in labelname:
    for i in range(1,fore_len_in_labelcsv+1):
        label_name=label_name+[j+'_'+str(i)]
label_name=['icustay_id']+label_name
comtest_label = pd.read_csv('comtest_'+database+'_timesteps_4_gap_4_forelen_4_'+labelname[0]+'.csv',usecols=label_name)

label_localtion_pre = ((comtest_label.iloc[:,1:]==0)+0)+((comtest_label.iloc[:,1:]==1)+0);
label_localtion = label_localtion_pre.iloc[:,0:fore_len].sum(axis=1);label_localtion[label_localtion<fore_len_in_labelcsv]=fore_len_in_labelcsv                                                


comtest = pd.DataFrame(np.array(comtest).reshape(-1, timesteps, data_dim)[label_localtion>=fore_len,:,:].reshape(-1, data_dim),columns=param_name)
comtest_label = comtest_label[label_localtion>=fore_len].iloc[:,:fore_len+1];comtest_label[comtest_label==0.5]=-1



comtest=comtest[comtest_age>50]
comtest_label=comtest_label[comtest_age[comtest_age.index%4==0].reset_index().age>50]



## 目前label中没有索引ID，正在解决
x_train_id, x_test_id, y_train_id, y_test_id = model_selection.train_test_split(np.array(icustay_id),range(len(icustay_id)), test_size = 0.2,random_state = 1) 
x_test=comtest.loc[comtest['icustay_id'].isin(x_test_id)].iloc[:,1:comtest.shape[1]];y_test=comtest_label.loc[comtest_label['icustay_id'].isin(x_test_id)].iloc[:,1:]

x_train_id, x_val_id, y_train_id, y_val_id = model_selection.train_test_split(np.array(x_train_id),range(len(x_train_id)), test_size = 0.2,random_state = 1) 
x_train=comtest.loc[comtest['icustay_id'].isin(x_train_id)].iloc[:,1:comtest.shape[1]];y_train=comtest_label.loc[comtest_label['icustay_id'].isin(x_train_id)].iloc[:,1:]
x_val=comtest.loc[comtest['icustay_id'].isin(x_val_id)].iloc[:,1:comtest.shape[1]];y_val=comtest_label.loc[comtest_label['icustay_id'].isin(x_val_id)].iloc[:,1:]


# x_train=x_train.iloc[0:int(x_train.shape[0]/timesteps/2)*timesteps*2,:]
# y_train=y_train.iloc[0:int(y_train.shape[0]/timesteps/2)*timesteps*2]
# x_val=x_val.iloc[0:int(x_val.shape[0]/timesteps/2)*timesteps*2,:]
# y_val=y_val.iloc[0:int(y_val.shape[0]/timesteps/2)*timesteps*2]   
# x_test=x_test.iloc[0:int(x_test.shape[0]/timesteps/2)*timesteps*2,:]
# y_test=y_test.iloc[0:int(y_test.shape[0]/timesteps/2)*timesteps*2] 


data_dim=x_train.shape[1] 
x_train=np.array(x_train).reshape(-1, timesteps, data_dim) 
x_val=np.array(x_val).reshape(-1, timesteps, data_dim) 
x_test=np.array(x_test).reshape(-1, timesteps, data_dim) 


for i in ['y_train','y_val','y_test']:
    exec(i+'='+i+'.max(axis=1)')
batch_size=64
x_train=x_train[0:int(x_train.shape[0]/batch_size)*batch_size,:,:]
y_train=y_train[0:int(y_train.shape[0]/batch_size)*batch_size]
x_val=x_val[0:int(x_val.shape[0]/batch_size)*batch_size,:,:]
y_val=y_val[0:int(y_val.shape[0]/batch_size)*batch_size]   
x_test=x_test[0:int(x_test.shape[0]/batch_size)*batch_size,:,:]
y_test=y_test[0:int(y_test.shape[0]/batch_size)*batch_size] 


# for i in ['y_train','y_val','y_test']:
#     exec(i+'=np_utils.to_categorical('+i+'.max(axis=1), num_classes=2)')
# batch_size=64
# x_train=x_train[0:int(x_train.shape[0]/batch_size)*batch_size,:,:]
# y_train=y_train[0:int(y_train.shape[0]/batch_size)*batch_size,:]
# x_val=x_val[0:int(x_val.shape[0]/batch_size)*batch_size,:,:]
# y_val=y_val[0:int(y_val.shape[0]/batch_size)*batch_size,:]   
# x_test=x_test[0:int(x_test.shape[0]/batch_size)*batch_size,:,:]
# y_test=y_test[0:int(y_test.shape[0]/batch_size)*batch_size,:] 

    
##############################################################################################################
##############################################################################################################

from tensorflow.keras.utils import multi_gpu_model

from keras_self_attention import SeqSelfAttention
model = load_model('breast_cancer_model_for_hemoshocklabel_fore_len_4_without_stateful_and_ni_param',custom_objects=SeqSelfAttention.get_custom_objects())
# model.save('breast_cancer_model_for_hemoshocklabel_fore_len_4_without_stateful')
# model = multi_gpu_model(model, gpus=2) #设置多GPU并行

from tensorflow.keras.models import Sequential, Model,load_model
from tensorflow.keras.layers import LSTM, Dropout,Activation,Dense,Bidirectional,BatchNormalization,\
    Conv1D,Input,Permute,RepeatVector,Lambda,MaxPool1D
    
# from keras.utils import np_utils,to_categorical
from tensorflow.python.keras.utils.np_utils import to_categorical 
from tensorflow.python.keras.utils import np_utils

from tensorflow.keras.utils import multi_gpu_model

from tensorflow.keras import metrics 

from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import LearningRateScheduler
import math

def step_decay(epoch):   #学习率指数衰减，通过在固定的epoch周期将学习速率降低x%实现
    initial_lrate =  0.001 #初始学习率定为0.001
    # initial_lrate = 0.0001
    drop = 0.7 #学习率降低50%
    epochs_drop = 1.0  #每2个epochs降低一次
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

model = Sequential()
model.add(Conv1D(60,kernel_size=4,batch_input_shape=(None, 4, 22)))

model.add(Bidirectional(LSTM(100, return_sequences=True, stateful=False)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(SeqSelfAttention(attention_activation='relu'))
model.add(Dropout(0.5))

model.add(Bidirectional(LSTM(100,  stateful=False))) 
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))  
model.compile(loss='binary_crossentropy', metrics=['accuracy','AUC'],optimizer='rmsprop')

##########################################################################################################################


x_train = np.random.random((64*100, 4, 22))
y_train = np.round(np.random.random((64*100, 1)))

# 生成虚拟验证数据
x_val = np.random.random((64*10,  4, 22))
y_val = np.round(np.random.random((64*10, 1)))


result = model.fit(x_train, y_train, 
          epochs=1,
          verbose=1, #0 为不在标准输出流输出日志信息；1，输出进度条记录；2，为每个epoch输出一行记录
          validation_data=(x_val, y_val),
          callbacks=[LearningRateScheduler(step_decay)],
          # class_weight={1:454,0:1},
          batch_size=64
          )     




# x_val=pd.DataFrame(x_val,param_name)


import shap
param_num=22

import time
s=time.time()
background = x_train[np.random.choice(x_train.shape[0], 15000, replace=False)]
explainer = shap.DeepExplainer(model, background)
shap_values = explainer.shap_values(x_val[:15000])
print((time.time()-s)/60)
pd.DataFrame(np.array(shap_values).reshape(-1,22)).to_csv('shap_values_for_x_val_age_more_than_50.csv',index=False)


shap_values = explainer.shap_values(x_val[1:5])
shap.image_plot(shap_values, -x_val[1:5])

shap_values = pd.read_csv('shap_values_for_x_val.csv')
shap_values=np.array(shap_values).reshape(-1,4,22)
shap_values=shap_values.reshape(shap_values.shape[1],shap_values.shape[2],shap_values.shape[3])

# pd.DataFrame(np.array(shap_values).reshape(-1,22)).to_csv('shap_values_for_x_val.csv',index=False)

# param shapley value show with plot_type="bar"
background = x_train[np.random.choice(x_train.shape[0], 5000, replace=False)]
explainer = shap.DeepExplainer(model, background)
shap_values = explainer.shap_values(x_train[:1000])
shap.summary_plot(np.array(shap_values).reshape(-1,param_num),  \
                  pd.DataFrame(x_val[:15000].reshape(-1,param_num),columns=param_name[1:]),\
                      plot_type="bar")
param_shapley_value = np.abs(np.array(shap_values).reshape(-1,param_num)).mean(0)


# 对个体中哪些特征对预测结果起了什么作用做分析
shap.force_plot(explainer.expected_value, shap_values[0,0,:], pd.DataFrame(x_val[1,0,:].reshape(-1,param_num),columns=param_name[1:]),matplotlib=True)

# 将上个图换个方向，并将多人的预测分析拼接起来  需要在jupyter notebook上绘制，且载入shap后要加上shap.initjs()命令保证正常出图
# 该图可用来分析随时间变化某患者参数对预警影响的变化
# shap.force_plot(explainer.expected_value[1], shap_values[1][:1000,:], x_train.iloc[:1000,:])

# 在数据整体上对参数什么样的值对预测有什么影响做分析,
shap.summary_plot(shap_values.reshape(-1,param_num), pd.DataFrame(x_val[1:5,:,:].reshape(-1,param_num),columns=param_name[1:]))

# shap 参数依赖关系图
shap.dependence_plot("heartrate", shap_values.reshape(-1,param_num), \
                     pd.DataFrame(x_val[1:5,:,:].reshape(-1,param_num),columns=param_name[1:]), \
                         interaction_index="tempc") # when disapper interaction_index,then 参数依赖关系图;if interaction_index=None then part参数依赖关系图 
# shap.dependence_plot('age', shap_values, data[cols], interaction_index=None, show=False)

    
import matplotlib.pylab as pl
import numpy as np


# param_timesteps_shapley_value=pd.DataFrame(np.array(shap_values).mean(1).reshape(-1,param_num),columns=param_name[1:])

shap_value_for_ana_charttime=pd.DataFrame(np.array(shap_values).reshape(-1,param_num),columns=param_name[1:])

'''
#############################################################################################################
#############################################################################################################
'''
param_name=['icustay_id','age','gender_0','gender_1','bmi','chart_time','gcs','gcseyes','gcsmotor','gcsverbal','tidalvolume',\
  'peep','fio2','ph','pco2','po2','urineoutput','sysbp','diasbp','heartrate','spo2','resprate','tempc']
comtest = pd.read_csv('comtest_'+database+'_timesteps_4_gap_4_forelen_4_param.csv',usecols=param_name)

# comtest = pd.read_csv('comtest_'+database+'_timesteps_4_gap_4_forelen_4_param.csv')
param_name=list(comtest)

icustay_id=np.array(list(set(comtest.icustay_id)));data_dim=comtest.shape[1]
param_name=list(comtest)
# scaler = StandardScaler()   #对病例数据进行标准化处理
# comtest.iloc[:,1:comtest.shape[1]] = scaler.fit_transform(comtest.iloc[:,1:comtest.shape[1]])

label_name=[]
for j in labelname:
    for i in range(1,fore_len_in_labelcsv+1):
        label_name=label_name+[j+'_'+str(i)]
label_name=['icustay_id']+label_name
comtest_label = pd.read_csv('comtest_'+database+'_timesteps_4_gap_4_forelen_4_'+labelname[0]+'.csv',usecols=label_name)

label_localtion_pre = ((comtest_label.iloc[:,1:]==0)+0)+((comtest_label.iloc[:,1:]==1)+0);
label_localtion = label_localtion_pre.iloc[:,0:fore_len].sum(axis=1);label_localtion[label_localtion<fore_len_in_labelcsv]=fore_len_in_labelcsv                                                


comtest = pd.DataFrame(np.array(comtest).reshape(-1, timesteps, data_dim)[label_localtion>=fore_len,:,:].reshape(-1, data_dim),columns=param_name)
comtest_label = comtest_label[label_localtion>=fore_len].iloc[:,:fore_len+1];comtest_label[comtest_label==0.5]=-1

## 目前label中没有索引ID，正在解决
x_train_id, x_test_id, y_train_id, y_test_id = model_selection.train_test_split(np.array(icustay_id),range(len(icustay_id)), test_size = 0.2,random_state = 1) 
x_test=comtest.loc[comtest['icustay_id'].isin(x_test_id)].iloc[:,1:comtest.shape[1]];y_test=comtest_label.loc[comtest_label['icustay_id'].isin(x_test_id)].iloc[:,1:]

x_train_id, x_val_id, y_train_id, y_val_id = model_selection.train_test_split(np.array(x_train_id),range(len(x_train_id)), test_size = 0.2,random_state = 1) 
x_train=comtest.loc[comtest['icustay_id'].isin(x_train_id)].iloc[:,1:comtest.shape[1]];y_train=comtest_label.loc[comtest_label['icustay_id'].isin(x_train_id)].iloc[:,1:]
x_val=comtest.loc[comtest['icustay_id'].isin(x_val_id)].iloc[:,1:comtest.shape[1]];y_val=comtest_label.loc[comtest_label['icustay_id'].isin(x_val_id)].iloc[:,1:]

data_dim=x_train.shape[1] 
x_train=np.array(x_train).reshape(-1, timesteps, data_dim) 
x_val=np.array(x_val).reshape(-1, timesteps, data_dim) 
x_test=np.array(x_test).reshape(-1, timesteps, data_dim) 

for i in ['y_train','y_val','y_test']:
    exec(i+'='+i+'.max(axis=1)')
batch_size=64
x_train=x_train[0:int(x_train.shape[0]/batch_size)*batch_size,:,:]
y_train=y_train[0:int(y_train.shape[0]/batch_size)*batch_size]
x_val=x_val[0:int(x_val.shape[0]/batch_size)*batch_size,:,:]
y_val=y_val[0:int(y_val.shape[0]/batch_size)*batch_size]   
x_test=x_test[0:int(x_test.shape[0]/batch_size)*batch_size,:,:]
y_test=y_test[0:int(y_test.shape[0]/batch_size)*batch_size] 


# z = pd.DataFrame(np.array(x_train[:5000]).reshape(-1,param_num),columns=param_name[1:])
'''
#############################################################################################################
#############################################################################################################
'''

shap_value_for_ana_charttime.chart_time=pd.DataFrame(np.array(x_val[:15000]).reshape(-1,param_num),columns=param_name[1:]).chart_time
# shap_value_for_ana_charttime = shap_value_for_ana_charttime.groupby('chart_time',as_index=False).mean()
shap_value_for_ana_charttime = shap_value_for_ana_charttime.abs().groupby('chart_time',as_index=True).mean()


# 绘制热力图
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
ax = sns.heatmap(shap_value_for_ana_charttime.iloc[:72,:].T, cmap="YlGnBu")   # cmap是热力图颜色的参数
plt.show()



z=pd.DataFrame(np.abs(shap_values).mean(0),columns=param_name[1:]);z['timesteps']=[0,1,2,3];z=z.groupby('timesteps',as_index=True).mean()

sns.set()
ax = sns.heatmap(z.T, cmap="YlGnBu")   # cmap是热力图颜色的参数
plt.show()


# import time
# s_time = time.time()
# shap_values = explainer.shap_values(x_train[:5000])
# print(np.round((time.time() - s_time)/60))


background=np.vstack((x_train,x_val,x_test))
background_y=pd.concat([y_train,y_val,y_test])
explainer = shap.DeepExplainer(model, np.vstack((background[background_y==1], background[0:1300])))
shap_values = explainer.shap_values(np.vstack((background[background_y==1], background[0:1300])))

shap_values=np.array(shap_values).reshape(-1,4,22)
shap_values=shap_values.reshape(shap_values.shape[1],shap_values.shape[2],shap_values.shape[3])

z=pd.DataFrame(np.abs(shap_values).mean(1).reshape(4,22),columns=param_name[1:]);z['timesteps']=[0,1,2,3];z=z.groupby('timesteps',as_index=True).mean()

sns.set()
ax = sns.heatmap(z.T, cmap="YlGnBu")   # cmap是热力图颜色的参数
plt.show()
