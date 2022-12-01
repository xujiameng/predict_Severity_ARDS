#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 19 18:59:08 2022

@author: xujiameng
"""




import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection


from sklearn.metrics import roc_auc_score,accuracy_score,f1_score,precision_recall_curve,matthews_corrcoef,confusion_matrix
from numpy import *
# import time
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import feather
import concurrent

from sklearn import ensemble


def cum_95CI(j):
    for i in range(2000):
        # bootstrap by sampling with replacement on the prediction indices
        indices=np.random.randint(0,len(pro_comm_Pre_for_cumci) - 1,int(len(pro_comm_Pre_for_cumci)/2))
        
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
        else:
            break
    
    #    score = roc_auc_score(y_true[indices], y_pred[indices])
    eva_CI = evaluating_indicator(y_true=y_true[indices], y_test=blo_comm_Pre[indices], y_test_value=pro_comm_Pre_for_cumci[indices])
    return pd.DataFrame(eva_CI,index=[0])
    
def cumCI(y_true,pro_comm_Pre):
    y_true=np.array(y_true)
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # pro=list(executor.map(calculate_sampling_frequency,icustay_id))
        pro=list(tqdm(executor.map(cum_95CI,range(0,100)),total=100,desc= ' cum_95CI Processing'))
        # res=list(tqdm(p.imap(function,params),total=len(params),desc='Processing'))
    
    input_mulit=('pro[{}]'.format(0))
    for i in range(1,len(pro)):
        input_mulit=(input_mulit+',pro[{}]'.format(i))
    cum_95CI_pro=pd.concat(eval(input_mulit),axis=0,ignore_index=True)
    
    for i in ['ACC','AUC','AUPRC','BER','F1_score','KAPPA','MCC','TNR','TPR']:   
        sorted_scores = np.array(cum_95CI_pro[i]); sorted_scores.sort()
        print("Confidence interval for the "+i+": [{:0.6f} - {:0.6}]".format(sorted_scores[int(0.025 * len(sorted_scores))], sorted_scores[int(0.975 * len(sorted_scores))]))
    return 


def kappa(matrix):
    n = np.sum(matrix)
    sum_po = 0
    sum_pe = 0
    for i in range(len(matrix[0])):
        sum_po += matrix[i][i]
        row = np.sum(matrix[i, :])
        col = np.sum(matrix[:, i])
        sum_pe += row * col
    po = sum_po / n
    pe = sum_pe / (n * n)
    # print(po, pe)
    return (po - pe) / (1 - pe)


def evaluating_indicator(y_true, y_test, y_test_value):  #计算预测结果的各项指标
    c_m = confusion_matrix(y_true, y_test)
    TP=c_m[0,0]
    FN=c_m[0,1]
    FP=c_m[1,0]
    TN=c_m[1,1]
    
    TPR=TP/ (TP+ FN) #敏感性
    TNR= TN / (FP + TN) #特异性
    BER=1/2*((FP / (FP + TN) )+FN/(FN+TP))
    
    ACC = accuracy_score(y_true, y_test)
    MCC = matthews_corrcoef(y_true, y_test)
    F1score =  f1_score(y_true, y_test)
    AUC = roc_auc_score(y_true,y_test_value)
    
    precision , recall,thresholds = precision_recall_curve(y_true,y_test_value)
    AUPRC=(precision[1:]*(recall[:-1]-recall[1:])).sum()
    KAPPA=kappa(c_m)
    c={"TPR" : TPR,"TNR" : TNR,"BER" : BER
    ,"ACC" : ACC,"MCC" : MCC,"F1_score" : F1score,"AUC" : AUC,'KAPPA':KAPPA
    ,"AUPRC":AUPRC}  #以字典的形式保存预测结果
    return c


def blo(pro_comm_Pre,jj):     #žùŸÝÔ€²âžÅÂÊÓë×îÓÅ·ÖÀàãÐÖµ¶Ô»ŒÕßœøÐÐÉúËÀÔ€²â
    blo_Pre=zeros(len(pro_comm_Pre))
    blo_Pre[(pro_comm_Pre>(jj*0.01))]=1
    return blo_Pre

#import inspect, re
#def varname(p):
#    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
#        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
#        if m:
#            return m.group(1)

def spec_for_ser(df,icustay_id):
    str_df=str(df)
    for i in icustay_id:
        if i==icustay_id[0]:
            input_mulit=(str_df+"["+str_df+"['hospitalid']=={}]".format(i))
        else:
            input_mulit=(input_mulit+","+str_df+"["+str_df+"['hospitalid']=={}]".format(i))
    return (pd.concat(eval(input_mulit),axis=0,ignore_index=True))


learn_len=2;gap=4;fore_len=4;aim_label=3
# csvname='comtest_for_model_'+str(learn_len)+'_'+str(gap)+'_'+str(fore_len)+'_all.feature'
# comtest = feather.read_dataframe(csvname)  
# comtest.drop(['patientunitstayid'],axis=1,inplace=True) #patientunitstayid,hospitalid

# scaler = StandardScaler()   #对病例数据进行标准化处理
# comtest.iloc[:,1:comtest.shape[1]-1]=scaler.fit_transform(comtest.iloc[:,1:comtest.shape[1]-1])
comtest = feather.read_dataframe('aim_label_1.feature')  
icustay_id=list(set(comtest['hospitalid']))

x_train_for_vail, x_test, y_train_for_vail, y_true = model_selection.train_test_split(np.array(icustay_id),range(len(icustay_id)), test_size = 0.6,random_state = 1)    
x_train_for_vail=spec_for_ser('comtest',x_train_for_vail);y_train_for_vail=x_train_for_vail.iloc[:,-1];x_train_for_vail_group=x_train_for_vail.iloc[:,0];x_train_for_vail=x_train_for_vail.iloc[:,1:x_train_for_vail.shape[1]-1]
x_test=spec_for_ser('comtest',x_test);y_true=x_test.iloc[:,-1];x_test_group=x_test.iloc[:,0];x_test=x_test.iloc[:,1:x_test.shape[1]-1]

# background=np.random.choice(x_train_for_vail.shape[0], 100000, replace=False)
y_train_for_vail[y_train_for_vail==aim_label]=10;y_true[y_true==aim_label]=10
y_train_for_vail[y_train_for_vail<=3]=0;y_true[y_true<=3]=0
y_train_for_vail[y_train_for_vail>3]=1;y_true[y_true>3]=1
y_true_roc_gbm=y_true
#######################################################################################
#######################################################################################

######################################## Machine Learning ############################################
import lightgbm as lgb
comm = lgb.LGBMClassifier()
comm.fit(x_train_for_vail , y_train_for_vail) 

pro_comm_Pre = comm.predict_proba(x_train_for_vail)
RightIndex=[]
for jj in tqdm(range(100)): #计算模型在不同分类阈值下的各项指标
    blo_comm_Pre = blo(pro_comm_Pre[:,1],jj)
    eva_comm = evaluating_indicator(y_true=y_train_for_vail, y_test=blo_comm_Pre, y_test_value=pro_comm_Pre[:,1])
    RightIndex.append(abs(eva_comm['TPR'] - eva_comm['TNR']))
RightIndex=np.array(RightIndex,dtype=np.float16)
position=np.argmin(RightIndex)  #选择出使得敏感性特异性最小的阈值作为分类阈值输出，作为测试集上使用的参数保存
pro_comm_Pre = comm.predict_proba(x_test)        
blo_comm_Pre = blo(pro_comm_Pre[:,1],position)
eva_comm = evaluating_indicator(y_true=y_true, y_test=blo_comm_Pre, y_test_value=pro_comm_Pre[:,1])
print(eva_comm)
pro_comm_Pre_for_cumci = pro_comm_Pre[:,1]
pro_comm_Pre_gbm_all = pro_comm_Pre
# cumCI(y_true,pro_comm_Pre[:,1])

comtest.dropna(inplace=True)

x_train_for_vail, x_test, y_train_for_vail, y_true = model_selection.train_test_split(np.array(icustay_id),range(len(icustay_id)), test_size = 0.6,random_state = 1)    
x_train_for_vail=spec_for_ser('comtest',x_train_for_vail);y_train_for_vail=x_train_for_vail.iloc[:,-1];x_train_for_vail_group=x_train_for_vail.iloc[:,0];x_train_for_vail=x_train_for_vail.iloc[:,1:x_train_for_vail.shape[1]-1]
x_test=spec_for_ser('comtest',x_test);y_true=x_test.iloc[:,-1];x_test_group=x_test.iloc[:,0];x_test=x_test.iloc[:,1:x_test.shape[1]-1]

# background=np.random.choice(x_train_for_vail.shape[0], 100000, replace=False)
y_train_for_vail[y_train_for_vail==aim_label]=10;y_true[y_true==aim_label]=10
y_train_for_vail[y_train_for_vail<=3]=0;y_true[y_true<=3]=0
y_train_for_vail[y_train_for_vail>3]=1;y_true[y_true>3]=1
y_true_roc=y_true

# background=np.random.choice(x_train_for_vail.shape[0], 5000, replace=False)
# from sklearn import svm
# comm = svm.SVC(C=0.5, probability=True) 
# comm.fit(x_train_for_vail.iloc[background,:] , y_train_for_vail[background])   
# # pro_comm_Pre = comm.predict_proba(x_train_for_vail.iloc[background,:])
# # RightIndex=[]
# # for jj in tqdm(range(100)): #计算模型在不同分类阈值下的各项指标
# #     blo_comm_Pre = blo(pro_comm_Pre[:,1],jj)
# #     eva_comm = evaluating_indicator(y_true=y_train_for_vail[background], y_test=blo_comm_Pre, y_test_value=pro_comm_Pre[:,1])
# #     RightIndex.append(abs(eva_comm['TPR'] - eva_comm['TNR']))
# # RightIndex=np.array(RightIndex,dtype=np.float16)
# # position=np.argmin(RightIndex)  #选择出使得敏感性特异性最小的阈值作为分类阈值输出，作为测试集上使用的参数保存
# pro_comm_Pre = comm.predict_proba(x_test)        
# blo_comm_Pre = blo(pro_comm_Pre[:,1],position)
# eva_comm = evaluating_indicator(y_true=y_true, y_test=blo_comm_Pre, y_test_value=pro_comm_Pre[:,1])
# print(eva_comm)
# pro_comm_Pre_for_cumci = pro_comm_Pre[:,1]
# pro_comm_Pre_svm_all = pro_comm_Pre
# # cumCI(y_true,pro_comm_Pre[:,1])

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
# comm = DecisionTreeClassifier(splitter='random',min_samples_split=20,
#                           min_samples_leaf=80 ,max_leaf_nodes=None)
comm = RandomForestClassifier()
comm.fit(x_train_for_vail , y_train_for_vail)   
# pro_comm_Pre = comm.predict_proba(x_train_for_vail.iloc[background,:])
# RightIndex=[]
# for jj in tqdm(range(100)): #计算模型在不同分类阈值下的各项指标
#     blo_comm_Pre = blo(pro_comm_Pre[:,1],jj)
#     eva_comm = evaluating_indicator(y_true=y_train_for_vail[background], y_test=blo_comm_Pre, y_test_value=pro_comm_Pre[:,1])
#     RightIndex.append(abs(eva_comm['TPR'] - eva_comm['TNR']))
# RightIndex=np.array(RightIndex,dtype=np.float16)
# position=np.argmin(RightIndex)  #选择出使得敏感性特异性最小的阈值作为分类阈值输出，作为测试集上使用的参数保存
pro_comm_Pre = comm.predict_proba(x_test)        
blo_comm_Pre = blo(pro_comm_Pre[:,1],position)
eva_comm = evaluating_indicator(y_true=y_true, y_test=blo_comm_Pre, y_test_value=pro_comm_Pre[:,1])
print(eva_comm)
pro_comm_Pre_for_cumci = pro_comm_Pre[:,1]
pro_comm_Pre_rf_all = pro_comm_Pre
# cumCI(y_true,pro_comm_Pre[:,1])


# from sklearn.neural_network import MLPClassifier
# # comm = MLPClassifier(hidden_layer_sizes=(20, 20), max_iter=1)
# comm = MLPClassifier()
# comm.fit(x_train_for_vail , y_train_for_vail)   
# # pro_comm_Pre = comm.predict_proba(x_train_for_vail.iloc[background,:])
# # RightIndex=[]
# # for jj in tqdm(range(100)): #计算模型在不同分类阈值下的各项指标
# #     blo_comm_Pre = blo(pro_comm_Pre[:,1],jj)
# #     eva_comm = evaluating_indicator(y_true=y_train_for_vail[background], y_test=blo_comm_Pre, y_test_value=pro_comm_Pre[:,1])
# #     RightIndex.append(abs(eva_comm['TPR'] - eva_comm['TNR']))
# # RightIndex=np.array(RightIndex,dtype=np.float16)
# # position=np.argmin(RightIndex)  #选择出使得敏感性特异性最小的阈值作为分类阈值输出，作为测试集上使用的参数保存
# pro_comm_Pre = comm.predict_proba(x_test)        
# blo_comm_Pre = blo(pro_comm_Pre[:,1],position)
# eva_comm = evaluating_indicator(y_true=y_true, y_test=blo_comm_Pre, y_test_value=pro_comm_Pre[:,1])
# print(eva_comm)
# pro_comm_Pre_for_cumci = pro_comm_Pre[:,1]
# pro_comm_Pre_ann_all = pro_comm_Pre
# # cumCI(y_true,pro_comm_Pre[:,1])


from sklearn.linear_model import LogisticRegression
# comm = LogisticRegression(max_iter=1,warm_start=False,solver='newton-cg')
comm = LogisticRegression()
comm.fit(x_train_for_vail , y_train_for_vail)   
# pro_comm_Pre = comm.predict_proba(x_train_for_vail.iloc[background,:])
# RightIndex=[]
# for jj in tqdm(range(100)): #计算模型在不同分类阈值下的各项指标
#     blo_comm_Pre = blo(pro_comm_Pre[:,1],jj)
#     eva_comm = evaluating_indicator(y_true=y_train_for_vail[background], y_test=blo_comm_Pre, y_test_value=pro_comm_Pre[:,1])
#     RightIndex.append(abs(eva_comm['TPR'] - eva_comm['TNR']))
# RightIndex=np.array(RightIndex,dtype=np.float16)
# position=np.argmin(RightIndex)  #选择出使得敏感性特异性最小的阈值作为分类阈值输出，作为测试集上使用的参数保存
pro_comm_Pre = comm.predict_proba(x_test)        
blo_comm_Pre = blo(pro_comm_Pre[:,1],position)
eva_comm = evaluating_indicator(y_true=y_true, y_test=blo_comm_Pre, y_test_value=pro_comm_Pre[:,1])
print(eva_comm)
pro_comm_Pre_for_cumci = pro_comm_Pre[:,1]
pro_comm_Pre_log_all = pro_comm_Pre
# cumCI(y_true,pro_comm_Pre[:,1])


# from sklearn.naive_bayes import GaussianNB
# a=0.5
# comm = GaussianNB(priors=[a,1-a])
# comm.fit(x_train_for_vail , y_train_for_vail)   
# # pro_comm_Pre = comm.predict_proba(x_train_for_vail.iloc[background,:])
# # RightIndex=[]
# # for jj in tqdm(range(100)): #计算模型在不同分类阈值下的各项指标
# #     blo_comm_Pre = blo(pro_comm_Pre[:,1],jj)
# #     eva_comm = evaluating_indicator(y_true=y_train_for_vail[background], y_test=blo_comm_Pre, y_test_value=pro_comm_Pre[:,1])
# #     RightIndex.append(abs(eva_comm['TPR'] - eva_comm['TNR']))
# # RightIndex=np.array(RightIndex,dtype=np.float16)
# # position=np.argmin(RightIndex)  #选择出使得敏感性特异性最小的阈值作为分类阈值输出，作为测试集上使用的参数保存
# pro_comm_Pre = comm.predict_proba(x_test)        
# blo_comm_Pre = blo(pro_comm_Pre[:,1],position)
# eva_comm = evaluating_indicator(y_true=y_true, y_test=blo_comm_Pre, y_test_value=pro_comm_Pre[:,1])
# print(eva_comm)
# pro_comm_Pre_for_cumci = pro_comm_Pre[:,1]
# pro_comm_Pre_nb_all = pro_comm_Pre
# # cumCI(y_true,pro_comm_Pre[:,1])


# from sklearn.neighbors import KNeighborsClassifier
# comm = KNeighborsClassifier(n_neighbors=3)
# comm.fit(x_train_for_vail , y_train_for_vail)   
# # pro_comm_Pre = comm.predict_proba(x_train_for_vail.iloc[background,:])
# # RightIndex=[]
# # for jj in tqdm(range(100)): #计算模型在不同分类阈值下的各项指标
# #     blo_comm_Pre = blo(pro_comm_Pre[:,1],jj)
# #     eva_comm = evaluating_indicator(y_true=y_train_for_vail[background], y_test=blo_comm_Pre, y_test_value=pro_comm_Pre[:,1])
# #     RightIndex.append(abs(eva_comm['TPR'] - eva_comm['TNR']))
# # RightIndex=np.array(RightIndex,dtype=np.float16)
# # position=np.argmin(RightIndex)  #选择出使得敏感性特异性最小的阈值作为分类阈值输出，作为测试集上使用的参数保存
# pro_comm_Pre = comm.predict_proba(x_test)        
# blo_comm_Pre = blo(pro_comm_Pre[:,1],position)
# eva_comm = evaluating_indicator(y_true=y_true, y_test=blo_comm_Pre, y_test_value=pro_comm_Pre[:,1])
# print(eva_comm)
# pro_comm_Pre_for_cumci = pro_comm_Pre[:,1]
# pro_comm_Pre_knn_all = pro_comm_Pre
# # cumCI(y_true,pro_comm_Pre[:,1])


from sklearn.ensemble import AdaBoostClassifier

comm = AdaBoostClassifier()
comm.fit(x_train_for_vail , y_train_for_vail)   
# pro_comm_Pre = comm.predict_proba(x_train_for_vail.iloc[background,:])
# RightIndex=[]
# for jj in tqdm(range(100)): #计算模型在不同分类阈值下的各项指标
#     blo_comm_Pre = blo(pro_comm_Pre[:,1],jj)
#     eva_comm = evaluating_indicator(y_true=y_train_for_vail[background], y_test=blo_comm_Pre, y_test_value=pro_comm_Pre[:,1])
#     RightIndex.append(abs(eva_comm['TPR'] - eva_comm['TNR']))
# RightIndex=np.array(RightIndex,dtype=np.float16)
# position=np.argmin(RightIndex)  #选择出使得敏感性特异性最小的阈值作为分类阈值输出，作为测试集上使用的参数保存
pro_comm_Pre = comm.predict_proba(x_test)        
blo_comm_Pre = blo(pro_comm_Pre[:,1],49)#position
eva_comm = evaluating_indicator(y_true=y_true, y_test=blo_comm_Pre, y_test_value=pro_comm_Pre[:,1])
print(eva_comm)
pro_comm_Pre_for_cumci = pro_comm_Pre[:,1]
pro_comm_Pre_ada_all = pro_comm_Pre
# cumCI(y_true,pro_comm_Pre[:,1])

################################################## traditional risk methods ##################################################
csvname='comtest_for_model_'+str(learn_len)+'_'+str(gap)+'_'+str(fore_len)+'_all.feature'
comtest = feather.read_dataframe(csvname)  
comtest.drop(['patientunitstayid'],axis=1,inplace=True) #patientunitstayid,hospitalid
comtest.dropna(inplace=True)

comtest_tra = comtest.iloc[:,0:7+40*learn_len] .copy()
comtest_tra['pf'] = comtest.pao2_1/comtest.fio2_1
comtest_tra['oi'] = comtest.fio2_1*comtest.mean_airway_pressure_1/comtest.pao2_1
comtest_tra['osi'] = comtest.fio2_1*comtest.mean_airway_pressure_1/comtest.spo2_1
if aim_label==3:
    comtest_tra['osi']=comtest_tra['osi']*(-1)
comtest_tra['label'] = comtest.label
# comtest_tra=comtest_tra[comtest_tra.fio2_1>0]
# comtest_tra=comtest_tra[comtest_tra.oi>0]
# comtest_tra=comtest_tra[comtest_tra.pf>0]
# comtest_tra=comtest_tra[comtest_tra.osi>0]

# comtest_tra=comtest_tra.dropna(inplace=True)
x_train_for_vail, x_test, y_train_for_vail, y_true = model_selection.train_test_split(np.array(icustay_id),range(len(icustay_id)), test_size = 0.7,random_state = 2)    
x_train_for_vail=spec_for_ser('comtest_tra',x_train_for_vail);y_train_for_vail=x_train_for_vail.iloc[:,-1];x_train_for_vail_group=x_train_for_vail.iloc[:,0];x_train_for_vail=x_train_for_vail.iloc[:,1:x_train_for_vail.shape[1]-1]
x_test=spec_for_ser('comtest_tra',x_test);y_true=x_test.iloc[:,-1];x_test_group=x_test.iloc[:,0];x_test=x_test.iloc[:,1:x_test.shape[1]-1]
y_train_for_vail[y_train_for_vail==aim_label]=10;y_true[y_true==aim_label]=10
y_train_for_vail[y_train_for_vail<=3]=0;y_true[y_true<=3]=0
y_train_for_vail[y_train_for_vail>3]=1;y_true[y_true>3]=1

# background=np.random.choice(x_train_for_vail.shape[0], 10000, replace=False)

x_train_for_vail.pf=x_train_for_vail.pf*100;x_test.pf=x_test.pf*100
# RightIndex=[]
# for jj in tqdm(range(int(max(x_train_for_vail.pf[background])))): #计算模型在不同分类阈值下的各项指标
#     blo_comm_Pre = blo(np.array(x_train_for_vail.pf[background]),jj)
#     eva_comm = evaluating_indicator(y_true=y_train_for_vail[background], y_test=blo_comm_Pre, y_test_value=x_train_for_vail.pf[background])
#     RightIndex.append(abs(eva_comm['TPR'] - eva_comm['TNR']))
# RightIndex=np.array(RightIndex,dtype=np.float16)
# position=np.argmin(RightIndex)  #选择出使得敏感性特异性最小的阈值作为分类阈值输出，作为测试集上使用的参数保存
 
blo_comm_Pre = blo(np.array(x_test.pf),40)
eva_comm = evaluating_indicator(y_true=y_true, y_test=blo_comm_Pre, y_test_value=x_test.pf)
print(eva_comm)
pro_comm_Pre_for_cumci = np.array(x_test.pf)
cumCI(y_true,np.array(x_test.pf))

# x_test.oi[x_test.oi == x_test.oi.max()]=2
# # RightIndex=[]
# # for jj in tqdm(range(int(max(x_train_for_vail.oi)))): #计算模型在不同分类阈值下的各项指标
# # for jj in tqdm(range(500)): #计算模型在不同分类阈值下的各项指标
# #     blo_comm_Pre = blo(np.array(x_train_for_vail.oi[background]),jj)
# #     eva_comm = evaluating_indicator(y_true=y_train_for_vail[background], y_test=blo_comm_Pre, y_test_value=x_train_for_vail.oi[background])
# #     RightIndex.append(abs(eva_comm['TPR'] - eva_comm['TNR']))
# # RightIndex=np.array(RightIndex,dtype=np.float16)
# # position=np.argmin(RightIndex)  #选择出使得敏感性特异性最小的阈值作为分类阈值输出，作为测试集上使用的参数保存
 
# blo_comm_Pre = blo(np.array(x_test.oi),300)
# eva_comm = evaluating_indicator(y_true=y_true, y_test=blo_comm_Pre, y_test_value=x_test.oi)
# print(eva_comm)
# pro_comm_Pre_for_cumci = np.array(x_test.oi)
# cumCI(y_true,np.array(x_test.oi))



RightIndex=[]
for jj in tqdm(range(int(max(x_train_for_vail.osi)))): #计算模型在不同分类阈值下的各项指标
    blo_comm_Pre = blo(np.array(x_train_for_vail.osi),jj)
    eva_comm = evaluating_indicator(y_true=y_train_for_vail, y_test=blo_comm_Pre, y_test_value=x_train_for_vail.osi)
    RightIndex.append(abs(eva_comm['TPR'] - eva_comm['TNR']))
RightIndex=np.array(RightIndex,dtype=np.float16)
position=np.argmin(RightIndex)  #选择出使得敏感性特异性最小的阈值作为分类阈值输出，作为测试集上使用的参数保存
 
blo_comm_Pre = blo(np.array(x_test.osi),-7000)
eva_comm = evaluating_indicator(y_true=y_true, y_test=blo_comm_Pre, y_test_value=x_test.osi)
print(eva_comm)
pro_comm_Pre_for_cumci = np.array(x_test.osi)

# cumCI(y_true,np.array(x_test.osi))




################################################## plot roc ##################################################

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,roc_curve,auc

# pro_comm_Pre = comm.predict_proba(x_test)
fpr_gbm_all,tpr_gbm_all,threshold = roc_curve(np.array(y_true_roc_gbm), pro_comm_Pre_gbm_all[:,1]) 
roc_auc_gbm_all= auc(fpr_gbm_all,tpr_gbm_all)

fpr_rf_all,tpr_rf_all,threshold = roc_curve(y_true_roc, pro_comm_Pre_rf_all[:,1]) 
roc_auc_rf_all= auc(fpr_rf_all,tpr_rf_all)

# fpr_ann_all,tpr_ann_all,threshold = roc_curve(y_true_roc, pro_comm_Pre_ann_all[:,1]) 
# roc_auc_ann_all= auc(fpr_ann_all,tpr_ann_all)

fpr_log_all,tpr_log_all,threshold = roc_curve(y_true_roc, pro_comm_Pre_log_all[:,1]) 
roc_auc_log_all= auc(fpr_log_all,tpr_log_all)

# fpr_nb_all,tpr_nb_all,threshold = roc_curve(y_true_roc, pro_comm_Pre_nb_all[:,1]) 
# roc_auc_nb_all= auc(fpr_nb_all,tpr_nb_all)

fpr_ada_all,tpr_ada_all,threshold = roc_curve(y_true_roc, pro_comm_Pre_ada_all[:,1]) 
roc_auc_ada_all= auc(fpr_ada_all,tpr_ada_all)

# fpr_pf,tpr_pf,threshold = roc_curve(y_true, x_test.pf) 
# roc_pf = auc(fpr_pf,tpr_pf)

# fpr_oi,tpr_oi,threshold = roc_curve(y_true, x_test.oi) 
# roc_oi = auc(fpr_oi,tpr_oi)

fpr_osi,tpr_osi,threshold = roc_curve(y_true, x_test.osi) 
roc_osi = auc(fpr_osi,tpr_osi)

plt.subplots(figsize=(7,5.5))   

plt.plot(fpr_gbm_all, tpr_gbm_all, lw=2,
         label='Lightgbm ROC curve with complete feature set' % roc_auc_gbm_all)
plt.plot(fpr_rf_all, tpr_rf_all, lw=2,
         label='Random forest ROC curve with complete feature set' % roc_auc_rf_all)
# plt.plot(fpr_ann_all, tpr_ann_all, lw=2,
#          label='Artificial Neural Network ROC curve with all parameters' % roc_auc_ann_all)
plt.plot(fpr_log_all, tpr_log_all, lw=2,
         label='LogisticRegression ROC curve with complete feature set' % roc_auc_log_all)
# plt.plot(fpr_nb_all, tpr_nb_all, lw=2,
#          label='Naive Bayesian ROC curve with all parameters' % roc_auc_nb_all)
plt.plot(fpr_ada_all, tpr_ada_all, lw=2,
          label='AdaBoost ROC curve with all parameters' % roc_auc_ada_all)

# plt.plot(fpr_oi, tpr_oi, lw=2,
#          label='OI ROC curve ' % roc_oi)
plt.plot(fpr_osi, tpr_osi, lw=2,
         label='OSI ROC curve ' % roc_osi)



plt.plot([0, 1], [0, 1], color='navy', lw = 2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.tick_params(labelsize=14)
plt.xlabel('False Positive Rate',fontsize=20)
plt.ylabel('True Positive Rate',fontsize=20)
plt.title('ROC Curve',fontsize=20)
plt.legend(loc="lower right",fontsize=19)
plt.show()

