#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 11:04:14 2020

@author: amms
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 08:08:39 2020

@author: amms
"""


'''
bert-serving-start -model_dir /media/amms/80D89D09D89CFE9A/bert_model/uncased_L-2_H-128_A-2 -num_worker=2 -max_seq_len=60

'''

#import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import concurrent
import gc
pd.set_option('mode.chained_assignment', None)
import numpy as np
from numpy import concatenate
import time
comtest = pd.read_csv("verification_ards_degree.csv",low_memory=False)
# comtest = comtest.loc[comtest['pao2']!=np.nan]
# comtest = comtest.drop_duplicates() #去除重复
# comtest['bmi'].fillna(comtest['bmi'].median(), inplace=True)
# comtest = comtest.drop(['bilirubin','ntprobnp','fibrinogen','sedimentation_rate','ammonia','alkaline_phosphatas'],axis=1)

# comtest.isnull().sum(axis=0)/comtest.shape[0]
# comtest=comtest[comtest.isnull().sum(axis=1)/comtest.shape[1]<=0.7]  #为保留患者的机械通气信息应先对Vent状态插补，或根据患者数据缺失情况对患者全程信息删减

#pro_comtest=comtest
patientunitstayid=list(set(comtest.patientunitstayid))

comtest_index=list(comtest)
#################  label make  ##########################
def inter(i):
#    s=time.time()
    icu_id_param=comtest[comtest.patientunitstayid==i]
    c_offset=list(set(icu_id_param.c_offset))
    times=0
    for hour in c_offset:
    #对vent的插补
        if np.isnan(np.array(icu_id_param.vent_1.iloc[times])) and times==0:
            icu_id_param.vent_1.iloc[times]=0
            icu_id_param.vent_0.iloc[times]=1
        else:
            if np.isnan(np.array(icu_id_param.vent_1.iloc[times])):
                icu_id_param.vent_1.iloc[times]=icu_id_param.vent_1.iloc[times-1]
                icu_id_param.vent_0.iloc[times]=icu_id_param.vent_0.iloc[times-1]

    #对arterial_pao2的插补   #做预测时插补
#        if times>0:
#            if np.isnan(np.array(icu_id_param.pao2.iloc[times])):
#                if np.isnan(np.array(icu_id_param.spo2.iloc[times])):
##                    if np.isnan(np.array(icu_id_param.po2.iloc[times])):
#                        if np.isnan(np.array(icu_id_param.pao2.iloc[times-1])):
#                            icu_id_param.pao2.iloc[times]=np.nan
#                        else:
#                            icu_id_param.pao2.iloc[times]=icu_id_param.pao2.iloc[times-1]
##                    else:
##                        icu_id_param.arterial_pao2.iloc[times]=icu_id_param.po2.iloc[times]  
#                else:
#                    icu_id_param.pao2.iloc[times]=(icu_id_param.spo2.iloc[times])*300/315  

#    对arterial_pao2的插补   #做标签时插补
        if np.isnan(np.array(icu_id_param.pao2.iloc[times])):
            if ~np.isnan(np.array(icu_id_param.spo2.iloc[times])):
                icu_id_param.pao2.iloc[times]=(icu_id_param.spo2.iloc[times])*300/315 

    #对fio2的插补
        if np.isnan(np.array(icu_id_param.fio2.iloc[times])) and times==0:
            icu_id_param.fio2.iloc[times]=21
        else:
            if np.isnan(np.array(icu_id_param.fio2.iloc[times])):
                icu_id_param.fio2.iloc[times]=icu_id_param.fio2.iloc[times-1]    
#    #对peep的插补
#        if np.isnan(np.array(icu_id_param.peep.iloc[times])) and times==0:
#            icu_id_param.peep.iloc[times]=0
#        else:
#            if np.isnan(np.array(icu_id_param.peep.iloc[times])):
#                icu_id_param.peep.iloc[times]=icu_id_param.peep.iloc[times-1]
#    #对urineoutput,outputtotal的插补
#        if np.isnan(np.array(icu_id_param.urineoutput.iloc[times])):
#            icu_id_param.urineoutput.iloc[times]=0
#        if np.isnan(np.array(icu_id_param.outputtotal.iloc[times])):
#            icu_id_param.outputtotal.iloc[times]=0
#    #other_label
#        for param in other_label:
#            if np.isnan(np.array(icu_id_param[icu_id_param.chart_time==hour][param])) and times==0:
#                icu_id_param[param].iloc[times]=comtest[param].median()
#            else:
#                if np.isnan(np.array(icu_id_param[icu_id_param.chart_time==hour][param])):
#                    icu_id_param[param].iloc[times]=icu_id_param[param].iloc[times-1]
        times+=1
#    print(time.time()-s)
    return icu_id_param



#from multiprocessing.dummy import Pool as ThreadPool
#pool = ThreadPool()
#pro = []
#if __name__=='__main__':
#    pro=list(pool.imap(inter,patientunitstayid))
#    pool.close()
#    pool.join()



with concurrent.futures.ProcessPoolExecutor() as executor:
    pro=list(tqdm(executor.map(inter,patientunitstayid) ,total=len(patientunitstayid),desc='progress:'))

#    executor.map(inter,patientunitstayid)
for i in range(0,len(pro)):
    if i==0:
        input_mulit=('pro[{}]'.format(i))
    else:
        input_mulit=(input_mulit+',pro[{}]'.format(i))

p=pd.concat(eval(input_mulit),axis=0,ignore_index=True);del pro

def label_make(num_pf,vent_statu):
    label=[]
    for i in range(num_pf.shape[0]):
        if vent_statu[i]==1 and num_pf[i]<=300 and num_pf[i]>200:
            label.append(1);
        elif vent_statu[i]==1 and num_pf[i]<=200 and num_pf[i]>100:
            label.append(2)
        elif vent_statu[i]==1 and num_pf[i]<=100:
            label.append(3)
        elif vent_statu[i]==0 and num_pf[i]>300:
            label.append(0)
        else:
            label.append(np.nan)
    return np.array(label).reshape(-1,1)       
p['label']=label_make(p['pao2']*100/p['fio2'],p['vent_1'])
#p.to_csv('D:/comtest_INTERED_lack_label_add_ni_bp.csv',index=False)
comtest=p;del p

##############################################################

#comtest=comtest[comtest.patientunitstayid in np.array(set(comtest.iloc[:,[0,-1]].dropna().patientunitstayid))]
##############################################################


#################  数据插补  ##########################
other_label=comtest.columns.values.tolist()
for h in ['patientunitstayid','c_offset','age','gender_0','gender_1','bmi','vent_1','vent_0','urineoutput','outputtotal','fio2','pao2','peep','label','allergynotetype','specialtytype','usertype','rxincluded','writtenineicu','drugname_allergy','allergytype','allergyname','pasthistorynotetype','pasthistorypath','pasthistoryvalue','specialty','managingphysician','cplgroup','cplitemvalue','cplgoalcategory','cplgoalvalue','infectdiseasesite','infectdiseaseassessment','responsetotherapy','treatment','drugname_infusiondrug','infusionrate','drugamount','volumeoffluid','cellattributepath','celllabel','cellattribute','cellattributevalue']:
    other_label.remove(h)  #删除不需要插补,或有特殊插补方法的参数


def inter(i):
    
    icu_id_param=comtest[comtest.patientunitstayid==i]
    if np.isnan(max(icu_id_param.label)):
        return
    # s=time.time()
    c_offset=list(set(icu_id_param.c_offset))
    times=0
    for hour in c_offset:
#    #对vent的插补
#        if np.isnan(np.array(icu_id_param.vent_1.iloc[times])) and times==0:
#            icu_id_param.vent_1.iloc[times]=0
#            icu_id_param.vent_0.iloc[times]=1
#        else:
#            if np.isnan(np.array(icu_id_param.vent_1.iloc[times])):
#                icu_id_param.vent_1.iloc[times]=icu_id_param.vent_1.iloc[times-1]
#                icu_id_param.vent_0.iloc[times]=icu_id_param.vent_0.iloc[times-1]

    #对arterial_pao2的插补   #做预测时插补
        if times>0:
            if np.isnan(np.array(icu_id_param.pao2.iloc[times])):
                if np.isnan(np.array(icu_id_param.spo2.iloc[times])):
                    if np.isnan(np.array(icu_id_param.pao2.iloc[times-1])):
                        icu_id_param.pao2.iloc[times]=np.nan
                    else:
                        icu_id_param.pao2.iloc[times]=icu_id_param.pao2.iloc[times-1] 
                else:
                    icu_id_param.pao2.iloc[times]=(icu_id_param.spo2.iloc[times])*300/315  
        elif times==0:
            if np.isnan(np.array(icu_id_param.pao2.iloc[times])):
                icu_id_param['pao2'].iloc[times]=comtest['pao2'].median()
                
    #对peep的插补
        if np.isnan(np.array(icu_id_param.peep.iloc[times])) and times==0:
            icu_id_param.peep.iloc[times]=0
        else:
            if np.isnan(np.array(icu_id_param.peep.iloc[times])):
                icu_id_param.peep.iloc[times]=icu_id_param.peep.iloc[times-1]
    #对urineoutput,outputtotal的插补
        if np.isnan(np.array(icu_id_param.urineoutput.iloc[times])):
            icu_id_param.urineoutput.iloc[times]=0
        if np.isnan(np.array(icu_id_param.outputtotal.iloc[times])):
            icu_id_param.outputtotal.iloc[times]=0
    #对与时间变化无关的文本信息插补
        for param in ['allergynotetype','specialtytype','usertype','rxincluded','writtenineicu','drugname_allergy','allergytype','allergyname','pasthistorynotetype','pasthistorypath','pasthistoryvalue','specialty','managingphysician','cplgroup','cplitemvalue','cplgoalcategory','cplgoalvalue','infectdiseasesite','infectdiseaseassessment','responsetotherapy','treatment','drugname_infusiondrug','infusionrate','drugamount','volumeoffluid','cellattributepath','celllabel','cellattribute','cellattributevalue']:
            if pd.isna(icu_id_param[icu_id_param.c_offset==hour][param]).iloc[0] and times>0:
                icu_id_param[param].iloc[times]=icu_id_param[param].iloc[times-1]
            elif ~pd.isna(icu_id_param[param].iloc[times]) and ~pd.isna(icu_id_param[param].iloc[times-1]) and times>0:
                icu_id_param[param].iloc[times]=str(icu_id_param[param].iloc[times-1])+','+str(icu_id_param[param].iloc[times])
    #对与时间变化有关的文本信息插补
#        for param in ['cplgoalcategory','cplgoalvalue','infectdiseasesite','infectdiseaseassessment','responsetotherapy','treatment','drugname_infusiondrug','infusionrate','drugamount','volumeoffluid','cellattributepath','celllabel','cellattribute','cellattributevalue']:
#            if ~pd.isna(icu_id_param[param].iloc[times]):
#                icu_id_param[param].iloc[times]='At'+str(hour)+' hours'+','+str(icu_id_param[param].iloc[times])
#            if pd.isna(icu_id_param[icu_id_param.c_offset==hour][param]).iloc[0] and times>0:
#                icu_id_param[param].iloc[times]=icu_id_param[param].iloc[times-1]
#            elif ~pd.isna(icu_id_param[param].iloc[times]) and ~pd.isna(icu_id_param[param].iloc[times-1]) and times>0:
#                icu_id_param[param].iloc[times]=str(icu_id_param[param].iloc[times-1])+','+str(icu_id_param[param].iloc[times])
          
        
#    #other_label
#        for param in other_label:
#            if np.isnan(np.array(icu_id_param[icu_id_param.c_offset==hour][param])) and times==0 and str(icu_id_param[param].dropna().dtype) == 'float64':
#                icu_id_param[param].iloc[times]=comtest[param].median()
#            else:
#                if np.isnan(np.array(icu_id_param[icu_id_param.c_offset==hour][param])):
#                    icu_id_param[param].iloc[times]=icu_id_param[param].iloc[times-1]
        for param in other_label:
            if pd.isna(icu_id_param[icu_id_param.c_offset==hour][param]).iloc[0]  and times==0 and str(icu_id_param[param].dropna().dtype) == 'float64':
                icu_id_param[param].iloc[times]=comtest[param].median()
            else:
                if pd.isna(icu_id_param[icu_id_param.c_offset==hour][param]).iloc[0] and times>0:
                    icu_id_param[param].iloc[times]=icu_id_param[param].iloc[times-1]
        times+=1
    # print(time.time()-s)
    return icu_id_param

patientunitstayid=list(set(comtest.patientunitstayid))
#for i in range(len(patientunitstayid)):
#    comtest[comtest.patientunitstayid==patientunitstayid[i]]=inter(patientunitstayid[i])
#    if i%100==0:
#        print(i/len(patientunitstayid))
#pool = ThreadPool()
#pro = []
#if __name__=='__main__':
#    pro=list(pool.imap(inter,patientunitstayid))
#    pool.close()
#    pool.join()


with concurrent.futures.ProcessPoolExecutor() as executor:
    # pro=list(executor.map(inter,patientunitstayid))
    pro=list(tqdm(executor.map(inter,patientunitstayid) ,total=len(patientunitstayid),desc='progress:'))

for i in range(0,len(pro)):
    if i==0:
        input_mulit=('pro[{}]'.format(i))
    else:
        input_mulit=(input_mulit+',pro[{}]'.format(i))

p=pd.concat(eval(input_mulit),axis=0,ignore_index=True)
comtest=p;del p;del pro

# comtest.to_csv('/media/amms/80D89D09D89CFE9A/bert_model/test.csv',index=False)


