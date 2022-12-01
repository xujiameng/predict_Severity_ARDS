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

#from tqdm import tqdm
##import matplotlib.pyplot as plt
#import pandas as pd
#import concurrent
#import gc
#pd.set_option('mode.chained_assignment', None)
#import numpy as np
#from numpy import concatenate
#import time
#comtest = pd.read_csv("/media/amms/80D89D09D89CFE9A/dym_ards_warning_structured_and_text_data_limit_offset_12h.csv")
# # comtest = comtest.loc[comtest['pao2']!=np.nan]
# # comtest = comtest.drop_duplicates() #去除重复
# # comtest['bmi'].fillna(comtest['bmi'].median(), inplace=True)
# # comtest = comtest.drop(['bilirubin','ntprobnp','fibrinogen','sedimentation_rate','ammonia','alkaline_phosphatas'],axis=1)

# # comtest.isnull().sum(axis=0)/comtest.shape[0]
# # comtest=comtest[comtest.isnull().sum(axis=1)/comtest.shape[1]<=0.7]  #为保留患者的机械通气信息应先对Vent状态插补，或根据患者数据缺失情况对患者全程信息删减

# #pro_comtest=comtest
# patientunitstayid=list(set(comtest.patientunitstayid))

# comtest_index=list(comtest)
# #################  label make  ##########################
# def inter(i):
# #    s=time.time()
#     icu_id_param=comtest[comtest.patientunitstayid==i]
#     c_offset=list(set(icu_id_param.c_offset))
#     times=0
#     for hour in c_offset:
#     #对vent的插补
#         if np.isnan(np.array(icu_id_param.vent_1.iloc[times])) and times==0:
#             icu_id_param.vent_1.iloc[times]=0
#             icu_id_param.vent_0.iloc[times]=1
#         else:
#             if np.isnan(np.array(icu_id_param.vent_1.iloc[times])):
#                 icu_id_param.vent_1.iloc[times]=icu_id_param.vent_1.iloc[times-1]
#                 icu_id_param.vent_0.iloc[times]=icu_id_param.vent_0.iloc[times-1]

#     #对arterial_pao2的插补   #做预测时插补
# #        if times>0:
# #            if np.isnan(np.array(icu_id_param.pao2.iloc[times])):
# #                if np.isnan(np.array(icu_id_param.spo2.iloc[times])):
# ##                    if np.isnan(np.array(icu_id_param.po2.iloc[times])):
# #                        if np.isnan(np.array(icu_id_param.pao2.iloc[times-1])):
# #                            icu_id_param.pao2.iloc[times]=np.nan
# #                        else:
# #                            icu_id_param.pao2.iloc[times]=icu_id_param.pao2.iloc[times-1]
# ##                    else:
# ##                        icu_id_param.arterial_pao2.iloc[times]=icu_id_param.po2.iloc[times]  
# #                else:
# #                    icu_id_param.pao2.iloc[times]=(icu_id_param.spo2.iloc[times])*300/315  

# #    对arterial_pao2的插补   #做标签时插补
#         if np.isnan(np.array(icu_id_param.pao2.iloc[times])):
#             if ~np.isnan(np.array(icu_id_param.spo2.iloc[times])):
#                 icu_id_param.pao2.iloc[times]=(icu_id_param.spo2.iloc[times])*300/315 

#     #对fio2的插补
#         if np.isnan(np.array(icu_id_param.fio2.iloc[times])) and times==0:
#             icu_id_param.fio2.iloc[times]=21
#         else:
#             if np.isnan(np.array(icu_id_param.fio2.iloc[times])):
#                 icu_id_param.fio2.iloc[times]=icu_id_param.fio2.iloc[times-1]    
# #    #对peep的插补
# #        if np.isnan(np.array(icu_id_param.peep.iloc[times])) and times==0:
# #            icu_id_param.peep.iloc[times]=0
# #        else:
# #            if np.isnan(np.array(icu_id_param.peep.iloc[times])):
# #                icu_id_param.peep.iloc[times]=icu_id_param.peep.iloc[times-1]
# #    #对urineoutput,outputtotal的插补
# #        if np.isnan(np.array(icu_id_param.urineoutput.iloc[times])):
# #            icu_id_param.urineoutput.iloc[times]=0
# #        if np.isnan(np.array(icu_id_param.outputtotal.iloc[times])):
# #            icu_id_param.outputtotal.iloc[times]=0
# #    #other_label
# #        for param in other_label:
# #            if np.isnan(np.array(icu_id_param[icu_id_param.chart_time==hour][param])) and times==0:
# #                icu_id_param[param].iloc[times]=comtest[param].median()
# #            else:
# #                if np.isnan(np.array(icu_id_param[icu_id_param.chart_time==hour][param])):
# #                    icu_id_param[param].iloc[times]=icu_id_param[param].iloc[times-1]
#         times+=1
# #    print(time.time()-s)
#     return icu_id_param



# #from multiprocessing.dummy import Pool as ThreadPool
# #pool = ThreadPool()
# #pro = []
# #if __name__=='__main__':
# #    pro=list(pool.imap(inter,patientunitstayid))
# #    pool.close()
# #    pool.join()



# with concurrent.futures.ProcessPoolExecutor() as executor:
#     pro=list(tqdm(executor.map(inter,patientunitstayid) ,total=len(patientunitstayid),desc='progress:'))

# #    executor.map(inter,patientunitstayid)
# for i in range(0,len(pro)):
#     if i==0:
#         input_mulit=('pro[{}]'.format(i))
#     else:
#         input_mulit=(input_mulit+',pro[{}]'.format(i))

# p=pd.concat(eval(input_mulit),axis=0,ignore_index=True);del pro

# def label_make(num_pf,vent_statu):
#     label=[]
#     for i in range(num_pf.shape[0]):
#         if vent_statu[i]==1 and num_pf[i]<=300 and num_pf[i]>200:
#             label.append(1);
#         elif vent_statu[i]==1 and num_pf[i]<=200 and num_pf[i]>100:
#             label.append(2)
#         elif vent_statu[i]==1 and num_pf[i]<=100:
#             label.append(3)
#         elif vent_statu[i]==0 and num_pf[i]>300:
#             label.append(0)
#         else:
#             label.append(np.nan)
#     return np.array(label).reshape(-1,1)       
# p['label']=label_make(p['pao2']*100/p['fio2'],p['vent_1'])
# #p.to_csv('D:/comtest_INTERED_lack_label_add_ni_bp.csv',index=False)
# comtest=p;del p

# ##############################################################

# #comtest=comtest[comtest.patientunitstayid in np.array(set(comtest.iloc[:,[0,-1]].dropna().patientunitstayid))]
# ##############################################################


# #################  数据插补  ##########################
# other_label=comtest.columns.values.tolist()
# for h in ['patientunitstayid','c_offset','age','gender_0','gender_1','bmi','vent_1','vent_0','urineoutput','outputtotal','fio2','pao2','peep','label','death_label','to_discharge','vent_duration','allergynotetype','specialtytype','usertype','rxincluded','writtenineicu','drugname_allergy','allergytype','allergyname','pasthistorynotetype','pasthistorypath','pasthistoryvalue','specialty','managingphysician','cplgroup','cplitemvalue','cplgoalcategory','cplgoalvalue','infectdiseasesite','infectdiseaseassessment','responsetotherapy','treatment','drugname_infusiondrug','infusionrate','drugamount','volumeoffluid','cellattributepath','celllabel','cellattribute','cellattributevalue']:
#     other_label.remove(h)  #删除不需要插补,或有特殊插补方法的参数


# def inter(i):
    
#     icu_id_param=comtest[comtest.patientunitstayid==i]
#     if np.isnan(max(icu_id_param.label)):
#         return
#     # s=time.time()
#     c_offset=list(set(icu_id_param.c_offset))
#     times=0
#     for hour in c_offset:
# #    #对vent的插补
# #        if np.isnan(np.array(icu_id_param.vent_1.iloc[times])) and times==0:
# #            icu_id_param.vent_1.iloc[times]=0
# #            icu_id_param.vent_0.iloc[times]=1
# #        else:
# #            if np.isnan(np.array(icu_id_param.vent_1.iloc[times])):
# #                icu_id_param.vent_1.iloc[times]=icu_id_param.vent_1.iloc[times-1]
# #                icu_id_param.vent_0.iloc[times]=icu_id_param.vent_0.iloc[times-1]

#     #对arterial_pao2的插补   #做预测时插补
#         if times>0:
#             if np.isnan(np.array(icu_id_param.pao2.iloc[times])):
#                 if np.isnan(np.array(icu_id_param.spo2.iloc[times])):
#                     if np.isnan(np.array(icu_id_param.pao2.iloc[times-1])):
#                         icu_id_param.pao2.iloc[times]=np.nan
#                     else:
#                         icu_id_param.pao2.iloc[times]=icu_id_param.pao2.iloc[times-1] 
#                 else:
#                     icu_id_param.pao2.iloc[times]=(icu_id_param.spo2.iloc[times])*300/315  
#         elif times==0:
#             if np.isnan(np.array(icu_id_param.pao2.iloc[times])):
#                 icu_id_param['pao2'].iloc[times]=comtest['pao2'].median()
                
#     #对peep的插补
#         if np.isnan(np.array(icu_id_param.peep.iloc[times])) and times==0:
#             icu_id_param.peep.iloc[times]=0
#         else:
#             if np.isnan(np.array(icu_id_param.peep.iloc[times])):
#                 icu_id_param.peep.iloc[times]=icu_id_param.peep.iloc[times-1]
#     #对urineoutput,outputtotal的插补
#         if np.isnan(np.array(icu_id_param.urineoutput.iloc[times])):
#             icu_id_param.urineoutput.iloc[times]=0
#         if np.isnan(np.array(icu_id_param.outputtotal.iloc[times])):
#             icu_id_param.outputtotal.iloc[times]=0
#     #对与时间变化无关的文本信息插补
#         for param in ['allergynotetype','specialtytype','usertype','rxincluded','writtenineicu','drugname_allergy','allergytype','allergyname','pasthistorynotetype','pasthistorypath','pasthistoryvalue','specialty','managingphysician','cplgroup','cplitemvalue','cplgoalcategory','cplgoalvalue','infectdiseasesite','infectdiseaseassessment','responsetotherapy','treatment','drugname_infusiondrug','infusionrate','drugamount','volumeoffluid','cellattributepath','celllabel','cellattribute','cellattributevalue']:
#             if pd.isna(icu_id_param[icu_id_param.c_offset==hour][param]).iloc[0] and times>0:
#                 icu_id_param[param].iloc[times]=icu_id_param[param].iloc[times-1]
#             elif ~pd.isna(icu_id_param[param].iloc[times]) and ~pd.isna(icu_id_param[param].iloc[times-1]) and times>0:
#                 icu_id_param[param].iloc[times]=str(icu_id_param[param].iloc[times-1])+','+str(icu_id_param[param].iloc[times])
#     #对与时间变化有关的文本信息插补
# #        for param in ['cplgoalcategory','cplgoalvalue','infectdiseasesite','infectdiseaseassessment','responsetotherapy','treatment','drugname_infusiondrug','infusionrate','drugamount','volumeoffluid','cellattributepath','celllabel','cellattribute','cellattributevalue']:
# #            if ~pd.isna(icu_id_param[param].iloc[times]):
# #                icu_id_param[param].iloc[times]='At'+str(hour)+' hours'+','+str(icu_id_param[param].iloc[times])
# #            if pd.isna(icu_id_param[icu_id_param.c_offset==hour][param]).iloc[0] and times>0:
# #                icu_id_param[param].iloc[times]=icu_id_param[param].iloc[times-1]
# #            elif ~pd.isna(icu_id_param[param].iloc[times]) and ~pd.isna(icu_id_param[param].iloc[times-1]) and times>0:
# #                icu_id_param[param].iloc[times]=str(icu_id_param[param].iloc[times-1])+','+str(icu_id_param[param].iloc[times])
          
        
# #    #other_label
# #        for param in other_label:
# #            if np.isnan(np.array(icu_id_param[icu_id_param.c_offset==hour][param])) and times==0 and str(icu_id_param[param].dropna().dtype) == 'float64':
# #                icu_id_param[param].iloc[times]=comtest[param].median()
# #            else:
# #                if np.isnan(np.array(icu_id_param[icu_id_param.c_offset==hour][param])):
# #                    icu_id_param[param].iloc[times]=icu_id_param[param].iloc[times-1]
#         for param in other_label:
#             if pd.isna(icu_id_param[icu_id_param.c_offset==hour][param]).iloc[0]  and times==0 and str(icu_id_param[param].dropna().dtype) == 'float64':
#                 icu_id_param[param].iloc[times]=comtest[param].median()
#             else:
#                 if pd.isna(icu_id_param[icu_id_param.c_offset==hour][param]).iloc[0] and times>0:
#                     icu_id_param[param].iloc[times]=icu_id_param[param].iloc[times-1]
#         times+=1
#     # print(time.time()-s)
#     return icu_id_param

# patientunitstayid=list(set(comtest.patientunitstayid))
# #for i in range(len(patientunitstayid)):
# #    comtest[comtest.patientunitstayid==patientunitstayid[i]]=inter(patientunitstayid[i])
# #    if i%100==0:
# #        print(i/len(patientunitstayid))
# #pool = ThreadPool()
# #pro = []
# #if __name__=='__main__':
# #    pro=list(pool.imap(inter,patientunitstayid))
# #    pool.close()
# #    pool.join()


# with concurrent.futures.ProcessPoolExecutor(4) as executor:
#     # pro=list(executor.map(inter,patientunitstayid))
#     pro=list(tqdm(executor.map(inter,patientunitstayid) ,total=len(patientunitstayid),desc='progress:'))

# for i in range(0,len(pro)):
#     if i==0:
#         input_mulit=('pro[{}]'.format(i))
#     else:
#         input_mulit=(input_mulit+',pro[{}]'.format(i))

# p=pd.concat(eval(input_mulit),axis=0,ignore_index=True)
# comtest=p;del p;del pro
#
#comtest.to_csv('/media/amms/80D89D09D89CFE9A/bert_model/test.csv',index=False)
#comtest=pd.read_csv('/media/amms/80D89D09D89CFE9A/bert_model/test.csv')
##import pandas as pd
####pd.set_option('mode.chained_assignment', None)
##import numpy as np
##import time
##comtest = pd.read_csv("/media/amms/80D89D09D89CFE9A/bert_model/test.csv")
#comtest = comtest[comtest['c_offset']>=0]
##comtest = comtest.drop(['ni_sysbp','ni_meanbp','ni_diasbp'],axis=1)
##comtest(['alkaline_phosphatas','fio2_chartevents'],inplace=True)
#
##comtest=p
#

#######################################################################################################################
#######################################################################################################################
#######################################################################################################################

'''

from tqdm import tqdm
import pandas as pd
import concurrent
import gc
pd.set_option('mode.chained_assignment', None)
import numpy as np
from numpy import concatenate
import time
# comtest = pd.read_csv("verification_ards_degree_intered.csv")


def inter_learn_len(chart_time_opt,icu_id_param):
    icu_id_param=icu_id_param[icu_id_param.c_offset<=chart_time_opt[-1]]
    if icu_id_param[icu_id_param.c_offset==0].shape[0]==0:
        icu_id_param=pd.concat([icu_id_param[icu_id_param.c_offset==list(set(icu_id_param.c_offset))[0]],icu_id_param],ignore_index=True)
        icu_id_param.c_offset.iloc[0]=0;icu_id_param.label.iloc[0]=np.nan

    for i in range(0,int(chart_time_opt[-1])):
        if icu_id_param[icu_id_param.c_offset==i].shape[0]==0:
            icu_id_param=pd.concat([icu_id_param[icu_id_param.c_offset<i],icu_id_param[icu_id_param.c_offset<i].iloc[-1].to_frame().T,icu_id_param[icu_id_param.c_offset>i]],ignore_index=True)
            icu_id_param.c_offset.iloc[i]=i;icu_id_param.label.iloc[i]=np.nan
    return icu_id_param
#

def final_data(i):
#    temporary=comtest.iloc[[0,1],:]
    # s=time.time()
    icu_id_param=comtest[comtest.patientunitstayid==i]
    # icu_id_param.sort_values(by="c_offset", inplace=True, ascending=True)  ## 患者数据随时间排序
    if len(set(icu_id_param.c_offset))!=len(icu_id_param):
        for i_pro in list(set(icu_id_param.c_offset)):
            if i_pro==list(set(icu_id_param.c_offset))[0]:
                z=pd.DataFrame(icu_id_param[icu_id_param.c_offset==i_pro].iloc[0,:]).T
            else:
                z=pd.concat([z,pd.DataFrame(icu_id_param[icu_id_param.c_offset==i_pro].iloc[0,:]).T])
        icu_id_param=z

    chart_time_opt=np.array(icu_id_param.c_offset[icu_id_param.label.notnull()])
    # chart_time_opt=np.array(icu_id_param.c_offset[icu_id_param.death_label.notnull()])
    # chart_time_opt=np.append(chart_time_opt,np.array(icu_id_param.c_offset[icu_id_param.to_discharge.notnull()]))
    # chart_time_opt=np.array(icu_id_param.c_offset[icu_id_param.vent_duration.notnull()])
    chart_time_opt=np.append(chart_time_opt,np.array(icu_id_param.c_offset[icu_id_param.label.notnull()]))    
    chart_time_opt=np.array(list(set(chart_time_opt)));chart_time_opt=np.array(sorted(chart_time_opt))
    chart_time_opt=chart_time_opt[chart_time_opt>=learn_len+gap] #
    
    # if len(chart_time_opt)!=len(set(chart_time_opt)):
    #     return
    temp_return=np.ones([1,len(new_name)]); #len=2时， 160
    if len(chart_time_opt)!=0:
        icu_id_param=inter_learn_len(chart_time_opt,icu_id_param)
        for jj in chart_time_opt:
            for fore_len_l in range(0,fore_len):
                if jj-gap-learn_len-fore_len_l>=0:
                    index_1=np.ones([icu_id_param.shape[0],1])*0;index_1[icu_id_param.c_offset<jj-gap-fore_len_l]=1
                    index_2=np.ones([icu_id_param.shape[0],1])*0;index_2[icu_id_param.c_offset>=jj-gap-learn_len-fore_len_l]=1
                    temporary=icu_id_param[index_1+index_2>=2];#temporary=temporary.iloc[:,0:icu_id_param.shape[1]-2]
                    # temp=np.append(np.array(temporary[['patientunitstayid','age','gender_0','gender_1','bmi']].iloc[0].to_frame().T),np.array(temporary.drop(columns=['patientunitstayid','age','gender_0','gender_1','bmi','c_offset','death_label','to_discharge','vent_duration','label'])).T.reshape([1,-1]))  # 缺少'out_time'
                    temp=np.append(np.array(temporary[['hospitalid','patientunitstayid','age','gender_0','gender_1','bmi','c_offset']].iloc[0].to_frame().T),np.array(temporary.drop(columns=['hospitalid','patientunitstayid','age','gender_0','gender_1','bmi','c_offset','label'])).T.reshape([1,-1]))  # 缺少'out_time'
                    
                    
                    # temp=np.append(temp,np.array(icu_id_param[icu_id_param.c_offset==jj].death_label));temp=np.append(temp,np.array(icu_id_param[icu_id_param.c_offset==jj].to_discharge))
                    # temp=np.append(temp,np.array(icu_id_param[icu_id_param.c_offset==jj].vent_duration));
                    temp=np.append(temp,np.array(icu_id_param[icu_id_param.c_offset==jj].label)).reshape(1,-1)
                    temp_return=np.append(temp_return,temp,axis=0)
    temp_return=np.delete(temp_return,0,axis=0)
    temp_return=pd.DataFrame(temp_return)
    # print(time.time()-s)
    return temp_return



##############################################################################3
import pandas as pd
pd.set_option('mode.chained_assignment', None)
import itertools
import os
import feather
#from functools import partial
from bert_serving.client import BertClient
#bc = BertClient(ip='localhost')
bc = BertClient()

for gap in [24,12,8,4,2]:
    comtest=pd.read_csv('verification_ards_degree_intered.csv',nrows=80000)
    comtest_index=list(comtest);
    learn_len=2;gap=gap;fore_len=4
    patientunitstayid=list(set(comtest.patientunitstayid))
    
    
    
    # old_name=comtest_index
    # new_name=['patientunitstayid','age','gender_0','gender_1','bmi']
    # for j in range(6,82+1):
    #     for i in range(1,learn_len+1):
    #         new_name=new_name+[old_name[j]+'_'+str(i)]
    # new_name=new_name+['death_label']+['to_discharge']+['vent_duration']+['label']     #缺少'out_time'
    
    old_name=comtest_index
    new_name=['hospitalid','patientunitstayid','age','gender_0','gender_1','bmi','c_offset']
    for j in range(len(new_name)+1,83+1):
        for i in range(1,learn_len+1):
            new_name=new_name+[old_name[j]+'_'+str(i)]
    # new_name=new_name+['death_label']+['to_discharge']+['vent_duration']+['label']     #缺少'out_time'
    new_name=new_name+['label']     #缺少'out_time'


    with concurrent.futures.ProcessPoolExecutor() as executor:
        # pro=list(executor.map(final_data,patientunitstayid))
        pro=list(tqdm(executor.map(final_data,patientunitstayid) ,total=len(patientunitstayid),desc='progress:'))

    
    for i in range(0,len(pro)):
        if i==0:
            input_mulit=('pro[{}]'.format(i))
        else:
            input_mulit=(input_mulit+',pro[{}]'.format(i))
    
    p=pd.concat(eval(input_mulit),axis=0,ignore_index=True);
    del pro,comtest;gc.collect()
    comtest=p;del p;gc.collect()
    #
    
    #p.rename=[new_name]   
    comtest.columns=new_name   
#    csvname='comtest_for_model_'+str(learn_len)+'_'+str(gap)+'_'+str(fore_len)+'_all'+'.csv'
#    comtest.to_csv('/media/amms/80D89D09D89CFE9A/bert_model/'+csvname,index=False)
#    del comtest
    
    # cc=comtest.iloc[:,0:6+40*learn_len]
    
#    cc=pd.DataFrame(np.hstack((np.array(comtest.patientunitstayid).reshape(-1,1),cc,np.array(comtest.label).reshape(-1,1))) ) 
    # cc['label']=comtest.label;
    # cc['death_label']=comtest.death_label
    # cc['to_discharge']=comtest.to_discharge; cc['vent_duration']=comtest.vent_duration
        
    
    # csvname='comtest_for_model_'+str(learn_len)+'_'+str(gap)+'_'+str(fore_len)+'_all'+'jiegouhua'+'.csv'
    # cc.to_csv(csvname,index=False)
    
    pro = comtest.iloc[:,7+40*learn_len:(comtest).shape[1]-1] #中间两个是hospitalid & wardid
    le=pro.shape[1]
    # del comtest;gc.collect()
    
    
    def merge_data(comtest_copy_pro): 
        comtest_copy_pro=comtest_copy_pro.astype(str)   
        comtest_copy=pd.DataFrame([])
        for i in range(int(comtest_copy_pro.shape[1]/learn_len)):
            for j in range(learn_len):
                if j==0:
                    comtest_copy[old_name[i]]=comtest_copy_pro.iloc[:,i*learn_len].str.slice(0,max_seq_len)
                else:
                    comtest_copy[old_name[i]]= comtest_copy[old_name[i]]+'. the data in next hour as flollows: '+comtest_copy_pro.iloc[:,i*learn_len+j].str.slice(0,max_seq_len)
        return  comtest_copy
#    del comtest_copy_pro;gc.collect()
#    z=np.array(comtest.patientunitstayid).reshape(-1,1)
#    zz=np.array(comtest.label).reshape(-1,1)
    
    #z=comtest.iloc[:,0:1];z['text_data']=np.nan
    c=[];len_bert=4000;max_seq_len=1000
    for j in tqdm(range(0,int(len(pro)/len_bert))):
        print('deal with num '+str(j) + ' task. '+time.strftime('%Y-%m-%d %H:%M:%S'))
#        print()
        c_pro=[];
        if j==int(len(pro)/len_bert)-1:
            c_pro=pro.iloc[j*len_bert:len(pro),0:le]
        else: 
            c_pro=pro.iloc[j*len_bert:(j+1)*len_bert,0:le]
#                    c_pro.extend(pro[i][0:le])
        if j==0 or j == int((len(pro)/len_bert)/3)+1 or j == int((len(pro)/len_bert)/3)*2+1:
            c=bc.encode(list(itertools.chain.from_iterable(merge_data(c_pro).values.tolist()))).reshape(-1,int(le*128/learn_len))
        else:
            c=concatenate((c,bc.encode(list(itertools.chain.from_iterable(merge_data(c_pro).values.tolist()))).reshape(-1,int(le*128/learn_len))),axis=0)
        del c_pro;gc.collect()
        
        if j == int((len(pro)/len_bert)/3):
            # csvname='comtest_for_model_'+str(learn_len)+'_'+str(gap)+'_'+str(fore_len)+'_all'+'textxianglianghua_0'+'.csv'
            # pd.DataFrame(c).to_csv(csvname,index=False)
            csvname='comtest_for_model_'+str(learn_len)+'_'+str(gap)+'_'+str(fore_len)+'_all'+'textxianglianghua_0'+'.feature'
            feather.write_dataframe(pd.DataFrame(c), csvname)
            del c;gc.collect()
        if j == int((len(pro)/len_bert)/3)*2:
            # csvname='comtest_for_model_'+str(learn_len)+'_'+str(gap)+'_'+str(fore_len)+'_all'+'textxianglianghua_1'+'.csv'
            # pd.DataFrame(c).to_csv(csvname,index=False)
            csvname='comtest_for_model_'+str(learn_len)+'_'+str(gap)+'_'+str(fore_len)+'_all'+'textxianglianghua_1'+'.feature'
            feather.write_dataframe(pd.DataFrame(c), csvname)
            del c;gc.collect()
    del pro;gc.collect()
    
    # csvname='comtest_for_model_'+str(learn_len)+'_'+str(gap)+'_'+str(fore_len)+'_all'+'textxianglianghua_2'+'.csv'
    # pd.DataFrame(c).to_csv(csvname,index=False)
    
    csvname='comtest_for_model_'+str(learn_len)+'_'+str(gap)+'_'+str(fore_len)+'_all'+'textxianglianghua_2'+'.feature'
    feather.write_dataframe(pd.DataFrame(c), csvname)
    
    del c;gc.collect()

    all_data_frame = []
    row_count = 0
    for file in ['comtest_for_model_'+str(learn_len)+'_'+str(gap)+'_'+str(fore_len)+'_all'+'textxianglianghua_0'+'.feature' , 'comtest_for_model_'+str(learn_len)+'_'+str(gap)+'_'+str(fore_len)+'_all'+'textxianglianghua_1'+'.feature' , 'comtest_for_model_'+str(learn_len)+'_'+str(gap)+'_'+str(fore_len)+'_all'+'textxianglianghua_2'+'.feature' ]:
        data_frame = feather.read_dataframe(file)
        all_data_frame.append(data_frame)
        os.remove(file)
        # axis=0纵向合并 axis=1横向合并
    c = pd.concat(all_data_frame, axis=0, ignore_index=True)
    c = pd.concat([comtest.iloc[:,0:7+40*learn_len],c,pd.DataFrame(comtest.iloc[:,-1])], axis=1) 
    
    csvname='comtest_for_model_'+str(learn_len)+'_'+str(gap)+'_'+str(fore_len)+'_all.feature'    
    # c.to_csv(csvname, index=False, encoding="utf-8") 
    del all_data_frame;del data_frame;gc.collect()
    
    with open(csvname, 'wb') as f:
        feather.write_dataframe(c, f)
        


'''
############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################



import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection

import lightgbm as lgb
from sklearn.metrics import roc_auc_score,accuracy_score,f1_score,precision_recall_curve,matthews_corrcoef,confusion_matrix
from numpy import *
# import time
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import feather
import concurrent

def cum_95CI(j):
    for i in range(2000):
        # bootstrap by sampling with replacement on the prediction indices
        indices=np.random.randint(0,len(pro_comm_Pre[:,1]) - 1,int(len(pro_comm_Pre[:,1])/2))
        
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
        else:
            break
    
    #    score = roc_auc_score(y_true[indices], y_pred[indices])
    eva_CI = evaluating_indicator(y_true=y_true[indices], y_test=blo_comm_Pre[indices], y_test_value=pro_comm_Pre[indices,1])
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
    
    # for i in ['ACC','AUC','AUPRC','BER','F1_score','KAPPA','MCC','TNR','TPR']:   
    #     sorted_scores = np.array(cum_95CI_pro[i]); sorted_scores.sort()
    #     print("Confidence interval for the "+i+": [{:0.6f} - {:0.6}]".format(sorted_scores[int(0.025 * len(sorted_scores))], sorted_scores[int(0.975 * len(sorted_scores))]))
    return cum_95CI_pro


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



def run_ards_ser(learn_len,gap,aim_label,fore_len=4,only_ni_param=False,min_set=False):
    global comtest,pro_comm_Pre,y_true,blo_comm_Pre 
    csvname='comtest_for_model_'+str(learn_len)+'_'+str(gap)+'_'+str(fore_len)+'_all.feature'
    comtest = feather.read_dataframe(csvname)  
    # comtest['hospitalid'] = comtest['patientunitstayid']
    
    if only_ni_param==True:
        comtest['hospitalid'] = comtest['patientunitstayid']
        comtest = pd.concat([comtest.iloc[:,0:7+40*learn_len],pd.DataFrame(comtest.iloc[:,-1])], axis=1) 
        # drop_param=[]
        # for i in ['pao2','paco2','ph','aniongap','basedeficit','baseexcess','ethnicity','hospitalid','sao2']:
        # # for i in ['pao2','ph','aniongap','basedeficit','baseexcess','ethnicity','hospitalid','sao2']:
        #     for j in range(1,learn_len+1):
        #         drop_param = drop_param+[i+'_'+str(j)]
        # comtest.drop(list(drop_param),axis=1,inplace=True) 
    
    if min_set==True:
        # comtest['hospitalid'] = comtest['patientunitstayid']
        comtest['spo2_1'] = comtest['paco2_1']
        min_set_param=['hospitalid','patientunitstayid','c_offset']
        for i in [
                # 'bp_systolic','bp_diastolic','bp_mean','heartrate','respiratoryrate','spo2','paco2'
                #############################################################################################
                    'fio2','peep','o2_flow','etco2','heartrate','respiratoryrate','spo2','bp_systolic','bp_diastolic','bp_mean',
                    'tv_each_kg_ibw','plateau_pressure','flow_sensitivity','rr','peak_pressure','exhaled_tv_machine',
                    'lpm_o2','mean_airway_pressure','pressure_control','pressure_support','peak_flow','humidifier_temp',
                    'vent_rate','exhaled_tv_patient','tidal_volume_set','compliance','exhaled_mv'
                #############################################################################################
                  # 'paco2', 'ph', 'aniongap','baseexcess', 'basedeficit', 'spo2', 'o2_flow'
                ]:
            for j in range(1,learn_len+1):
                min_set_param=min_set_param+[i+'_'+str(j)]
        comtest = pd.concat([comtest[min_set_param],pd.DataFrame(comtest.iloc[:,-1])], axis=1) 
    
    comtest.drop(['patientunitstayid'],axis=1,inplace=True) #patientunitstayid,hospitalid
    
    # comtest_tra = comtest.iloc[:,0:7+40*learn_len] 
    # comtest_tra['pf'] = comtest.pao2_1/comtest.fio2_1
    # comtest_tra['oi'] = comtest.fio2_1*comtest.mean_airway_pressure_1/comtest.pao2_1
    # comtest_tra['osi'] = comtest.fio2_1*comtest.mean_airway_pressure_1/comtest.spo2_1
    # comtest_tra['label'] = comtest.label
    
    icustay_id=list(set(comtest['hospitalid']))
    
    scaler = StandardScaler()   #对病例数据进行标准化处理
    comtest.iloc[:,1:comtest.shape[1]-1]=scaler.fit_transform(comtest.iloc[:,1:comtest.shape[1]-1])
    #x_train_for_vail, x_test, y_train_for_vail, y_true = model_selection.train_test_split(comtest.iloc[0:len(comtest),0:comtest.shape[1]-1],comtest.iloc[0:len(comtest),-1], test_size = 0.2,random_state = 1)    
    
    x_train_for_vail, x_test, y_train_for_vail, y_true = model_selection.train_test_split(np.array(icustay_id),range(len(icustay_id)), test_size = 0.6,random_state = 1)    
    x_train_for_vail=spec_for_ser('comtest',x_train_for_vail);y_train_for_vail=x_train_for_vail.iloc[:,-1];x_train_for_vail_group=x_train_for_vail.iloc[:,0];x_train_for_vail=x_train_for_vail.iloc[:,1:x_train_for_vail.shape[1]-1]
    x_test=spec_for_ser('comtest',x_test);y_true=x_test.iloc[:,-1];x_test_group=x_test.iloc[:,0];x_test=x_test.iloc[:,1:x_test.shape[1]-1]
    y_true_roc=y_true
    # background=np.random.choice(x_train_for_vail.shape[0], 100000, replace=False)
    
    
    comm = lgb.LGBMClassifier(class_weight=None,n_jobs=-1)
    y_train_for_vail[y_train_for_vail==aim_label]=10;y_true[y_true==aim_label]=10
    y_train_for_vail[y_train_for_vail<=3]=0;y_true[y_true<=3]=0
    y_train_for_vail[y_train_for_vail>3]=1;y_true[y_true>3]=1
    comm.fit(x_train_for_vail , y_train_for_vail)
    
    pro_comm_Pre = comm.predict_proba(x_train_for_vail)
    RightIndex=[]
    for jj in tqdm(range(1,100)): #计算模型在不同分类阈值下的各项指标
        blo_comm_Pre = blo(pro_comm_Pre[:,1],jj)
        eva_comm = evaluating_indicator(y_true=y_train_for_vail, y_test=blo_comm_Pre, y_test_value=pro_comm_Pre[:,1])
        RightIndex.append(abs(eva_comm['TPR'] - eva_comm['TNR']))
    RightIndex=np.array(RightIndex,dtype=np.float16)
    position=np.argmin(RightIndex)  #选择出使得敏感性特异性最小的阈值作为分类阈值输出，作为测试集上使用的参数保存
    pro_comm_Pre = comm.predict_proba(x_test)
    blo_comm_Pre = blo(pro_comm_Pre[:,1],position)
    eva_comm = evaluating_indicator(y_true=y_true, y_test=blo_comm_Pre, y_test_value=pro_comm_Pre[:,1])
    print(eva_comm)
    # cum_95CI_pro = cumCI(y_true,pro_comm_Pre[:,1])
    # for i in ['ACC','AUC','AUPRC','BER','F1_score','KAPPA','MCC','TNR','TPR']:   
    #     sorted_scores = np.array(cum_95CI_pro[i]); sorted_scores.sort()
    #     print("Confidence interval for the "+i+": [{:0.6f} - {:0.6}]".format(sorted_scores[int(0.025 * len(sorted_scores))], sorted_scores[int(0.975 * len(sorted_scores))]))
    
    lgb_feat_imp = pd.Series(comm.feature_importances_,list(x_test)).sort_values(ascending=False)
    
    x=x_test.copy();y=y_true.copy()





    comtest = feather.read_dataframe(csvname)  
    
    if only_ni_param==True:
        comtest['hospitalid'] = comtest['patientunitstayid']
        comtest = pd.concat([comtest.iloc[:,0:7+40*learn_len],pd.DataFrame(comtest.iloc[:,-1])], axis=1) 
    
    if min_set==True:
        comtest['spo2_1'] = comtest['paco2_1']
        min_set_param=['hospitalid','patientunitstayid','c_offset']
        for i in [
                # 'bp_systolic','bp_diastolic','bp_mean','heartrate','respiratoryrate','spo2','paco2'
                #############################################################################################
                    'fio2','peep','o2_flow','etco2','heartrate','respiratoryrate','spo2','bp_systolic','bp_diastolic','bp_mean',
                    'tv_each_kg_ibw','plateau_pressure','flow_sensitivity','rr','peak_pressure','exhaled_tv_machine',
                    'lpm_o2','mean_airway_pressure','pressure_control','pressure_support','peak_flow','humidifier_temp',
                    'vent_rate','exhaled_tv_patient','tidal_volume_set','compliance','exhaled_mv'
                #############################################################################################
                  # 'paco2', 'ph', 'aniongap','baseexcess', 'basedeficit', 'spo2', 'o2_flow'
                ]:
            for j in range(1,learn_len+1):
                min_set_param=min_set_param+[i+'_'+str(j)]
        comtest = pd.concat([comtest[min_set_param],pd.DataFrame(comtest.iloc[:,-1])], axis=1) 
    
    comtest.drop(['patientunitstayid'],axis=1,inplace=True) #patientunitstayid,hospitalid
    icustay_id=list(set(comtest['hospitalid']))
    x_train_for_vail, x_test, y_train_for_vail, y_true = model_selection.train_test_split(np.array(icustay_id),range(len(icustay_id)), test_size = 0.6,random_state = 1)    
    x_train_for_vail=spec_for_ser('comtest',x_train_for_vail);y_train_for_vail=x_train_for_vail.iloc[:,-1];x_train_for_vail_group=x_train_for_vail.iloc[:,0];x_train_for_vail=x_train_for_vail.iloc[:,1:x_train_for_vail.shape[1]-1]
    x_test=spec_for_ser('comtest',x_test);y_true=x_test.iloc[:,-1];x_test_group=x_test.iloc[:,0];x_test=x_test.iloc[:,1:x_test.shape[1]-1]
    y_true_roc=y_true
    
    return eva_comm,lgb_feat_imp,comm,x,y,blo_comm_Pre,pro_comm_Pre,x_test.c_offset


eva_comm,lgb_feat_imp,comm,x_test,y_true,blo_comm_Pre,y_test_value,c_offset = run_ards_ser(learn_len = 1, gap = 4, aim_label = 3, min_set=True)  #min_set,only_ni_param



    
# # # # # # # # # # # # # # # # # # # # # # # # # shap参数分析 # # # # # # # # # # # # # # # # # # # # # # # # # # # 
new_columns=[
    'c_offset','fio2','peep','o2_flow','etco2','heartrate','respiratoryrate',\
  'spo2','bp_systolic','bp_diastolic','bp_mean','tv_each_kg_ibw','plateau_pressure','flow_sensitivity',\
  'rr','peak_pressure','exhaled_tv_machine','lpm_o2','mean_airway_pressure','pressure_control',\
  'pressure_support','peak_flo2','humidifier_temp','vent_rate','exhaled_tv_patient','tidal_volume_set',\
  'compliance','exhaled_mv'
    ]
x_test.columns=new_columns
# x_test=pd.DataFrame(np.array(x_test),columns={})

import shap
explainer = shap.TreeExplainer(comm)
shap_values = explainer.shap_values(x_test)
# shap.summary_plot(shap_values[0], x_test)

# # # # # # # # # # # # # 绘制热力图 # # # # 

# shap_value_for_ana_charttime=pd.DataFrame(np.array(shap_values).reshape(-1,param_num),columns=param_name[1:])
# shap_value_for_ana_charttime.chart_time=pd.DataFrame(np.array(x_test).reshape(-1,param_num),columns=param_name[1:]).chart_time

# shap_value_for_ana_charttime=x_test.copy();shap_value_for_ana_charttime.c_offset=c_offset
shap_value_for_ana_charttime=pd.DataFrame(np.array(shap_values[0]),columns=new_columns);shap_value_for_ana_charttime.c_offset=c_offset
shap_value_for_ana_charttime = shap_value_for_ana_charttime.abs().groupby('c_offset',as_index=True).mean()
shap_value_for_ana_charttime=shap_value_for_ana_charttime.iloc[5:56,:]

param_name=['fio2','peep','o2_flow','etco2','heartrate','respiratoryrate',\
  'spo2','bp_systolic','bp_diastolic','bp_mean','tv_each_kg_ibw','plateau_pressure','flow_sensitivity',\
  'rr','peak_pressure','exhaled_tv_machine','lpm_o2','mean_airway_pressure','pressure_control',\
  'pressure_support','peak_flo2','humidifier_temp','vent_rate','exhaled_tv_patient','tidal_volume_set',\
  'compliance','exhaled_mv']
shap_value_for_ana_charttime=shap_value_for_ana_charttime[param_name]

import matplotlib.pyplot as plt
import seaborn as sns

# shap_values=np.array(shap_values).reshape(-1,learn_len,22)
# shap_values=shap_values.reshape(shap_values.shape[1],shap_values.shape[2],shap_values.shape[3])
# z=pd.DataFrame(np.abs(shap_values).mean(1).reshape(learn_len,22),columns=param_name[1:]);z['timesteps']=[0,1,2,3];z=z.groupby('timesteps',as_index=True).mean()

sns.set()
ax = sns.heatmap(shap_value_for_ana_charttime.T, cmap="YlGnBu",linewidths=0.005,yticklabels=True)   # cmap是热力图颜色的参数
plt.yticks(fontweight='bold',fontsize=15)
plt.xticks(fontweight='bold',fontsize=15)
plt.xlabel("hours",fontsize=24,fontweight='bold')
plt.show()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


# # # # # # # # # # # # # # # # # # # # # # # # # 绘制小提琴图 # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# import matplotlib.pyplot as plt
# import seaborn as sns

# #设置绘图风格
# plt.style.use('ggplot')
# #处理中文乱码
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# #坐标轴负号的处理
# plt.rcParams['axes.unicode_minus']=False
# # 读取数据
# # tips = pd.read_excel(r'酒吧消费数据.xlsx')
# tips = x_test.copy();tips['label']=y_true;tips['probability']=y_test_value[:,1];tips['c_offset']=c_offset
# # 绘制分组小提琴图
# z = tips.label.value_counts() 
# print(z)
# sns.violinplot(x = "c_offset", # 指定x轴的数据
#                 y = "probability", # 指定y轴的数据
#                 hue = "label", # 指定分组变量
#                 data = tips, # 指定绘图的数据集
#                 order = [5,10,15,20,25,30,35,40,45,50,55], # 指定x轴刻度标签的顺序
#                 scale = 'count', # 以男女客户数调节小提琴图左右的宽度
#                 split = True # 将小提琴图从中间割裂开，形成不同的密度曲线；
#                 # palette = 'RdBu' # 指定不同性别对应的颜色（因为hue参数为设置为性别变量）
#               )
# # sns.violinplot(x = "c_offset",y = "probability",hue = "label",data = tips,order = ['12','20','25'])

# # 设置图例
# # plt.legend(['Negative', 'Positive'] ,loc = 'upper left', ncol = 2,fontsize=28)
# plt.legend(loc = 'upper left', ncol = 2,fontsize=28)
# # 显示图形
# plt.ylim(-0.4,1.5)
# plt.xticks(fontsize=28)
# plt.xlabel("hours",fontsize=28,fontweight='bold')
# plt.yticks(fontsize=28)
# plt.ylabel("Score",fontsize=28,fontweight='bold')
# plt.show()
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  


# # # # # # # # # # # # # # # # # # # # # # # # # 绘制AUC随住院时间变化图 # # # # # # # # # # # # # # # # # # # # # # # # # # #  
# index_for_plot='AUC';first_offset=6
# y=y_true.copy();b=blo_comm_Pre.copy();p=y_test_value.copy()
# y_true=y[c_offset==first_offset].reset_index().label;blo_comm_Pre=b[c_offset==first_offset];pro_comm_Pre=p[c_offset==first_offset]
# eva_comm=[evaluating_indicator(y_true=y_true, y_test=blo_comm_Pre, y_test_value=pro_comm_Pre[:,1])]
# sorted_scores=[cumCI(y_true,pro_comm_Pre[:,1])]

# eva_comm_auc=[eva_comm[0][index_for_plot]];
# cum_95CI_pro_auc_low=[sorted_scores[0][index_for_plot][int(0.025 * len(sorted_scores[0]))]]
# cum_95CI_pro_auc_high=[sorted_scores[0][index_for_plot][int(0.975 * len(sorted_scores[0]))]]
# # for i in list(set(c_offset)):
# for i in range(first_offset+1,56):
#     y_true=y[c_offset==i].reset_index().label;blo_comm_Pre=b[c_offset==i];pro_comm_Pre=p[c_offset==i]
#     eva_comm = eva_comm+[evaluating_indicator(y_true=y_true, y_test=blo_comm_Pre, y_test_value=pro_comm_Pre[:,1])]
#     sorted_scores = sorted_scores+[cumCI(y_true,pro_comm_Pre[:,1])]
    
#     eva_comm_auc.append(eva_comm[i-first_offset][index_for_plot])
#     cum_95CI_pro_auc_low.append(sorted_scores[i-first_offset][index_for_plot][int(0.025 * len(sorted_scores[i-first_offset]))])
#     cum_95CI_pro_auc_high.append(sorted_scores[i-first_offset][index_for_plot][int(0.975 * len(sorted_scores[i-first_offset]))])

# # # # # # # # # # # # # plot # # # # 
# # import seaborn as sns
# import matplotlib.pyplot as plt
# import matplotlib
# import numpy as np
# # import scipy.stats as st
# matplotlib.rcParams.update({'font.size': 12})


# predicted_expect=eva_comm_auc.copy()
# low_CI_bound=cum_95CI_pro_auc_low.copy()
# high_CI_bound=cum_95CI_pro_auc_high.copy()

# x = np.linspace(0, 50, num=50)

# plt.plot(predicted_expect, linewidth=3., label='AUC with Continuous Non-invasive parameters')
# plt.fill_between(x, low_CI_bound, high_CI_bound, alpha=0.5,
#                 label='95% CI with Continuous Non-invasive parameters')


# # predicted_expect=np.array([0.884450053317942,0.869080465561182,0.784309975279747,0.746296191414668,0.710925290170664])
# # low_CI_bound=np.array([0.872720,0.858800,0.767167,0.728699,0.688960])
# # high_CI_bound=np.array([0.89518,0.87666,0.802918,0.765804,0.734975])

# # plt.plot(predicted_expect, linewidth=3., label='AUC with Complete feature set')
# # plt.fill_between(x, low_CI_bound, high_CI_bound, alpha=0.5,
# #                 label='95% CI with Complete feature set') #confidence interval
# # plt.legend(fontsize=24,loc='lower left')


# # values = ['1 hour', '2 hours', '4 hours', '8 hours', '12 hours'] 
# # plt.xticks(x,values,fontsize=28)
# # plt.xlabel("Lead time",fontsize=28,fontweight='bold')
# # plt.yticks(fontsize=28)
# # plt.ylabel("AUC",fontsize=28,fontweight='bold')
# # plt.ylim(0.5,0.95)
# plt.show()













