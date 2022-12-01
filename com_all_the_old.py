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
    s=time.time()
    icu_id_param=comtest[comtest.patientunitstayid==i]
#    chart_time_opt=np.array(icu_id_param.c_offset[icu_id_param.label.notnull()])
    chart_time_opt=np.array(icu_id_param.c_offset[icu_id_param.death_label.notnull()])
    chart_time_opt=np.append(chart_time_opt,np.array(icu_id_param.c_offset[icu_id_param.to_discharge.notnull()]))
    chart_time_opt=np.array(icu_id_param.c_offset[icu_id_param.vent_duration.notnull()])
    chart_time_opt=np.append(chart_time_opt,np.array(icu_id_param.c_offset[icu_id_param.label.notnull()]))    
    chart_time_opt=np.array(list(set(chart_time_opt)));chart_time_opt=np.array(sorted(chart_time_opt))
    chart_time_opt=chart_time_opt[chart_time_opt>=learn_len+gap] #
    
    if len(chart_time_opt)!=len(set(chart_time_opt)):
        return
    chart_time_opt=chart_time_opt[chart_time_opt>=learn_len+gap] #
    temp_return=np.ones([1,len(new_name)]); #len=2时， 160
    if len(chart_time_opt)!=0:
        icu_id_param=inter_learn_len(chart_time_opt,icu_id_param)
        for jj in chart_time_opt:
            for fore_len_l in range(0,fore_len):
                if jj-gap-learn_len-fore_len_l>=0:
                    index_1=np.ones([icu_id_param.shape[0],1])*0;index_1[icu_id_param.c_offset<jj-gap-fore_len_l]=1
                    index_2=np.ones([icu_id_param.shape[0],1])*0;index_2[icu_id_param.c_offset>=jj-gap-learn_len-fore_len_l]=1
                    temporary=icu_id_param[index_1+index_2>=2];#temporary=temporary.iloc[:,0:icu_id_param.shape[1]-2]
                    temp=np.append(np.array(temporary[['patientunitstayid','age','gender_0','gender_1','bmi']].iloc[0].to_frame().T),np.array(temporary.drop(columns=['patientunitstayid','age','gender_0','gender_1','bmi','c_offset','death_label','to_discharge','vent_duration','label'])).T.reshape([1,-1]))  # 缺少'out_time'
                    temp=np.append(temp,np.array(icu_id_param[icu_id_param.c_offset==jj].death_label));temp=np.append(temp,np.array(icu_id_param[icu_id_param.c_offset==jj].to_discharge))
                    temp=np.append(temp,np.array(icu_id_param[icu_id_param.c_offset==jj].vent_duration));temp=np.append(temp,np.array(icu_id_param[icu_id_param.c_offset==jj].label)).reshape(1,-1)
                    temp_return=np.append(temp_return,temp,axis=0)
    temp_return=np.delete(temp_return,0,axis=0)
    temp_return=pd.DataFrame(temp_return)
    print(time.time()-s)
    return temp_return
    
#
#
#learn_len=1;gap=12;fore_len=4
#patientunitstayid=list(set(comtest.patientunitstayid))
#
#with concurrent.futures.ProcessPoolExecutor(4) as executor:
#    pro=list(executor.map(final_data,patientunitstayid))
#
#for i in range(0,len(pro)):
#    if i==0:
#        input_mulit=('pro[{}]'.format(i))
#    else:
#        input_mulit=(input_mulit+',pro[{}]'.format(i))
#
#p=pd.concat(eval(input_mulit),axis=0,ignore_index=True);comtest=p;del p;del pro
##comtest_for_mac=np.ones([1,86])
##for i in patientunitstayid:
###    comtest_for_mac=final_data(i)
##    comtest_for_mac=np.concatenate((comtest_for_mac,final_data(i)),axis=0)
##comtest_for_mac=np.delete(comtest_for_mac,0,axis=0)
###comtest=pd.DataFrame(comtest)
##comtest=pd.DataFrame(comtest_for_mac)
##del comtest_for_mac
##comtest=pd.DataFrame(comtest)
##
#
#old_name=comtest_index
#new_name=['patientunitstayid','age','gender_0','gender_1','bmi']
#for j in range(6,82+1):
#    for i in range(1,learn_len+1):
#        new_name=new_name+[old_name[j]+'_'+str(i)]
#new_name=new_name+['death_label']+['to_discharge']+['vent_duration']+['label']     #缺少'out_time'
##p.rename=[new_name]   
#comtest.columns=new_name   
#csvname='comtest_for_model_'+str(learn_len)+'_'+str(gap)+'_'+str(fore_len)+'_all'+'.csv'
#comtest.to_csv('/media/amms/80D89D09D89CFE9A/bert_model/'+csvname,index=False)
##del comtest
#
#
#
##comtest = pd.read_csv("/media/amms/80D89D09D89CFE9A/bert_model/comtest_for_model_1_1_4_all.csv")
###comtest=comtest.iloc[0:6000,:]
##index=comtest.columns
##z=comtest.iloc[:,0:1];z['text_data']=np.nan
##for i in range(comtest.shape[0]):  #comtest.shape[0]:
##    line=str()
##    for j in range(comtest.shape[1]-2):
##        line=line+index[j+1]+' is '+ str(comtest.iloc[i,j+1]) + '. '
##    z.iloc[i,1]=line
##z['label']=comtest.iloc[:,-1]
##comtest=z
##del z
##comtest.to_csv('/media/amms/80D89D09D89CFE9A/bert_model/comtest_for_model_1_1_4_nltk_all',index=False)
#comtest = pd.read_csv('/media/amms/80D89D09D89CFE9A/bert_model/comtest_for_model_1_12_4_all.csv')
#patientunitstayid=list(set(comtest.patientunitstayid))
#
#from bert_serving.client import BertClient
##bc = BertClient(ip='localhost')
#bc = BertClient()
#
#
##comtest =comtest_copy
#
#cc=comtest.iloc[:,1:44]
#cc=pd.DataFrame(np.hstack((np.array(comtest.patientunitstayid).reshape(-1,1),cc,np.array(comtest.label).reshape(-1,1))) ) 
#index=cc.columns
#cc.rename(columns={index[0]:'patientunitstayid', index[-1]:'label'}, inplace = True)
#csvname='comtest_for_model_'+str(learn_len)+'_'+str(gap)+'_'+str(fore_len)+'_all'+'jiegouhua'+'.csv'
#cc.to_csv('/media/amms/80D89D09D89CFE9A/bert_model/'+csvname,index=False)
#
#
#comtest_copy = comtest.iloc[:,45:-1]; index=comtest_copy.columns;cc=comtest.iloc[:,1:44] 
#z=np.array(comtest.patientunitstayid).reshape(-1,1)
#zz=np.array(comtest.label).reshape(-1,1)
#comtest_copy=comtest_copy.astype(str)
#del comtest;gc.collect()
##z=comtest.iloc[:,0:1];z['text_data']=np.nan
#le=comtest_copy.shape[1];
#def be_list(i):
#    line=list(comtest_copy.iloc[i,:])
#    return line
#    #        line=np.concatenate((line,bc.encode(list([str(comtest.iloc[i,j])]))),axis = 1)
#
#with concurrent.futures.ProcessPoolExecutor(4) as executor:
#    pro=list(executor.map(be_list,range(comtest_copy.shape[0])))
#del comtest_copy;gc.collect()
#
##for i in range(0,len(pro)):
##    if i==0:
##        input_mulit=('pro[{}]'.format(i))
##    else:
##        input_mulit=(input_mulit+',pro[{}]'.format(i))
#c=[];len_bert=500
#for j in range(228,int(len(pro)/len_bert)-1):
#    print(j)
#    c_pro=[];
#    if j==int(len(pro)/len_bert)-2:
#        for i in range(j*len_bert,len(pro)):
#            if i==j*len_bert:
#                c_pro=pro[i][0:le];
#            else:
#                c_pro.extend(pro[i][0:le])
#    else: 
##        if j==0:
##            pro_num=j*len_bert+1
##        else:
##        pro_num=j*len_bert
#        for i in range(j*len_bert,(j+1)*len_bert):
#            if i==j*len_bert:
#                c_pro=pro[i][0:le]
#            else:
#                c_pro.extend(pro[i][0:le])
#    if j==0:
#        c=bc.encode(c_pro).reshape(-1,le*128)
#    else:
#        c=concatenate((c,bc.encode(c_pro).reshape(-1,le*128)),axis=0)
#
##for i in range(len(pro)):
##    if len(pro[i])>40:
##        print(i)
#del pro;gc.collect()
#c=pd.DataFrame(np.hstack((z,cc,c,zz)) ) 
#index=c.columns
#c.rename(columns={index[0]:'patientunitstayid', index[-1]:'label'}, inplace = True)
#csvname='comtest_for_model_'+str(learn_len)+'_'+str(gap)+'_'+str(fore_len)+'_all'+'jiegouhua+textxianglianghua'+'.csv'
#c.to_csv('/media/amms/80D89D09D89CFE9A/bert_model/'+csvname,index=False)
#
#del c;gc.collect()
#
#



##############################################################################3
##############################################################################3
##############################################################################3
##############################################################################3
##############################################################################3
import pandas as pd
import concurrent
import gc
pd.set_option('mode.chained_assignment', None)
import numpy as np
from numpy import concatenate
import time
import itertools
import os

#from functools import partial
from bert_serving.client import BertClient
#bc = BertClient(ip='localhost')
bc = BertClient()

for gap in [12,8,4,1]:
    
    comtest=pd.read_csv('/media/amms/80D89D09D89CFE9A/bert_model/test.csv')
    comtest_index=list(comtest);
    learn_len=4;gap=gap;fore_len=4
    patientunitstayid=list(set(comtest.patientunitstayid))
    
    
    old_name=comtest_index
    new_name=['patientunitstayid','age','gender_0','gender_1','bmi']
    for j in range(6,82+1):
        for i in range(1,learn_len+1):
            new_name=new_name+[old_name[j]+'_'+str(i)]
    new_name=new_name+['death_label']+['to_discharge']+['vent_duration']+['label']     #缺少'out_time'
    
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        pro=list(executor.map(final_data,patientunitstayid))
    
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
    
    cc=comtest.iloc[:,0:5+40*learn_len]
    
#    cc=pd.DataFrame(np.hstack((np.array(comtest.patientunitstayid).reshape(-1,1),cc,np.array(comtest.label).reshape(-1,1))) ) 
    cc['label']=comtest.label;cc['hospitalid']=comtest.hospitalid_1; cc['death_label']=comtest.death_label
    cc['to_discharge']=comtest.to_discharge; cc['vent_duration']=comtest.vent_duration
        
    
    csvname='comtest_for_model_'+str(learn_len)+'_'+str(gap)+'_'+str(fore_len)+'_all'+'jiegouhua'+'.csv'
#    cc.to_csv('/media/amms/80D89D09D89CFE9A/bert_model/'+csvname=False)
    cc.to_csv('/media/amms/D23CA3653CA34379/'+csvname,index=False)
    
    pro = comtest.iloc[:,5+40*learn_len:(comtest).shape[1]-4]
    le=pro.shape[1]
    del comtest;gc.collect()
    
    

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
    c=[];len_bert=2000;max_seq_len=1000
    for j in range(0,int(len(pro)/len_bert)):
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
            csvname='comtest_for_model_'+str(learn_len)+'_'+str(gap)+'_'+str(fore_len)+'_all'+'jiegouhua+textxianglianghua_0'+'.csv'
            pd.DataFrame(c).to_csv('/media/amms/D23CA3653CA34379/'+csvname,index=False)
            del c;gc.collect()
        if j == int((len(pro)/len_bert)/3)*2:
            csvname='comtest_for_model_'+str(learn_len)+'_'+str(gap)+'_'+str(fore_len)+'_all'+'jiegouhua+textxianglianghua_1'+'.csv'
            pd.DataFrame(c).to_csv('/media/amms/D23CA3653CA34379/'+csvname,index=False)
            del c;gc.collect()
    del pro;gc.collect()
    
    csvname='comtest_for_model_'+str(learn_len)+'_'+str(gap)+'_'+str(fore_len)+'_all'+'jiegouhua+textxianglianghua_2'+'.csv'
    pd.DataFrame(c).to_csv('/media/amms/D23CA3653CA34379/'+csvname,index=False)
    del c;gc.collect()

    all_data_frame = []
    row_count = 0
    for file in ['comtest_for_model_'+str(learn_len)+'_'+str(gap)+'_'+str(fore_len)+'_all'+'jiegouhua+textxianglianghua_0'+'.csv' , 'comtest_for_model_'+str(learn_len)+'_'+str(gap)+'_'+str(fore_len)+'_all'+'jiegouhua+textxianglianghua_1'+'.csv' , 'comtest_for_model_'+str(learn_len)+'_'+str(gap)+'_'+str(fore_len)+'_all'+'jiegouhua+textxianglianghua_2'+'.csv' ]:
        data_frame = pd.read_csv('/media/amms/D23CA3653CA34379/'+file)
        all_data_frame.append(data_frame)
        os.remove('/media/amms/D23CA3653CA34379/'+file)
        # axis=0纵向合并 axis=1横向合并
    c = pd.concat(all_data_frame, axis=0, ignore_index=True, sort=True)
    csvname='comtest_for_model_'+str(learn_len)+'_'+str(gap)+'_'+str(fore_len)+'_all'+'jiegouhua+textxianglianghua'+'.csv'    
    c.to_csv('/media/amms/D23CA3653CA34379/'+csvname, index=False, encoding="utf-8") 
    del all_data_frame;gc.collect()
    
    c=pd.DataFrame(np.hstack((cc.iloc[:,0:5+40*learn_len],c,np.array(cc.label).reshape(-1,1))) ) 

    
    
    print(time.strftime('%Y-%m-%d %H:%M:%S'))
    
    index=c.columns
    c.rename(columns={index[0]:'patientunitstayid', index[-1]:'label'}, inplace = True)
    
    c['hospitalid']=cc.hospitalid; c['death_label']=cc.death_label
    c['to_discharge']=cc.to_discharge; c['vent_duration']=cc.vent_duration
    
    del cc ;gc.collect()
    csvname='comtest_for_model_'+str(learn_len)+'_'+str(gap)+'_'+str(fore_len)+'_all'+'jiegouhua+textxianglianghua'+'.csv'
    c.to_csv('/media/amms/D23CA3653CA34379/'+csvname,index=False)
#    c.to_csv('/media/amms/80D89D09D89CFE9A/bert_model/'+csvname,index=False)
    del c;gc.collect()




#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 10:34:18 2020

@author: amms
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 09:25:07 2020

@author: amms
"""

#import pandas as pd
#import numpy as np
#from numpy import concatenate


#import torch
#torch.cuda.empty_cache()

#from sklearn.datasets import load_iris
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.svm import SVC 
import gc
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn import model_selection
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef # MCC
from sklearn.metrics import confusion_matrix #混淆矩阵
from numpy import *
import time
from sklearn.model_selection import StratifiedKFold
import scipy
from sklearn import metrics
from sklearn.metrics import r2_score
from pycm import *
from sklearn import ensemble
#from sklearn.ensemble import RandomForestClassifier

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
    F1score =  f1_score(y_true, y_test, average="micro")
    AUC = roc_auc_score(y_true,y_test_value[:,1])
    KAPPA=kappa(c_m)
    
    c={"TPR" : TPR,"TNR" : TNR,"BER" : BER
    ,"ACC" : ACC,"MCC" : MCC,"F1_score" : F1score,"AUC" : AUC,'KAPPA':KAPPA}
    return c

def blo(pro_comm_Pre,jj):     #根据预测概率与最优分类阈值对患者进行生死预测
    blo_Pre=zeros(len(pro_comm_Pre))
    blo_Pre[(pro_comm_Pre[:,1]>(jj*0.001))]=1
    return blo_Pre

#import inspect, re
#def varname(p):
#    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
#        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
#        if m:
#            return m.group(1)

def spec_for_ser(df,patientunitstayid):
    str_df=str(df)
    for i in patientunitstayid:
        if i==patientunitstayid[0]:
            input_mulit=(str_df+"["+str_df+"['patientunitstayid']=={}]".format(i))
        else:
            input_mulit=(input_mulit+","+str_df+"["+str_df+"['patientunitstayid']=={}]".format(i))
    return (pd.concat(eval(input_mulit),axis=0,ignore_index=True))







##################################################### 使用特征全集计算时使用 ################################################
#comtest = pd.read_csv("D:/python_data/6_2_4_derivative/comtest_for_model_2_1_4.csv",nrows=1000)
#comtest=comtest.dropna()
#
#learn_len=1
#old_name=['ni_sysbp','ni_meanbp','ni_diasbp']
#new_name=[]
#for j in range(3):
#    for i in range(1,learn_len+1):
#        new_name=new_name+[old_name[j]+'_'+str(i)]
#    if j!=2 and i!=learn_len:
#        new_name=new_name+','
#comtest=comtest.drop(new_name,axis=1)

##################################################### 使用无创数据计算时使用 ################################################
#comtest = pd.read_csv("D:/Work/需完成/for20200616/comtest_for_model_1_1_4.csv")
#comtest=comtest.dropna()
#learn_len=1
#old_name=['vent_1','vent_0', 'gcs', 'pain_score', 'fio2',
#       'ph', 'aniongap', 'basedeficit', 'baseexcess', 'peep',
#       'o2_flow', 'etco2', 'outputtotal', 'urineoutput', 'heartrate',
#       'respiratoryrate', 'spo2', 'bp_systolic', 'bp_diastolic',
#       'bp_mean', 'tv_each_kg_ibw', 'plateau_pressure',
#       'flow_sensitivity', 'rr', 'peak_pressure', 'exhaled_tv_machine',
#       'lpm_o2', 'mean_airway_pressure', 'pressure_control',
#       'pressure_support', 'peak_flow', 'humidifier_temp', 'vent_rate',
#       'exhaled_tv_patient', 'tidal_volume_set', 'compliance',
#       'exhaled_mv', 'sao2',
#       
#       'pao2', 'paco2','albumin','bilirubin','bun','calcium','creatinine','glucose','bicarbonate','totalco2',
#       'hematocrit','hemoglobin','inr','lactate','platelets','potassium','ptt','sodium','wbc',
#       'bands','alt','ast','alp'
#       ]
#new_name=['patientunitstayid', 'age', 'gender_0', 'gender_1', 'bmi']
#for j in range(0,len(old_name)):
#    for i in range(1,learn_len+1):
#        new_name=new_name+[old_name[j]+'_'+str(i)]
#new_name=new_name+['death_label']+['to_discharge']+['vent_duration']+['label']


#comtest=comtest.iloc[np.array(comtest[new_name[0:len(new_name)-4]].dropna().index),:]
##comtest = pd.read_csv("D:/Work/需完成/for20200519/动态ARDS亚型分类工作/comtest_for_model_1_1_4.csv",usecols=new_name)
#scaler = StandardScaler()   #对病例数据进行标准化处理
#comtest.iloc[:,1:comtest.shape[1]-4]=scaler.fit_transform(comtest.iloc[:,1:comtest.shape[1]-4]) #4个label,所以是-4

#del comtest

#x_train=np.append(x_train, x_test,axis=0)
#y_train=np.append(y_train, y_true,axis=0)
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials



def hyperopt_class(params):
#    t = params['type']
#    del params['type']
#    if t == 'lgbm':
    clf = lgb.LGBMClassifier(**params,device_type='gpu')
#    clf = xgb.XGBClassifier(**params)
#    elif t == 'svm':
#        clf = SVC(**params)
#    else:
#        return 0
    return cross_val_score(clf, x_train_for_vail, y_train_for_vail,groups= x_train_for_vail_group,cv=3, scoring = 'roc_auc').mean()

#    return cross_val_score(clf, x_train, y_train,cv=5,scoring = 'balanecd_accurary').mean()

#space = hp.choice('classifier_type', [
space_class =       {
#                'type': 'lgbm',
                'max_bin': hp.choice('max_bin', range(20,200)),
                'learning_rate': hp.uniform('learning_rate',0.0001,0.01),
                'n_estimators': hp.choice('n_estimators', range(100,1000)),
                'reg_alpha': hp.uniform('reg_alpha', 0,1),
#                'max_depth': hp.choice('max_depth', range(20,800)),
                'min_child_weight': hp.choice('min_child_weight', range(1,80))
#                'max_leaves': hp.choice('max_leaves', range(1,20))
#                ,'device_type':hp.choice('device_type', ['gpu']),
#                ,'gpu_platform_id':hp.choice('gpu_platform_id', [1])
#                ,'tree_method':hp.choice('tree_method', ['gpu_hist'])
#                ,'gpu_id':hp.choice('gpu_id', [0,1])
#                'feature_fraction':hp.choice('feature_fraction', 0.5)
#                'early_stopping_round':hp.choice('early_stopping_round', [120])
#                ,'num_boost_round': hp.choice('num_boost_round', range(10,300))
                }
#        ,
#space_class =       {
##                'type': 'lgbm',
#                'max_bin': hp.choice('max_bin', range(10,11)),
#                'learning_rate': hp.uniform('learning_rate',0.9,0.91),
#                'n_estimators': hp.choice('n_estimators', range(10,11)),
#                'reg_alpha': hp.uniform('reg_alpha', 0,1),
#                'max_depth': hp.choice('max_depth', range(3,4)),
#                'min_child_weight': hp.choice('min_child_weight', range(3,4)),
#                'max_leaves': hp.choice('max_leaves', range(3,4))
##                ,'device':hp.choice('device', ['gpu'])
#                
#                ,'tree_method':hp.choice('tree_method', ['gpu_hist'])
#                ,'gpu_id':hp.choice('gpu_id', [0,1])
##                'feature_fraction':hp.choice('feature_fraction', 0.5)
##                'early_stopping_round':hp.choice('early_stopping_round', [120])
##                ,'num_boost_round': hp.choice('num_boost_round', range(10,300))
#                }
#        ])

count_class = 0
best_class = 0

def f_class(params):
    global best_class, count_class
    count_class += 1
    auc = hyperopt_class(params.copy())
    if auc > best_class:
#        print ('new best:', acc, 'using', params['type'])
        print ('new best:', auc, 'using')
        best_class = auc
    if count_class % 30 == 0:
        print ('iters:', count_class, ', auc:', auc, 'using', params)
    return {'loss': -auc, 'status': STATUS_OK}

#trials = Trials()
#best = fmin(f, space, algo=tpe.suggest, max_evals=150, trials=trials)
#print ('best:', best) 

##############################################################################

def hyperopt_Regressor(params):
    clf = lgb.LGBMRegressor(**params)
    return cross_val_score(clf, x_train_for_vail, y_train_for_vail,groups= x_train_for_vail_group,cv=3, scoring = 'r2').mean() #使用预测数据和原始数据对应点误差的平方和的均值作为调参标准


space_Regressor =       {
#                'type': 'lgbm',
                'max_bin': hp.choice('max_bin', range(20,3100)),
                'learning_rate': hp.uniform('learning_rate',0.001,0.9),
                'n_estimators': hp.choice('n_estimators', range(10,1000)),
                'reg_alpha': hp.uniform('reg_alpha', 0,1),
                'max_depth': hp.choice('max_depth', range(1,51)),
                'min_child_weight': hp.choice('min_child_weight', range(1,51)),
                'max_leaves': hp.choice('max_leaves', range(1,51))
#                ,'num_boost_round': hp.choice('num_boost_round', range(10,300))
                }


count_Regressor = 0
best_Regressor = -10

def f_Regressor(params):
    global best_Regressor, count_Regressor
    count_Regressor += 1
    pro = hyperopt_Regressor(params.copy())
    if pro > best_Regressor:
#        print ('new best:', acc, 'using', params['type'])
        print ('new best:', pro, 'using', params)
        best_Regressor = pro
#    if count % 30 == 0:
#        print ('iters:', count, ', RMSE:', RMSE, 'using', params)
    return {'loss': -pro, 'status': STATUS_OK}

#trials = Trials()
#best = fmin(f, space_Regressor, algo=tpe.suggest, max_evals=150, trials=trials)
#print ('best:', best) 



##################################################################################
##################################################################################

def RUN(x_train_for_vail,y_train_for_vail,x_train_for_vail_group):   #根据训练集与验证集获取最优分类阈值
    EVA_ACC_TRAIN=[];EVA_ACC_TEST=[]
    EVA_AUC_TRAIN=[];EVA_AUC_TEST=[]
    EVA_MCC_TRAIN=[];EVA_MCC_TEST=[]
    EVA_F1_score_TRAIN=[];EVA_F1_score_TEST=[]
    EVA_BER_TRAIN=[];EVA_BER_TEST=[]
    EVA_KAPPA_TRAIN=[];EVA_KAPPA_TEST=[]
    EVA_TPR_TRAIN=[];EVA_TPR_TEST=[]
    EVA_TNR_TRAIN=[];EVA_TNR_TEST=[]
    CUT_OFF=[]
#    tiaocan_train, ceshi_train, tiaocan_train_test, ceshi_true = cross_validation.train_test_split(comtest.iloc[0:len(comtest),1:comtest.shape[1]-1],comtest.iloc[0:len(comtest),-1], test_size = 0.2,random_state = 0)    
    position=[];
#    skf=StratifiedKFold(n_splits=5)  #设置十折交叉验证
    group_kfold=GroupKFold(n_splits=2)
    tiaocan_train=np.array(x_train_for_vail)
    tiaocan_train_test=np.array(y_train_for_vail)
#    group_kfold.get_n_splits(tiaocan_train,tiaocan_train_test,np.array(x_train_for_vail_group))
#    para_times=0
#
#    for param in para_val:
#        para_times+=1
#        print('第%s个参数值结果验证, 共 %s 个参数值 '%(para_times ,para_val.shape[0]))
    position=[]
    EVA_ACC_TRAIN_cv=[];EVA_ACC_TEST_cv=[]
    EVA_AUC_TRAIN_cv=[];EVA_AUC_TEST_cv=[]
    EVA_MCC_TRAIN_cv=[];EVA_MCC_TEST_cv=[]
    EVA_F1_score_TRAIN_cv=[];EVA_F1_score_TEST_cv=[]
    EVA_BER_TRAIN_cv=[];EVA_BER_TEST_cv=[]
    EVA_KAPPA_TRAIN_cv=[];EVA_KAPPA_TEST_cv=[]
    EVA_TPR_TRAIN_cv=[];EVA_TPR_TEST_cv=[]
    EVA_TNR_TRAIN_cv=[];EVA_TNR_TEST_cv=[]
    CUT_OFF_cv=[]
    times=0
    for train, test in group_kfold.split(tiaocan_train,tiaocan_train_test,np.array(x_train_for_vail_group)):
        alltime_start=time.time()
        times+=1
        x_train=tiaocan_train[train]
        y_train=tiaocan_train_test[train]
        x_test=tiaocan_train[test]
        y_true=tiaocan_train_test[test]        
##########################################################################################################################     
#        comm_cut_off = RandomForestClassifier()
#        comm_cut_off=lgb.LGBMClassifier(device_type='gpu')
        comm_cut_off=xgb.XGBClassifier(tree_method='gpu_hist', gpu_id=1)
##################################################################################
        comm_cut_off.fit(x_train , y_train) 
#            comm.fit_transform(x_train , y_train)    #对机器学习模型进行训练
        pro_comm_Pre = comm_cut_off.predict_proba(x_test)
#        predict_res = comm_cut_off.predict(x_train)
#        print(str(predict_res[y_train==0]).count('0')/y_train[y_train==0].shape[0])
#        print(str(predict_res[y_train==1]).count('1')/y_train[y_train==1].shape[0])
#        print(str(predict_res[y_train==2]).count('2')/y_train[y_train==2].shape[0])
#        print(str(predict_res[y_train==3]).count('3')/y_train[y_train==3].shape[0])
#        print('\n')
#        predict_res = comm_cut_off.predict(x_test)
#        print(str(predict_res[y_true==0]).count('0')/y_true[y_true==0].shape[0])
#        print(str(predict_res[y_true==1]).count('1')/y_true[y_true==1].shape[0])
#        print(str(predict_res[y_true==2]).count('2')/y_true[y_true==2].shape[0])
#        print(str(predict_res[y_true==3]).count('3')/y_true[y_true==3].shape[0])
#        print('\n')
#        predict_res = comm_cut_off.predict(x_test)
#        print(str(predict_res[y_true==0]).count('0')/y_true[y_true==0].shape[0])
#        print(str(predict_res[y_true==1]).count('1')/y_true[y_true==1].shape[0])
#        print(str(predict_res[y_true==2]).count('2')/y_true[y_true==2].shape[0])
#        print(str(predict_res[y_true==3]).count('3')/y_true[y_true==3].shape[0])
        
        RightIndex=[]
        for jj in range(1000): #计算模型在不同分类阈值下的各项指标
            blo_comm_Pre = blo(pro_comm_Pre,jj)
            eva_comm = evaluating_indicator(y_true=y_true, y_test=blo_comm_Pre, y_test_value=pro_comm_Pre)
            RightIndex.append(abs(eva_comm['TPR'] - eva_comm['TNR']))
        RightIndex=np.array(RightIndex,dtype=np.float16)
        position=np.argmin(RightIndex)  #选择出使得敏感性特异性最小的阈值作为分类阈值输出
        position=position.mean()
        CUT_OFF_cv.append(position)
        blo_comm_Pre = blo(pro_comm_Pre,position)
        eva_comm = evaluating_indicator(y_true=y_true, y_test=blo_comm_Pre, y_test_value=pro_comm_Pre)
        
        EVA_ACC_TEST_cv.append(eva_comm['ACC'])
        EVA_AUC_TEST_cv.append(eva_comm['AUC'])
        EVA_MCC_TEST_cv.append(eva_comm['MCC'])
        EVA_F1_score_TEST_cv.append(eva_comm['F1_score'])
        EVA_BER_TEST_cv.append(eva_comm['BER'])
        EVA_KAPPA_TEST_cv.append(eva_comm['KAPPA'])
        EVA_TPR_TEST_cv.append(eva_comm['TPR'])
        EVA_TNR_TEST_cv.append(eva_comm['TNR'])
        pro_comm_Pre = comm_cut_off.predict_proba(x_train)
        blo_comm_Pre = blo(pro_comm_Pre,position)
        eva_comm = evaluating_indicator(y_true=y_train, y_test=blo_comm_Pre, y_test_value=pro_comm_Pre)
        EVA_ACC_TRAIN_cv.append(eva_comm['ACC'])
        EVA_AUC_TRAIN_cv.append(eva_comm['AUC'])
        EVA_MCC_TRAIN_cv.append(eva_comm['MCC'])
        EVA_F1_score_TRAIN_cv.append(eva_comm['F1_score'])
        EVA_BER_TRAIN_cv.append(eva_comm['BER'])
        EVA_KAPPA_TRAIN_cv.append(eva_comm['KAPPA'])
        EVA_TPR_TRAIN_cv.append(eva_comm['TPR'])
        EVA_TNR_TRAIN_cv.append(eva_comm['TNR'])
        alltime_end=time.time()
#            print('第%s个参数值结果验证, 共 %s 个参数值.  done , 第%s次交叉验证   , time: %s  s  '%(para_times ,para_val.shape[0],times ,alltime_end-alltime_start))
#            print('        done , 第%s次验证   , time: %s  s '%(times ,alltime_end-alltime_start)) 
        print('第%s次交叉验证   , time: %s  s  '%(times ,alltime_end-alltime_start))  
    EVA_ACC_TRAIN.append(np.array(EVA_ACC_TRAIN_cv).mean()); EVA_ACC_TEST.append(np.array(EVA_ACC_TEST_cv).mean())
    EVA_AUC_TRAIN.append(np.array(EVA_AUC_TRAIN_cv).mean()); EVA_AUC_TEST.append(np.array(EVA_AUC_TEST_cv).mean())
    EVA_MCC_TRAIN.append(np.array(EVA_MCC_TRAIN_cv).mean()); EVA_MCC_TEST.append(np.array(EVA_MCC_TEST_cv).mean())
    EVA_F1_score_TRAIN.append(np.array(EVA_F1_score_TRAIN_cv).mean()); EVA_F1_score_TEST.append(np.array(EVA_F1_score_TEST_cv).mean())
    EVA_BER_TRAIN.append(np.array(EVA_BER_TRAIN_cv).mean()); EVA_BER_TEST.append(np.array(EVA_BER_TEST_cv).mean())
    EVA_KAPPA_TRAIN.append(np.array(EVA_KAPPA_TRAIN_cv).mean()); EVA_KAPPA_TEST.append(np.array(EVA_KAPPA_TEST_cv).mean())   
    EVA_TPR_TRAIN.append(np.array(EVA_TPR_TRAIN_cv).mean());EVA_TPR_TEST.append(np.array(EVA_TPR_TEST_cv).mean())       
    EVA_TNR_TRAIN.append(np.array(EVA_TNR_TRAIN_cv).mean());EVA_TNR_TEST.append(np.array(EVA_TNR_TEST_cv).mean())    
    CUT_OFF.append(np.array(CUT_OFF_cv).mean())    
    C_TRAIN={"TPR" : EVA_TPR_TRAIN,"TNR" : EVA_TNR_TRAIN,"BER" : EVA_BER_TRAIN,"ACC" : EVA_ACC_TRAIN,"MCC" : EVA_MCC_TRAIN,"F1_score" : EVA_F1_score_TRAIN,"AUC" : EVA_AUC_TRAIN,'KAPPA':EVA_KAPPA_TRAIN,'CUT_OFF':CUT_OFF}
    C_TEST={"TPR" : EVA_TPR_TEST,"TNR" : EVA_TNR_TEST,"BER" : EVA_BER_TEST,"ACC" : EVA_ACC_TEST,"MCC" : EVA_MCC_TEST,"F1_score" : EVA_F1_score_TEST,"AUC" : EVA_AUC_TEST,'KAPPA':EVA_KAPPA_TEST,'CUT_OFF':CUT_OFF}

######################################################################################
    return  np.array(C_TEST["CUT_OFF"]).mean(),C_TRAIN,C_TEST
####################################################################################
####################################################################################



def mul_pre_calss(pre_label,label_name,comtest_copy): 
    global x_train_for_vail, y_train_for_vail, x_train_for_vail_group
    if label_name=='label':
        pre_label_set=list(set(comtest_copy.label))
        for i in range(len(pre_label_set)):
            if i==pre_label:
                comtest_copy.label[comtest_copy.label==i]=10
            else:
                comtest_copy.label[comtest_copy.label==i]=-1
        comtest_copy.label[comtest_copy.label==10]=1;comtest_copy.label[comtest_copy.label==-1]=0   
    if label_name=='death_label':
        pre_label_set=list(set(comtest_copy.death_label))
        for i in range(len(pre_label_set)):
            if i==pre_label:
                comtest_copy.death_label[comtest_copy.death_label==i]=10
            else:
                comtest_copy.death_label[comtest_copy.death_label==i]=-1
        comtest_copy.death_label[comtest_copy.death_label==10]=1;comtest_copy.death_label[comtest_copy.death_label==-1]=0   
        
    patientunitstayid=list(set(comtest_copy['patientunitstayid']))
    #x_train_for_vail, x_test, y_train_for_vail, y_true = model_selection.train_test_split(comtest.iloc[0:len(comtest),0:comtest.shape[1]-1],comtest.iloc[0:len(comtest),-1], test_size = 0.2,random_state = 1)    
    x_train_for_vail, x_test, y_train_for_vail, y_true = model_selection.train_test_split(np.array(patientunitstayid),range(len(patientunitstayid)), test_size = 0.2,random_state = random_state)    
#    x_train_for_vail=spec_for_ser('comtest_copy',x_train_for_vail);
    
    str_df=str('comtest_copy')
    for i in x_train_for_vail:
        if i==x_train_for_vail[0]:
            input_mulit=(str_df+"["+str_df+"['patientunitstayid']=={}]".format(i))
        else:
            input_mulit=(input_mulit+","+str_df+"["+str_df+"['patientunitstayid']=={}]".format(i))
    x_train_for_vail=pd.concat(eval(input_mulit),axis=0,ignore_index=True)
      
    y_train_for_vail=x_train_for_vail[label_name];x_train_for_vail_group=x_train_for_vail.iloc[:,0];x_train_for_vail=x_train_for_vail.iloc[:,1:x_train_for_vail.shape[1]-1]
#    x_test=spec_for_ser('comtest_copy',x_test)
    
    str_df=str('comtest_copy')
    for i in x_test:
        if i==x_test[0]:
            input_mulit=(str_df+"["+str_df+"['patientunitstayid']=={}]".format(i))
        else:
            input_mulit=(input_mulit+","+str_df+"["+str_df+"['patientunitstayid']=={}]".format(i))
    x_test=pd.concat(eval(input_mulit),axis=0,ignore_index=True)
    
    y_true=x_test[label_name];x_test_group=x_test.iloc[:,0];x_test=x_test.iloc[:,1:x_test.shape[1]-1]
    
    y_train_for_vail=np.array(y_train_for_vail,int32);y_true=np.array(y_true,int32)
    
#    trials = Trials()
#    best_class = fmin(f_class, space_class, algo=tpe.suggest, max_evals= 5, trials=trials)
    
#    comm = lgb.LGBMClassifier(device_type='gpu')
    comm = xgb.XGBClassifier(tree_method='gpu_hist', gpu_id=1)
    comm.fit(x_train_for_vail ,y_train_for_vail)
    pro_comm_Pre = comm.predict_proba(x_test)
    cut_off,C_TRAIN,C_TEST=RUN(x_train_for_vail,y_train_for_vail,x_train_for_vail_group)
    blo_comm_Pre = blo(pro_comm_Pre,cut_off)
    eva_comm = evaluating_indicator(y_true=y_true, y_test=blo_comm_Pre, y_test_value=pro_comm_Pre)
    
    
    Corrected_probability_value=((pro_comm_Pre[:,1]-cut_off*0.001)/(1-cut_off*0.001))*0.5+0.5
    z=((cut_off*0.001-pro_comm_Pre[:,1])/(cut_off*0.001))*0.5
    Corrected_probability_value[Corrected_probability_value<0.5]=z[Corrected_probability_value<0.5]

    return eva_comm,Corrected_probability_value,C_TRAIN


def mul_pre_Regr(label_name,comtest_copy):
    global x_train_for_vail, y_train_for_vail, x_train_for_vail_group
    patientunitstayid=list(set(comtest_copy['patientunitstayid']))

#    x_train_for_vail, x_test, y_train_for_vail, y_true = model_selection.train_test_split(comtest.iloc[0:len(comtest),0:comtest.shape[1]-1],comtest.iloc[0:len(comtest),-1], test_size = 0.2,random_state = 1)    
    x_train_for_vail, x_test, y_train_for_vail, y_true = model_selection.train_test_split(np.array(patientunitstayid),range(len(patientunitstayid)), test_size = 0.2,random_state = random_state)    
    
    str_df=str('comtest_copy')
    for i in x_train_for_vail:
        if i==x_train_for_vail[0]:
            input_mulit=(str_df+"["+str_df+"['patientunitstayid']=={}]".format(i))
        else:
            input_mulit=(input_mulit+","+str_df+"["+str_df+"['patientunitstayid']=={}]".format(i))
    x_train_for_vail=pd.concat(eval(input_mulit),axis=0,ignore_index=True)
      
    y_train_for_vail=x_train_for_vail[label_name];x_train_for_vail_group=x_train_for_vail.iloc[:,0];x_train_for_vail=x_train_for_vail.iloc[:,1:x_train_for_vail.shape[1]-4]
#    x_test=spec_for_ser('comtest_copy',x_test)
    
    str_df=str('comtest_copy')
    for i in x_test:
        if i==x_test[0]:
            input_mulit=(str_df+"["+str_df+"['patientunitstayid']=={}]".format(i))
        else:
            input_mulit=(input_mulit+","+str_df+"["+str_df+"['patientunitstayid']=={}]".format(i))
    x_test=pd.concat(eval(input_mulit),axis=0,ignore_index=True)
    y_true=x_test[label_name];x_test_group=x_test.iloc[:,0];x_test=x_test.iloc[:,1:x_test.shape[1]-4]
    
#    trials = Trials()
#    best_Regr = fmin(f_Regressor, space_Regressor, algo=tpe.suggest, max_evals=5, trials=trials)
#    comm = lgb.LGBMRegressor(**best_Regr)
    comm = lgb.LGBMRegressor(device_type='gpu')
#    comm = ensemble.RandomForestRegressor()
#    comm = xgb.XGBRegressor(tree_method='gpu_hist')
    comm.fit(x_train_for_vail ,y_train_for_vail)
    pro_comm_Pre = comm.predict(x_test)

    result_NMI=metrics.normalized_mutual_info_score(y_true, pro_comm_Pre)
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(y_true, pro_comm_Pre)
    eva_comm={"RMSE" : (sum(np.array(pro_comm_Pre-y_true)**2)/len(y_true))**0.5,  #平均根误差
          "Mean_Absolute_Error":(sum(abs(y_true-pro_comm_Pre))/len(y_true)),
              "R_Squared" : r2_score(y_true, pro_comm_Pre) ,"MI":result_NMI,
          "Explained_Variance_Score" : 1-(np.var(y_true-pro_comm_Pre)/np.var(y_true)),
          "Median_Absolute_Error":np.median(abs(y_true-pro_comm_Pre))}    
    return eva_comm


#############################################################
#############################################################
#############################################################
#gap=4
learn_len=4 ;random_state=0
for gap in [12,8,4,1]:
#    com_hospitalid=pd.read_csv('/media/amms/D23CA3653CA34379/comtest_for_model_'+str(learn_len)+'_'+str(gap)+'_4_alljiegouhua.csv').dropna(axis=0,subset = ["label"]).hospitalid
#    
#    c['hospitalid']=cc.hospitalid; c['death_label']=cc.death_label
#    c['to_discharge']=cc.to_discharge; c['vent_duration']=cc.vent_duration
    
    
    com=pd.read_csv('/media/amms/D23CA3653CA34379/comtest_for_model_'+str(learn_len)+'_'+str(gap)+'_4_alljiegouhua.csv').dropna(axis=0,subset = ["label"])
    com_hospitalid=com.hospitalid
    death_label_pro=com.death_label
    to_discharge_pro=com.to_discharge
    vent_duration_pro=com.vent_duration
#    com.label.to_csv('/media/amms/80D89D09D89CFE9A/bert_model/label_pro.csv',index=False)
    del com;gc.collect()
    
    #############################################################
    #############################################################
    #############################################################

    for label_set in [0,1,2,3]:
        com = pd.read_csv('/media/amms/D23CA3653CA34379/comtest_for_model_'+str(learn_len)+'_'+str(gap)+'_4_alljiegouhua+textxianglianghua.csv').dropna(axis=0,subset = ["label"])
        com.patientunitstayid=com_hospitalid
        com.drop(columns=['hospitalid','death_label','to_discharge','vent_duration'],inplace=True)
        exec("eva_comm_gap"+str(gap)+"_"+str(label_set)+"_j_z,cpv_gap"+str(gap)+"_"+str(label_set)+"_j_z,C_TRAIN_gap"+str(gap)+"_"+str(label_set)+"_j_z=mul_pre_calss(pre_label=label_set,label_name='label',comtest_copy=com.dropna())")
    
#    com.label=pd.read_csv('/media/amms/80D89D09D89CFE9A/bert_model/label_pro.csv')
    com = pd.read_csv('/media/amms/D23CA3653CA34379/comtest_for_model_'+str(learn_len)+'_'+str(gap)+'_4_alljiegouhua+textxianglianghua.csv').dropna(axis=0,subset = ["label"])
    com.patientunitstayid=com_hospitalid
    com.drop(columns=['hospitalid','death_label','to_discharge','vent_duration'],inplace=True)
    w_0, w_1, w_2, w_3 = np.bincount(list(com.dropna().label)) / len(com.dropna().label)
#    del com;gc.collect()
    #print(eva_comm_0_j_z['AUC'],eva_comm_1_j_z['AUC'],eva_comm_2_j_z['AUC'],eva_comm_3_j_z['AUC'])
    
    exec("pre_label_acc"+str(gap)+"_j_z=w_0*eva_comm_gap"+str(gap)+"_0_j_z['ACC']+w_1*eva_comm_gap"+str(gap)+"_1_j_z['ACC']+w_2*eva_comm_gap"+str(gap)+"_2_j_z['ACC']+w_3*eva_comm_gap"+str(gap)+"_3_j_z['ACC']")
    exec("B_AUC"+str(gap)+"_j_z=w_0*eva_comm_gap"+str(gap)+"_0_j_z['AUC']+w_1*eva_comm_gap"+str(gap)+"_1_j_z['AUC']+w_2*eva_comm_gap"+str(gap)+"_2_j_z['AUC']+w_3*eva_comm_gap"+str(gap)+"_3_j_z['AUC']")
    
    
    exec("cpv_all = {0: cpv_gap"+str(gap)+"_0_j_z, 1: cpv_gap"+str(gap)+"_1_j_z, 2: cpv_gap"+str(gap)+"_2_j_z, 3: cpv_gap"+str(gap)+"_3_j_z }")
    cpv_all=pd.DataFrame(cpv_all)
    y_pred=[]
    for i in range(cpv_all.shape[0]):
        c_m=cpv_all.iloc[i,:].max()
        for j in range(cpv_all.shape[1]):
            if cpv_all.iloc[i,j]==c_m:
                y_pred.append(list(cpv_all)[j])
                
    #######################
#    com = pd.read_csv('/media/amms/D23CA3653CA34379/comtest_for_model_'+str(learn_len)+'_'+str(gap)+'_4_alljiegouhua+textxianglianghua.csv').dropna(axis=0,subset = ["label"])
#    com.patientunitstayid=com_hospitalid
#    com.drop(columns=['hospitalid','death_label','to_discharge','vent_duration'],inplace=True)
    patientunitstayid=list(set(com.dropna()['patientunitstayid']))
    
    x_train_for_vail, x_test, y_train_for_vail, y_true = model_selection.train_test_split(np.array(patientunitstayid),range(len(patientunitstayid)), test_size = 0.2,random_state = random_state)    
    str_df=str('com.dropna()')
    for i in x_test:
        if i==x_test[0]:
            input_mulit=(str_df+"["+str_df+"['patientunitstayid']=={}]".format(i))
        else:
            input_mulit=(input_mulit+","+str_df+"["+str_df+"['patientunitstayid']=={}]".format(i))
    x_test=pd.concat(eval(input_mulit),axis=0,ignore_index=True)
    y_true=x_test['label']
    
##############################################################################################
    com = pd.read_csv('/media/amms/D23CA3653CA34379/comtest_for_model_'+str(learn_len)+'_'+str(gap)+'_4_alljiegouhua+textxianglianghua.csv').dropna(axis=0,subset = ["label"])
    com.patientunitstayid=com_hospitalid
    com.drop(columns=['hospitalid','label','to_discharge','vent_duration'],inplace=True)
    exec("eva_death_label_gap"+str(gap)+"_j_z,cpv_gap"+str(gap)+"_j_z,C_death_label_TRAIN_gap"+str(gap)+"_j_z=mul_pre_calss(pre_label=1,label_name='death_label',comtest_copy=com.dropna())")
    
    com = pd.read_csv('/media/amms/D23CA3653CA34379/comtest_for_model_'+str(learn_len)+'_'+str(gap)+'_4_alljiegouhua+textxianglianghua.csv').dropna(axis=0,subset = ["label"])
    com.patientunitstayid=com_hospitalid
    com.drop(columns=['hospitalid','death_label','label','vent_duration'],inplace=True)
    exec("eva_to_discharge_gap"+str(gap)+"_j_z=mul_pre_Regr(label_name='to_discharge',comtest_copy=com.dropna())")
   
    com = pd.read_csv('/media/amms/D23CA3653CA34379/comtest_for_model_'+str(learn_len)+'_'+str(gap)+'_4_alljiegouhua+textxianglianghua.csv').dropna(axis=0,subset = ["label"])
    com.patientunitstayid=com_hospitalid
    com.drop(columns=['hospitalid','death_label','to_discharge','label'],inplace=True)
    exec("eva_vent_duration_gap"+str(gap)+"_j_z=mul_pre_Regr(label_name='vent_duration',comtest_copy=com.dropna())")
    
    del com;gc.collect()
    #######################
    exec("cm_gap"+str(gap)+"_j_z = ConfusionMatrix(actual_vector=np.array(list(y_true),int32), predict_vector=np.array(y_pred,int32))")
    
    #cm_1 = ConfusionMatrix(actual_vector=np.array(list(y_true),int32), predict_vector=np.array(y_pred,int32))
    #print(cm_1.overall_stat['ACC Macro'])
    #print(cm_1.overall_stat['Kappa'])
    #print(cm_1.overall_stat['F1 Micro'])
    #print(cm_1.overall_stat['Overall MCC'])
    
    
    ##############################################################################
    ##############################################################################h
    ##############################################################################
    ##############################################################################
#    com=pd.read_csv('/media/amms/80D89D09D89CFE9A/bert_model/comtest_for_model_'+str(learn_len)+'_'+str(gap)+'_4_all.csv').dropna(axis=0,subset = ["label"])
#    com_hospitalid=com.hospitalid_1
#    death_label_pro=com.death_label
#    to_discharge_pro=com.to_discharge
#    vent_duration_pro=com.vent_duration
#    com.label.to('/media/amms/80D89D09D89CFE9A/bert_model/label_pro.csv',index=False)
#    del com;gc.collect()
    
#    com = pd.read_csv('/media/amms/80D89D09D89CFE9A/bert_model/comtest_for_model_1_'+str(gap)+'_4_alljiegouhua+textxianglianghua.csv').dropna(axis=0,subset = ["label"])
    
    #############################################################
    #############################################################
    #############################################################

        
    for label_set in [0,1,2,3]:
        com = pd.read_csv('/media/amms/D23CA3653CA34379/comtest_for_model_'+str(learn_len)+'_'+str(gap)+'_4_alljiegouhua+textxianglianghua.csv').dropna(axis=0,subset = ["label"])
        com.patientunitstayid=com_hospitalid 
        com.drop(columns=['hospitalid','death_label','to_discharge','vent_duration'],inplace=True)
        com=pd.DataFrame(np.hstack((np.array(com.iloc[:,0]).reshape(-1,1),com.iloc[:,5+40*learn_len:com.shape[1]])) );index=com.columns
        com.rename(columns={index[0]:'patientunitstayid', index[-1]:'label'}, inplace = True)
        exec("eva_comm_gap"+str(gap)+"_z,cpv_gap"+str(gap)+"_"+"_z,C_TRAIN_gap"+str(gap)+"_"+"_z=mul_pre_calss(pre_label=label_set,label_name='label',comtest_copy=com.dropna())")
    
    com = pd.read_csv('/media/amms/D23CA3653CA34379/comtest_for_model_'+str(learn_len)+'_'+str(gap)+'_4_alljiegouhua+textxianglianghua.csv').dropna(axis=0,subset = ["label"])
    com.patientunitstayid=com_hospitalid 
    com.drop(columns=['hospitalid','death_label','to_discharge','vent_duration'],inplace=True)
    com=pd.DataFrame(np.hstack((np.array(com.iloc[:,0]).reshape(-1,1),com.iloc[:,5+40*learn_len:com.shape[1]])) );index=com.columns
    com.rename(columns={index[0]:'patientunitstayid', index[-1]:'label'}, inplace = True)
    w_0, w_1, w_2, w_3 = np.bincount(list(com.dropna().label)) / len(com.dropna().label)
    
    #print(eva_comm_0_z['AUC'],eva_comm_1_z['AUC'],eva_comm_2_z['AUC'],eva_comm_3_z['AUC'])
    exec("pre_label_acc"+str(gap)+"_z=w_0*eva_comm_gap"+str(gap)+"_0_z['ACC']+w_1*eva_comm_gap"+str(gap)+"_1_z['ACC']+w_2*eva_comm_gap"+str(gap)+"_2_z['ACC']+w_3*eva_comm_gap"+str(gap)+"_3_z['ACC']")
    exec("B_AUC"+str(gap)+"_z=w_0*eva_comm_gap"+str(gap)+"_0_z['AUC']+w_1*eva_comm_gap"+str(gap)+"_1_z['AUC']+w_2*eva_comm_gap"+str(gap)+"_2_z['AUC']+w_3*eva_comm_gap"+str(gap)+"_3_z['AUC']")
    
    #from pycm import *
    exec("cpv_all = {0: cpv_gap"+str(gap)+"_0_z, 1: cpv_gap"+str(gap)+"_1_z, 2: cpv_gap"+str(gap)+"_2_z, 3: cpv_gap"+str(gap)+"_3_z }")
    cpv_all=pd.DataFrame(cpv_all)
    y_pred=[]
    for i in range(cpv_all.shape[0]):
        c_m=cpv_all.iloc[i,:].max()
        for j in range(cpv_all.shape[1]):
            if cpv_all.iloc[i,j]==c_m:
                y_pred.append(list(cpv_all)[j])
    #######################
#    com = pd.read_csv('/media/amms/80D89D09D89CFE9A/bert_model/comtest_for_model_'+str(learn_len)+'_'+str(gap)+'_4_alljiegouhua+textxianglianghua.csv').dropna(axis=0,subset = ["label"])
#    com.patientunitstayid=com_hospitalid 
#    com.drop(columns=['hospitalid','death_label','to_discharge','vent_duration'],inplace=True)
#    com=pd.DataFrame(np.hstack((np.array(com.iloc[:,0]).reshape(-1,1),com.iloc[:,5+40*learn_len:com.shape[1]])) );index=com.columns
#    com.rename(columns={index[0]:'patientunitstayid', index[-1]:'label'}, inplace = True)
    patientunitstayid=list(set(com.dropna()['patientunitstayid']))
    
    x_train_for_vail, x_test, y_train_for_vail, y_true = model_selection.train_test_split(np.array(patientunitstayid),range(len(patientunitstayid)), test_size = 0.2,random_state = random_state)    
    str_df=str('com.dropna()')
    for i in x_test:
        if i==x_test[0]:
            input_mulit=(str_df+"["+str_df+"['patientunitstayid']=={}]".format(i))
        else:
            input_mulit=(input_mulit+","+str_df+"["+str_df+"['patientunitstayid']=={}]".format(i))
    x_test=pd.concat(eval(input_mulit),axis=0,ignore_index=True)
    y_true=x_test['label']
        
##############################################################################################
    com = pd.read_csv('/media/amms/D23CA3653CA34379/comtest_for_model_'+str(learn_len)+'_'+str(gap)+'_4_alljiegouhua+textxianglianghua.csv').dropna(axis=0,subset = ["label"])
    com.patientunitstayid=com_hospitalid 
    com.drop(columns=['hospitalid','label','to_discharge','vent_duration'],inplace=True)
    com=pd.DataFrame(np.hstack((np.array(com.iloc[:,0]).reshape(-1,1),com.iloc[:,5+40*learn_len:com.shape[1]])) );index=com.columns
    com.rename(columns={index[0]:'patientunitstayid', index[-1]:'death_label'}, inplace = True)
    exec("eva_death_label_gap"+str(gap)+"_z,cpv_gap"+str(gap)+"_z,C_death_label_TRAIN_gap"+str(gap)+"_"+str(label_set)+"_z=mul_pre_calss(pre_label=1,label_name='death_label',comtest_copy=com.dropna())")
    
    com = pd.read_csv('/media/amms/D23CA3653CA34379/comtest_for_model_'+str(learn_len)+'_'+str(gap)+'_4_alljiegouhua+textxianglianghua.csv').dropna(axis=0,subset = ["label"])
    com.patientunitstayid=com_hospitalid 
    com.drop(columns=['hospitalid','death_label','label','vent_duration'],inplace=True)
    com=pd.DataFrame(np.hstack((np.array(com.iloc[:,0]).reshape(-1,1),com.iloc[:,5+40*learn_len:com.shape[1]])) );index=com.columns
    com.rename(columns={index[0]:'patientunitstayid', index[-1]:'to_discharge'}, inplace = True)
    exec("eva_to_discharge_gap"+str(gap)+"_z=mul_pre_Regr(label_name='to_discharge',comtest_copy=com.dropna())")
   
    com = pd.read_csv('/media/amms/D23CA3653CA34379/comtest_for_model_'+str(learn_len)+'_'+str(gap)+'_4_alljiegouhua+textxianglianghua.csv').dropna(axis=0,subset = ["label"])
    com.patientunitstayid=com_hospitalid 
    com.drop(columns=['hospitalid','death_label','to_discharge','label'],inplace=True)
    com=pd.DataFrame(np.hstack((np.array(com.iloc[:,0]).reshape(-1,1),com.iloc[:,5+40*learn_len:com.shape[1]])) );index=com.columns
    com.rename(columns={index[0]:'patientunitstayid', index[-1]:'vent_duration'}, inplace = True)
    exec("eva_vent_duration_gap"+str(gap)+"_z=mul_pre_Regr(label_name='vent_duration',comtest_copy=com.dropna())")
    
    del com;gc.collect()
    #######################
    exec("cm_gap"+str(gap)+"_z = ConfusionMatrix(actual_vector=np.array(list(y_true),int32), predict_vector=np.array(y_pred,int32))")
    #cm_2 = ConfusionMatrix(actual_vector=np.array(list(y_true),int32), predict_vector=np.array(y_pred,int32))
    #print(cm_2.overall_stat['ACC Macro'])
    #print(cm_2.overall_stat['Kappa'])
    #print(cm_2.overall_stat['F1 Micro'])
    #print(cm_2.overall_stat['Overall MCC'])
    
    
    
    
    ##############################################################################
    ##############################################################################
    ##############################################################################
    ##############################################################################
    #comtest_copy = pd.read_csv('/media/amms/80D89D09D89CFE9A/bert_model/comtest_for_model_1_12_4_alljiegouhua+textxianglianghua.csv',nrows=10000).dropna()
    #com=pd.DataFrame() ;index=c.columns
    #com.rename(columns={index[0]:'patientunitstayid', index[-1]:'label'}, inplace = True)


    for label_set in [0,1,2,3]:
        com = pd.read_csv('/media/amms/D23CA3653CA34379/comtest_for_model_'+str(learn_len)+'_'+str(gap)+'_4_alljiegouhua.csv').dropna(axis=0,subset = ["label"])
        com.patientunitstayid=com_hospitalid 
        com.drop(columns=['hospitalid','death_label','to_discharge','vent_duration'],inplace=True)
        exec("eva_comm_gap"+str(gap)+"_"+str(label_set)+"_j,cpv_gap"+str(gap)+"_"+str(label_set)+"_j,C_TRAIN_gap"+str(gap)+"_"+str(label_set)+"_j=mul_pre_calss(pre_label=label_set,label_name='label',comtest_copy=com.dropna())")
    
    com = pd.read_csv('/media/amms/D23CA3653CA34379/comtest_for_model_'+str(learn_len)+'_'+str(gap)+'_4_alljiegouhua.csv').dropna(axis=0,subset = ["label"])
    com.patientunitstayid=com_hospitalid 
    com.drop(columns=['hospitalid','death_label','to_discharge','vent_duration'],inplace=True)
    w_0, w_1, w_2, w_3 = np.bincount(list(com.dropna().label)) / len(com.dropna().label)
    exec("pre_label_acc"+str(gap)+"_j=w_0*eva_comm_gap"+str(gap)+"_0_j['ACC']+w_1*eva_comm_gap"+str(gap)+"_1_j['ACC']+w_2*eva_comm_gap"+str(gap)+"_2_j['ACC']+w_3*eva_comm_gap"+str(gap)+"_3_j['ACC']")
    exec("B_AUC"+str(gap)+"_j=w_0*eva_comm_gap"+str(gap)+"_0_j['AUC']+w_1*eva_comm_gap"+str(gap)+"_1_j['AUC']+w_2*eva_comm_gap"+str(gap)+"_2_j['AUC']+w_3*eva_comm_gap"+str(gap)+"_3_j['AUC']")
    
    #from pycm import *
    exec("cpv_all = {0: cpv_gap"+str(gap)+"_0_j, 1: cpv_gap"+str(gap)+"_1_j, 2: cpv_gap"+str(gap)+"_2_j, 3: cpv_gap"+str(gap)+"_3_j }")
    cpv_all=pd.DataFrame(cpv_all)
    y_pred=[]
    for i in range(cpv_all.shape[0]):
        c_m=cpv_all.iloc[i,:].max()
        for j in range(cpv_all.shape[1]):
            if cpv_all.iloc[i,j]==c_m:
                y_pred.append(list(cpv_all)[j])
    #######################
#    com = pd.read_csv('/media/amms/80D89D09D89CFE9A/bert_model/comtest_for_model_'+str(learn_len)+'_'+str(gap)+'_4_alljiegouhua.csv').dropna(axis=0,subset = ["label"])
#    com.patientunitstayid=com_hospitalid 
    patientunitstayid=list(set(com.dropna()['patientunitstayid']))
    x_train_for_vail, x_test, y_train_for_vail, y_true = model_selection.train_test_split(np.array(patientunitstayid),range(len(patientunitstayid)), test_size = 0.2,random_state = random_state)    
    str_df=str('com.dropna()')
    for i in x_test:
        if i==x_test[0]:
            input_mulit=(str_df+"["+str_df+"['patientunitstayid']=={}]".format(i))
        else:
            input_mulit=(input_mulit+","+str_df+"["+str_df+"['patientunitstayid']=={}]".format(i))
    x_test=pd.concat(eval(input_mulit),axis=0,ignore_index=True)
    y_true=x_test['label']
        
##############################################################################################
    com = pd.read_csv('/media/amms/D23CA3653CA34379/comtest_for_model_'+str(learn_len)+'_'+str(gap)+'_4_alljiegouhua.csv').dropna(axis=0,subset = ["label"])
    com.patientunitstayid=com_hospitalid 
    com.drop(columns=['hospitalid','label','to_discharge','vent_duration'],inplace=True)
    exec("eva_death_label_gap"+str(gap)+"_j,cpv_gap"+str(gap)+"_j,C_death_label_TRAIN_gap"+str(gap)+"_j=mul_pre_calss(pre_label=1,label_name='death_label',comtest_copy=com.dropna())")
    
    com = pd.read_csv('/media/amms/D23CA3653CA34379/comtest_for_model_'+str(learn_len)+'_'+str(gap)+'_4_alljiegouhua.csv').dropna(axis=0,subset = ["label"])
    com.patientunitstayid=com_hospitalid 
    com.drop(columns=['hospitalid','death_label','label','vent_duration'],inplace=True)
    exec("eva_to_discharge_gap"+str(gap)+"_j=mul_pre_Regr(label_name='to_discharge',comtest_copy=com.dropna())")
   
    com = pd.read_csv('/media/amms/D23CA3653CA34379/comtest_for_model_'+str(learn_len)+'_'+str(gap)+'_4_alljiegouhua.csv').dropna(axis=0,subset = ["label"])
    com.patientunitstayid=com_hospitalid 
    com.drop(columns=['hospitalid','death_label','to_discharge','label'],inplace=True)
    exec("eva_vent_duration_gap"+str(gap)+"_j=mul_pre_Regr(label_name='vent_duration',comtest_copy=com.dropna())")
    
    del com;gc.collect()
    #######################
    exec("cm_gap"+str(gap)+"_j = ConfusionMatrix(actual_vector=np.array(list(y_true),int32), predict_vector=np.array(y_pred,int32))")
    #print(cm_3.overall_stat['ACC Macro'])
    #print(cm_3.overall_stat['Kappa'])
    #print(cm_3.overall_stat['F1 Micro'])
    #print(cm_3.overall_stat['Overall MCC'])
    
    


