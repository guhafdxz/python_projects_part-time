import numpy as np
import pandas as pd
import seaborn as sns
import os
import sys
import math
from math import e
from matplotlib import pyplot as plt
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.svm import NaiveSurvivalSVM,FastSurvivalSVM,FastKernelSurvivalSVM
from sksurv.metrics import concordance_index_censored
from scipy.stats.stats import pearsonr

#β value
β_cox_event=np.array([2,-1.6,1.2,-0.8,0.4])   #Cox Event 
β_aft_censor=np.array([1,1,1,1,1])  #AFT censor
β_cox_censor=np.array([1,1,1,-2,-2])  #COX censor     


class SVHM():
    def __init__(self,n_sample=100,n_noise=0,censor_ratio=0.4):
        self.sample=n_sample
        self.noise=n_noise
        self.censor_ratio=censor_ratio

    
    def generate_event_time(self):  #dealth event observation time
        """
        Returns
        -------
        event-time 

        """
        corr=np.empty((5,5))     #correlation matrix
        for i in range(5):
          for j in range(5):
              corr[i,j]=0.5**abs(i-j)  #ρ=0.5 (Z_j ,Z_k ) = ρ^|j−k| , 
        cov=0.5*0.5*corr      # covariation matrix
        mean=np.zeros(5).T
        self.covariates=np.random.multivariate_normal(mean, cov, self.sample) #协方矩阵生成多元正态分布，边缘分布满足正态~N(0,0.5^2 )
        F=np.random.uniform(size=self.sample)
        Z=self.covariates
        S = 1 - F
        log_S = np.log(S)
        self.log_S= log_S
        Beta_Z=np.sum(β_cox_event.T*Z).sum()
        event_time = np.log(-0.25*log_S/(np.exp(Beta_Z))+1)/ (-0.25)   #指数分布 λ = 0.25-累计危险函数
        event_time=abs(np.ceil(event_time))
        self.event_time=event_time #取整
        return event_time
    def censor_time_aft(self):
        
        censor_aft=[]
        censor_time_aft=np.empty(self.sample)
        c_ratio_aft=0
        Z_Beta_C1 = np.sum(β_aft_censor.T*self.covariates).sum()
        for a in np.arange(-Z_Beta_C1+1,-Z_Beta_C1+5,0.01):  #make trials to get desired censor ratio
            for iteration in range(2000):
                    state=[]  
                     #AFT模型下censor_time满足对数正态分布
                    censor_time_aft=np.exp(np.random.normal( Z_Beta_C1+a,0.5,self.sample))
                    censor_time_aft=np.ceil(censor_time_aft)
                    for i in range(len(censor_time_aft)):
                        if censor_time_aft[i]<= abs(self.event_time[i]):  # study ended before practical event time
                            state.append(True)
                        else:
                            state.append(False)
                    c_ratio_aft=np.sum(state)/self.sample
            if c_ratio_aft==self.censor_ratio:
                censor_aft=state
                break
        return censor_time_aft,censor_aft
    
    def censor_time_cox(self):
        censor_cox=[]
        censor_time_cox=np.empty(self.sample)
        c_ratio_cox=0
        beta_z=np.dot(self.covariates,β_cox_censor.T).sum()
        for delta in np.arange(0,2,0.01): #make trials to get desired censor ratio
            for iteration in range(3000):
                censor_state=[]  
                b=np.exp(beta_z+delta)
                censor_time_cox =pow(-6*self.log_S/b*np.exp(beta_z),3)
                # censor_time_cox=np.random.uniform(0,b,size=(self.sample,))*np.exp(beta_z)
                censor_time_cox=np.ceil(censor_time_cox)
                for i in range(len(censor_time_cox)):
                     if censor_time_cox[i]<=abs(self.event_time[i]):
                         censor_state.append(True)
                     else:
                         censor_state.append(False)
                c_ratio_cox=np.sum(censor_state)/self.sample
            if c_ratio_cox==self.censor_ratio:
                censor_cox=censor_state
                break
        return censor_time_cox,censor_cox      
        
    def generate_data(self,censor_time,censor_state):
         self.train_data=pd.DataFrame(self.covariates,columns=['z1','z2','z3','z4','z5'])
         self.train_data['event_time']=self.event_time
         self.train_data['censor_time']=censor_time
         self.train_data['censor_state']=censor_state
         u=np.percentile(self.event_time,90)  # greater than 90 percentile of event time truncated
         self.train_data_trunc=self.train_data[self.train_data['event_time']<=u]
         self.train_data_trunc=self.train_data_trunc[self.train_data_trunc['censor_time']<=u]
  
    
    def add_noise(self,censor_time,censor_state):
        self.generate_data(censor_time,censor_state)
        noise_variable_df=pd.DataFrame(np.random.normal(0,0.5,size=(self.sample,self.noise)),columns=["noise{}".format(i) for i in range(self.noise)])
        combined_train_data=pd.concat([self.train_data,noise_variable_df],axis=1)
        del combined_train_data['censor_time']
        return combined_train_data      

    
    def train_data_split(self,data):
        data=data.copy()
        train_y=data.loc[:,"event_time":"censor_state"]
        tuple_array=[(bool(train_y.iloc[i,1]),train_y.iloc[i,0]) for i in range(train_y.shape[0])]
        train_y=np.array(tuple_array,dtype=[('event_indicator',bool),('event_time','float64')]) 
        del data['event_time']
        del data['censor_state']
        train_X=data     
        return train_X,train_y
        
     
    def ipcw_cox(self,train_X,train_y,test_X,test_y):
        model=CoxPHSurvivalAnalysis(alpha=0, ties='breslow', n_iter=100, tol=1e-09, verbose=0)
        model.fit(train_X,train_y)
        pred_risk_score_test=model.predict(test_X)
        pred_risk_score_train=model.predict(train_X)
        return pred_risk_score_test,pred_risk_score_train
    
 
    def naive_survival_svm(self,train_X,train_y,test_X,test_y):
        model=NaiveSurvivalSVM(penalty='l2', loss='squared_hinge', dual=False, tol=0.0001, alpha=1.0, verbose=0, random_state=None, max_iter=1000)
        model.fit(train_X,train_y)
        pred_rank_test=model.predict(test_X)
        pred_rank_train=model.predict(train_X)
        cindex_test= concordance_index_censored(
            test_y['event_indicator'],
            test_y['event_time'],
            -pred_rank_test,  # flip sign to obtain risk scores
            )
        cindex_train= concordance_index_censored(
            train_y['event_indicator'],
            train_y['event_time'],
            -pred_rank_train,  # flip sign to obtain risk scores
            )
        pred_risk_score_test=-pred_rank_test
        pred_risk_score_train=-pred_rank_train
        return pred_risk_score_test,pred_risk_score_train,cindex_test,cindex_train
   
    def fast_kernel_survival_svm(self,train_X,train_y,test_X,test_y):
        #kernel ("linear" | "poly" | "rbf" | "sigmoid" | "cosine" | "precomputed") – Kernel. Default: “linear”
        model=FastKernelSurvivalSVM(alpha=1, rank_ratio=1.0, fit_intercept=False, kernel='linear', gamma=None, degree=3, coef0=1, kernel_params=None, max_iter=20, verbose=False, tol=None, optimizer='avltree', random_state=None, timeit=False)
        model.fit(train_X,train_y)
        pred_rank_test=model.predict(test_X)
        pred_rank_train=model.predict(train_X)
        cindex_test= concordance_index_censored(
            test_y['event_indicator'],
            test_y['event_time'],
            -pred_rank_test,  # flip sign to obtain risk scores
            )
        cindex_train= concordance_index_censored(
            train_y['event_indicator'],
            train_y['event_time'],
            -pred_rank_train,  # flip sign to obtain risk scores
            )
        pred_risk_score_test=-pred_rank_test
        pred_risk_score_train=-pred_rank_train
        return pred_risk_score_test,pred_risk_score_train,cindex_test,cindex_train

def pred_time_risk_score(pred_risk_score_train,pred_risk_score_test,train_y,k_n=1):
   """
      Predicting event time of a future subject by k -nearest-neighbor
      Parameters
       ----------
       pred_risk_score_train : ndarray-float64  shape = (n_samples,))
            Risk score predication of train datasets 
       pred_risk_score_test : ndarray-float64   shape = (n_samples,))
           Risk score predication of test datasets
       train_y : structural ndarray  shape = (n_samples,))
           A structured array containing the binary event indicator as first field, and time of event or time of censoring as second field.
       k_n : TYPE, optional
          number of neighbors selected to predict the event time
       Returns
       -------
       list with the data of predication of event-time 
   """       
   pred_risk_score_train_non_censor=np.array([pred_risk_score_train[i] for i in  range(len(pred_risk_score_train)) if train_y[i]['event_indicator']==True]) # Compute the risk scores for non-censored subjects in the training data
   event_time_non_censor=[train_y[i]['event_time'] for i in  range(len(pred_risk_score_train)) if train_y[i]['event_indicator']==True]   
   # event_time_non_censor=test_y['event_time']
   pred_event_time=[]
   for risk_score in pred_risk_score_test:     
       distance=pred_risk_score_train_non_censor-risk_score  #compute the absolute diff value between  test and  train
       distance_sorted=np.sort(abs(distance))  
       k_close_dis=distance_sorted[:k_n]  #find k non-censored subjects in training data of which risk score close to
       k_neighbor_risk_score=[pred_risk_score_train_non_censor[i] for i in range(len(distance)) if abs(distance[i]) in  k_close_dis]
       
       sorted_event_time_=sorted(event_time_non_censor,reverse=False) #sort event time of all non-censored subjects in ascending order
       rank_sort=np.argsort(pred_risk_score_train_non_censor, axis=-1, kind='quicksort', order=None)  # get rank of the event time of all non-censored subjects of training datasets in descending order
       k_neighbor_rank=[rank_sort[index] for index in range(len(rank_sort)) if pred_risk_score_train_non_censor[index] in k_neighbor_risk_score]  #get the rank of k closest neighbors' event time  in non-censored subjects of training datasets
       pred_time=np.mean([sorted_event_time_[rank] for rank in k_neighbor_rank ])    #predict the event time as average of event-time of k neighbors
       pred_event_time.append(pred_time)  #aggregate all the predict risk score of testing subejects 
      
   return pred_event_time 
 
def rmse(pred_event_time,event_time): #compute the rmse
    
    return np.sqrt(np.sum(pow(pred_event_time-event_time,2))/ len(event_time))
    
   

if __name__ == '__main__':
    
    sample=200  #200
    subjects=10000
    noise=5  #0/5/15/95
    censor_ratio=0.6 #0.6
    svhm= SVHM(sample,noise,censor_ratio) 
    censor_aft=[]
    censor_cox=[]
    censor_time_aft=np.empty(sample,)
    censor_time_cox=np.empty(sample,)
    while len(censor_aft)==0 or len(censor_cox)==0:    # fail to  obtain the desired censoring ratio, try again
       event_time=svhm.generate_event_time()  
       if  0 not in event_time:  # assume 0 not exist in event-time array
         censor_time_aft,censor_aft=svhm.censor_time_aft()
         censor_time_cox,censor_cox=svhm.censor_time_cox()
         
    data_aft_add_noise=svhm.add_noise(censor_time_aft,censor_aft)
    data_cox_add_noise=svhm.add_noise(censor_time_cox,censor_cox)
    #train_x and train_y data generation
    train_x_aft, train_y_aft=svhm.train_data_split(data_aft_add_noise)
    train_x_cox,train_y_cox=svhm.train_data_split(data_cox_add_noise)
    
    #Randomly generate the testing datasets
    test_svhm=SVHM(subjects,noise,0)
    test_event_time=test_svhm.generate_event_time()
    while np.percentile(test_event_time,90)>=np.percentile(svhm.event_time,90) or 0 in test_event_time:
        test_svhm=SVHM(subjects,noise,0)
        test_event_time=test_svhm.generate_event_time()
    test_data=pd.DataFrame(test_svhm.covariates,columns=['z1','z2','z3','z4','z5'])
    test_data['event_time']=test_event_time
    test_data['censor_state']=np.ones(subjects)
    test_data['censor_state']=np.array([bool(i) for i in test_data['censor_state']])
    u=np.percentile(svhm.event_time,90)
    test_data=test_data[test_data['event_time']<=u]
    test_data.index=np.arange(len(test_data))
    test_noise_df=pd.DataFrame(np.random.normal(0,0.5,size=(len(test_data),noise)),columns=["noise{}".format(i) for i in range(noise)])
    combined_test_data=pd.concat([test_data, test_noise_df],axis=1)
    test_X,test_y=test_svhm.train_data_split(combined_test_data)
    
    #IPCW-Cox 
    pred_risk_score_aft_test, pred_risk_score_aft_train=svhm.ipcw_cox(train_x_aft,train_y_aft,test_X,test_y)
    pred_risk_score_cox_test, pred_risk_score_cox_train=svhm.ipcw_cox(train_x_cox,train_y_cox,test_X,test_y)
    
    # linear Survival SVM
    pred_rank_aft_test,pred_rank_aft_train,_,_=svhm.naive_survival_svm(train_x_aft,train_y_aft,test_X,test_y)
    pred_rank_cox_test,pred_rank_cox_train,_,_=svhm.naive_survival_svm(train_x_cox,train_y_cox,test_X,test_y)
    
    
    # Survival SVM with  kernel
    pred_rank_aft_test_fast,pred_rank_aft_train_fast,_,_=svhm.fast_kernel_survival_svm(train_x_aft,train_y_aft,test_X,test_y)
    pred_rank_cox_test_fast,pred_rank_cox_train_fast,_,_=svhm.fast_kernel_survival_svm(train_x_cox,train_y_cox,test_X,test_y)
    
    
    #predict time based on sorting of risk score
    
    pred_aft_ipcw_cox=pred_time_risk_score(pred_risk_score_aft_train,pred_risk_score_aft_test,train_y_aft,k_n=3)
    pred_cox_ipcw_cox=pred_time_risk_score(pred_risk_score_cox_train, pred_risk_score_cox_test,train_y_cox,k_n=3)
    pred_aft_linear_svhm=pred_time_risk_score(pred_rank_aft_train, pred_rank_aft_test,train_y_aft,k_n=3)
    pred_cox_linear_svhm=pred_time_risk_score(pred_rank_cox_train,pred_rank_cox_test,train_y_cox,k_n=3)
    pred_aft_svhm_fast=pred_time_risk_score( pred_rank_aft_train_fast,pred_rank_aft_test_fast,train_y_aft,k_n=3)
    pred_cox_svhm_fast=pred_time_risk_score( pred_rank_cox_train_fast,pred_rank_cox_test_fast,train_y_cox,k_n=3)
    
    
    
    data_table=pd.DataFrame({"predict_time(IPCW-Cox,AFT censored)":pred_aft_ipcw_cox,
                            "predict_time(IPCW-Cox,COX censored)": pred_cox_ipcw_cox,
                            "predict_time(SVHM,AFT censored)":pred_aft_linear_svhm,
                            "predict_time(SVHM,COX censored)":pred_cox_linear_svhm,
                            "predict_time(Fast_Kernel SVHM, AFT censored)":pred_aft_svhm_fast,
                            "predict_time(Fast_Kernel SVHM, COX censored)":pred_cox_svhm_fast,
                            "True_time":test_y['event_time']
                            }
                            )
    
    #Compute Pearsonr、RMSE、SD
    
    "Pearson index--p value"
    corr_aft_ipcw_cox=round(pearsonr(test_y['event_time'],pred_aft_ipcw_cox)[1],2)
    corr_cox_ipcw_cox=round(pearsonr(test_y['event_time'], pred_cox_ipcw_cox)[1],2)
    
    corr_aft_linear_svhm=round(pearsonr(test_y['event_time'],pred_aft_linear_svhm)[1],2)
    corr_cox_linear_svhm=round(pearsonr(test_y['event_time'],pred_cox_linear_svhm)[1],2)
    
    corr_aft_fast_svhm=round(pearsonr(test_y['event_time'],pred_aft_svhm_fast)[1],2)
    corr_cox_fast_svhm=round(pearsonr(test_y['event_time'],pred_cox_svhm_fast)[1],2)
    
    "RMSE"
    
    rmse_aft_ipcw_cox=rmse( pred_aft_ipcw_cox,test_y['event_time'])
    rmse_cox_ipcw_cox=rmse(  pred_cox_ipcw_cox,test_y['event_time'])
    
    rmse_aft_linear_svhm=rmse( pred_aft_linear_svhm,test_y['event_time'])
    rmse_cox_linear_svhm=rmse( pred_cox_linear_svhm,test_y['event_time'])
    
    rmse_aft_fast_svhm=rmse( pred_aft_svhm_fast,test_y['event_time'])
    rmse_cox_fast_svhn=rmse( pred_cox_svhm_fast,test_y['event_time'])
    
     
  
    #Plot the fitting graph bwtween predict  time and pracical time for four models
   
    data_table=pd.read_excel('./data_table.xlsx')
   
    cols=data_table.columns[:6]
    sns.set_style('whitegrid')
    plt.figure()
    fig,ax=plt.subplots(3,2,figsize=(60,30),)
    for i,col in enumerate(cols): 
       plt.subplot(3,2,i+1)  
       plt.tick_params(labelsize=30) 
       g=sns.regplot(data=data_table.iloc[:100,:],y="True_time",x=col,x_estimator=np.mean,
                        line_kws={'linewidth':3,'color':'r'},
                                  scatter_kws={'color':'b','s':30},
                                  fit_reg=True
                                  )
       g.set_title("Fitting Curve between true time and {}".format(col),fontsize=40)
       g.legend(['Fitting_time'])
    plt.show()
    
    
    
    
  
    g=sns.regplot(data=data_table.iloc[:100,:],y="True_time",x=cols[4],x_estimator=np.mean,
                     line_kws={'linewidth':3,'color':'r'},
                               scatter_kws={'color':'b','s':30},
                               fit_reg=True
                               )
    
    g.set_title("Fitting Curve between true time and {}".format(cols[4]),fontsize=10)  
    g.legend(['Predict_time'])
    
    
    
   
    
    
    
    
    
    
    
    
    
    
    
    
    