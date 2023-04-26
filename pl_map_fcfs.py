# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 23:01:28 2022

"""
import numpy as np
import math
import pandas as pd
from matplotlib import pyplot as plt


LS='ls' # denote the LS customer
PS='ps' # denote the PS customer
ksi_0=0.1   
p_min=3    # lowest price
p_max=10   # largest price
L_min=3    # lowest leadtime
L_max=10   # largest leadtime
beta=0.4   # order completion probability
gamma=0.5  # arrival probability of prospective customers
theta=0.1  # no-arrival and no completion probability
K=5         # largest number of  orders can be processed for system 
kappa=[0.2,0.6,1]  #contrast parameter
zeta=[0.05,0.5,0.95]  #proportion of LS customer arrivals
ind=[0,1,LS,PS]  # state index
penalty_pair=[(6,0),(0,2),(3,1)] #tardiness penalty structure: fixed only/unit varied only/ fixed and unit varied


def get_action_space(s):    # makes (p,L) to prospecitve PS/LS customers
    action_space=list()
    ind=s[0]
    # n=s[1]
    if ind in [LS,PS]:
        p=np.arange(3,11)
        L=np.arange(3,11)
        for i in range(len(p)):
            action_space+=list(zip([p[i]]*len(L),L))
    return action_space 

def compute_prob_decision(ind,k,quote):  #the probability of acceptance of a quote(p,L) for LS/PS customer 
    ksi_l_ls=ksi_L+k*ksi_L  #LS customer
    ksi_p_ls=ksi_p-k*ksi_p 
    
    ksi_l_ps=ksi_L-k*ksi_L  #PS cunstomer
    ksi_p_ps=ksi_p+k*ksi_p
    
    p=quote[0]      #price
    L=quote[1]      #leadtime
    if ind in [0,1]:
        return
    elif ind==LS:
        return 1./(1+ksi_0*(math.e**( ksi_l_ls*(L-L_min)+ksi_p_ls*(p-p_min))))
    else:
        return 1./(1+ksi_0*(math.e**( ksi_l_ps*(L-L_min)+ksi_p_ps*(p-p_min))))

def compute_total_cost(enter_epoch,leave_epoch,pair,quote): #compute the tcost fo an order
      remain_epoch=quote[1]-(leave_epoch-enter_epoch)  # FCFS manufacture pattern
      if remain_epoch<=0:  
        tcost=pair[0]+pair[1]*abs(remain_epoch)
      else:   # no cost when process finished before leadtime
        tcost=0
      return tcost
       

def transfer(s,quote,pair,k,zeta):   # the transition between state
    p=compute_prob_decision(s[0],k,quote) 
    n=s[1]  # current number of orders in the manufacturing system
    price=quote[0] #Leadtime
    #cost=pair[0]+pair[1]  # tardiness cost when an order is late by one period
    # cost=compute_total_cost(s,pair,initial_order=4)
    if s[0] in [0,1] and n<K and n!=0:
        trans_prob=[theta,beta,gamma*zeta,gamma*(1-zeta)]   #transition probability
        next_prob_state=[(0,n),(1,n-1),(LS,n),(PS,n)]   #next possible state
        #profit=[0,-cost,0,0]
        revenue=[0,0,0,0]
    if s[0] in [0,1] and n==0:
        trans_prob=[1-gamma,gamma*zeta,gamma*(1-zeta)] 
        next_prob_state=[(0,0),(LS,0),(PS,0)]
        #profit=[0,0,0]
        revenue=[0,0,0]
    if s[0] in [LS,PS] and n<K and n!=0: #acceptance of the quote and not acceptance
        trans_prob= [p*theta,p*beta,p*gamma*zeta,p*gamma*(1-zeta)]+[(1-p)*theta,(1-p)*beta,(1-p)*gamma*zeta,(1-p)*gamma*(1-zeta)]
        next_prob_state=[(0,n+1),(1,n),(LS,n+1),(PS,n+1)]+[(0,n),(1,n-1),(LS,n),(PS,n)]
        #profit=[p*price,p*(price-cost),p*price,p*price]+[0,(1-p)*(-cost),0,0]
        revenue=[p*price,p*price,p*price,p*price]+[0,0,0,0]
    if s[0] in [LS,PS] and  n==0:  #acceptance of the quote and not acceptance
        trans_prob=[p*theta,p*beta,p*gamma*zeta,p*gamma*(1-zeta)]+[(1-p)*(1-gamma),(1-p)*gamma*zeta,(1-p)*gamma*(1-zeta)]
        next_prob_state=[(0,1),(1,0),(LS,1),(PS,1)]+[(0,0),(LS,0),(PS,0)]
        #profit=[p*price,p*(price-cost),p*price,p*price]+[0,0,0]
        revenue=[p*price,p*price,p*price,p*price]+[0,0,0]
    if n==K and s[0]!=0: # full system with new order is impossible, orders are rejected
        trans_prob=[1]
        next_prob_state=[(0,K)]
        #profit=[0]
        revenue=[0]
    if n==K and s[0]==0:
        trans_prob=[1-beta,beta]    
        next_prob_state=[(0,K),(1,K-1)] 
        #profit=[0,-cost]  
        revenue=[0,0]                      
    return trans_prob,next_prob_state,revenue


def step(s,quote,pair,k,z): # each epoch 
    transfer_probability,next_prob_state,revenue=transfer(s,quote,pair,k,z) #from current state to next state
    num=np.random.choice(range(len(next_prob_state)),size=1,p=transfer_probability) # sampling according the probability
    next_state=next_prob_state[num[0]] #next state
    revenue_transfer=revenue[num[0]] # transition income
    new_order=0
    if (s[0] in ['ps','ls'] and next_state[0]==1 and next_state[1]==s[1]) or (next_state[1]==s[1]+1):
           new_order=s[0]  # record new add order
    # profit_transfer=profit[num[0]]
    return next_state,revenue_transfer,new_order

def train(s,quote,pair,k,z,initial_order):
    Revenue=[]
    order_track=[]
    complete_epoch=[]
    accept_order=[]
    order_track.append((s,0)) #initialization
    cumsum=None
    for epoch in range(1,2000):  #iteration 2000-20000 
        try:
             next_state, revenue_value,new_order=step(s,quote,pair,k,z)
             order_track.append((next_state,epoch)) # order the info every step  training
             if new_order!=0:
                 accept_order.append((new_order,epoch-1)) # the new order was input in the last epoch
             if next_state[0]==1:
                complete_epoch.append((next_state,epoch)) # the ind=1 means finished order
             Revenue.append(revenue_value)
             s=next_state  # state iteration
             cumsum=np.sum(Revenue)  # cumulative of revenue
        except Exception as e:
            print(e)
            
    return Revenue,order_track,complete_epoch[initial_order:],cumsum,accept_order


kappa_zeta=[]
for i in range(len(zeta)):
     kappa_zeta+=list(zip([kappa[i]]*len(zeta),zeta))  #9组变量组合(κ,ζ)
'''
超参数设置
'''

initial_order=4 #   0/1/2/3/4
s=(PS,initial_order)  #s=(PS,initial_order)
pair=penalty_pair[0]  #penalty_pair[1]--(0,2),penalty_pair[2]---(3,1)
ksi_L=0.75 #Case1:0.75 Case2:1.5 Case3:0.75 Case4:1.5
ksi_p=0.75 #Case1:0.75 Case2:0.75 Case3:1.5 Case4:1.5

'''
V 记录 9组变量(κ,ζ)，不同(q,l)策略进入生产系统FCFS订单进入到完成交单
订单收入：Revenu
生产线订单完成时间：complete_epoch
订单跟踪：order_track
LS/PS接受quote订单记录：accept_order
LS/PS接受quote订单进入生产线时间：enter_epoch
'''

V={param:{} for param in kappa_zeta}
for param in kappa_zeta:
   action=get_action_space(s)
   V[param]={quote:None for quote in action}
   for quote in action:
      V[param][quote]={'Revenu':None,'complete_epoch':None} 
      Revenue,order_track,complete_epoch,cumsum,accept_order=train(s,quote,pair,param[0],param[1],initial_order)
      V[param][quote]['Revenu']=cumsum # accumulative income of all orders
      V[param][quote]['complete_epoch']=complete_epoch  #  record all orders processing finished
      V[param][quote]['order_track']=order_track  #   track all order state
      V[param][quote]['accept_order']=accept_order  # record the quote order accepted by LS/PS customer including  type of customer and epoch info

'''
数据处理：收入-所有订单的累计Total_Cost得到利润数据profit，比较不同变量取值（(κ,ζ,initial_system_lorder,ksi_L,ksi_p)得到最大化利润的optimal_quote（p,L)策略
并记录各种（p，L) 策略(包含最大化利润)接受quote的PS/LS类型客户类型 Customer_type，计算占比percentage_type    
'''
max_value_list=[]   
optimal_quote={param:None for param in kappa_zeta}
Total_Cost={param:{} for param in kappa_zeta}
profit={param:{} for param in kappa_zeta}
Customer_type={param: {} for param in kappa_zeta}
for param in V:
   max_value=0
   for quote in V[param]:
       enter_epoch=[order[1] for order in V[param][quote]['accept_order']]
       Customer_type[param][quote]=[order[0] for order in V[param][quote]['accept_order']]
       leave_epoch=[order[1] for order in V[param][quote]['complete_epoch']]
       if len(enter_epoch)==len(leave_epoch):
           Total_Cost[param][quote]=[compute_total_cost(enter_epoch[i], leave_epoch[i], pair, quote) for i in range(len(enter_epoch))]
       else:
            Total_Cost[param][quote]=[compute_total_cost(enter_epoch[i], leave_epoch[i], pair, quote) for i in range(len(leave_epoch))]+[compute_total_cost(0,i,pair, quote) for i in range(len(enter_epoch)-len(leave_epoch))]
       total_cost=np.sum(Total_Cost[param][quote])
       total_profit=V[param][quote]['Revenu']-total_cost
       profit[param][quote]=total_profit
       if total_profit>=max_value:
         max_value=total_profit
         optimal_quote[param]=quote
   max_value_list.append(max_value)  
pecentage_type={param:{} for param in kappa_zeta}
for param in Customer_type:
    for quote in Customer_type[param]:
        if len(Customer_type[param][quote])>0:
            pecentage_type[param][quote]={'LS':None,'PS':None}
            pecentage_type[param][quote]['LS']=str(round(Customer_type[param][quote].count('ls')/len(Customer_type[param][quote]) *100,2))+"%"
            pecentage_type[param][quote]['PS']=str(round(Customer_type[param][quote].count('ps')/len(Customer_type[param][quote]) *100,2))+"%"

  
               #程序运行到这里完成FCFS模式下MDP求解，得到一系列最佳（p,L)值'''
"###############################################################################################################################################################################################################################"
"第二阶段程序基于上面得到的一组不同（p,L)下订单情况进入EDD系统生成优化，目标是降低延迟交货成本tcost来提高利润"
'''
Simultaneous Approach betwwen MDP-FCFS(Marketing) and EDD (Manufacture)
EDD排单生产
'''

"比较函数"    
def largest(arr, n):
    max = 0
    finalarr = []
    for i in range(0, n):
        if arr[i][1] > max:
            max = arr[i][1]
            finalarr = arr[i]
    return finalarr

"EDD排队函数"
def edd_processing(task_dict):  
     cache_order = []
     order = []
     total_processing = 0
     no_of_tasks = len(task_dict.keys())
     new_task_dict=dict(sorted(task_dict.items(),key=lambda x:x[1][0],reverse=False)) # sort the orders in ascending order according to their due dates
     k = 0
     key=list(new_task_dict.keys())
     # to decide on first Job
     if k==0 and  new_task_dict[key[0]][0]>=new_task_dict[key[0]][1] :
         cache_order.append(new_task_dict[key[0]])
         k += 1
         total_processing += new_task_dict[key[0]][1]
     else:
         order.append(new_task_dict[key[0]])
         k += 1
     n = len(cache_order)
     x = 0
     while k <= no_of_tasks - 1:
         if float(new_task_dict[key[k]][1]+total_processing)<=float(new_task_dict[key[k]][0]):
             cache_order.append(new_task_dict[key[k]])
             total_processing +=new_task_dict[key[k]][1]
             k += 1
             n = len(cache_order)

         else:
             cache_order.append(new_task_dict[key[k]])
             n = len(cache_order)
             order.append(largest(cache_order, n))
             x = largest(cache_order, n)
             total_processing -= x[1]
             total_processing += new_task_dict[key[k]][1]
             cache_order.remove(largest(cache_order, n))
             n = len(cache_order)
             k += 1
     order.reverse()
     final = cache_order + order  
     return final,len(order)

max_profit_per_time_sim=[]   
optimal_quote_order={param:{} for param in kappa_zeta}  # 最佳(p,L)策略订单记录
order_task_dictionary={param:{} for param in kappa_zeta} # 生成订单记录进入EDD系统处理
cost_edd_change=[]

for param in optimal_quote:
    optimal_quote_order[param][optimal_quote[param]]={}
    optimal_quote_order[param][optimal_quote[param]]['order_track']=V[param][optimal_quote[param]]['order_track']
    optimal_quote_order[param][optimal_quote[param]]['complete_epoch']=V[param][optimal_quote[param]]['complete_epoch']
    optimal_quote_order[param][optimal_quote[param]]['accept_order']=V[param][optimal_quote[param]]['accept_order']
    optimal_quote_order[param][optimal_quote[param]]['revenue']=V[param][optimal_quote[param]]['Revenu']
    enter_epoch=[order[1] for order in V[param][optimal_quote[param]]['accept_order']]
    leave_epoch=[order[1] for order in V[param][optimal_quote[param]]['complete_epoch']]
    if len(enter_epoch)==len(leave_epoch):
      optimal_quote_order[param][optimal_quote[param]]['remain']=[(leave_epoch[i]-enter_epoch[i]) for i in range(len(enter_epoch))]
    else:
      optimal_quote_order[param][optimal_quote[param]]['remain']=[(leave_epoch[i]-enter_epoch[i]) for i in range(len(leave_epoch))]+[optimal_quote[param][1]-i for i in range(len(enter_epoch)-len(leave_epoch))]
    optimal_quote_order[param][optimal_quote[param]]['intersect_list']=[i for i in range(1,len(leave_epoch)) if enter_epoch[i]<leave_epoch[i-1]]
    optimal_quote_order[param][optimal_quote[param]]['remain_priority']=[optimal_quote_order[param][optimal_quote[param]]['remain'][value] for value in  optimal_quote_order[param][optimal_quote[param]]['intersect_list']]
      
for param in optimal_quote_order:
  for quote in optimal_quote_order[param]:
        order_task_dictionary[param][quote]={}
        order_task_dictionary[param][quote]['task']=[optimal_quote_order[param][quote]['accept_order'][i] for i in optimal_quote_order[param][quote]['intersect_list']]
        order_task_dictionary[param][quote]['enter_epoch']=[optimal_quote_order[param][quote]['accept_order'][i][1] for i in optimal_quote_order[param][quote]['intersect_list']]   
        order_task_dictionary[param][quote]['remain']=optimal_quote_order[param][quote]['remain_priority']
        
        order_task_dictionary[param][quote]['task_dict']=dict(zip(order_task_dictionary[param][quote]['task'],list(zip(optimal_quote_order[param][quote]['remain_priority'],order_task_dictionary[param][quote]['enter_epoch']))))
        order_task_dictionary[param][quote]['task_dict']={key:list(value) for key,value in order_task_dictionary[param][quote]['task_dict'].items()}
        # order_task_dictionary[param][quote]['task_due']={ order_task_dictionary[param][quote]['task'][i]:[optimal_quote_order[param][quote]['complete_epoch'][i][1],optimal_quote_order[param][quote]['remain_priority'][i],i] for i in range(len(optimal_quote_order[param][quote]['remain_priority']))}
        order_task_dictionary[param][quote]['task_due']={ order_task_dictionary[param][quote]['task'][i]:[optimal_quote_order[param][quote]['remain_priority'][i]+ order_task_dictionary[param][quote]['enter_epoch'][i],optimal_quote_order[param][quote]['remain_priority'][i],i] for i in range(len(optimal_quote_order[param][quote]['remain_priority']))}
        try:
            edd_order_list,ntard_order=edd_processing( order_task_dictionary[param][quote]['task_due'])
            order_task_dictionary[param][quote]['edd_order']=edd_order_list
            order_task_dictionary[param][quote]['ntard_order']=ntard_order
            if ntard_order>0:
              order_task_dictionary[param][quote]['tard_order']=order_task_dictionary[param][quote]['edd_order'][-ntard_order:]
            else:
               order_task_dictionary[param][quote]['tard_order']=0 
            # 如果EDD排队改变了订单处理顺序，则计算这部分重排后和原先FCFS订单处理的成本差异
            cost_origin=[]
            if ntard_order<=quote[1] and ntard_order>0:
                # cost_origin=[]
                # cost_edd=ntard_order*(ntard_order-1)/2*pair[1]+pair[0]
                for order in  order_task_dictionary[param][quote]['tard_order']:
                     cost_origin.append(compute_total_cost( order[0]-order[1], order[0],pair,quote))
                order_task_dictionary[param][quote]['cost_change']=np.sum(cost_origin) 
            elif ntard_order>quote[1]:
                for order in  order_task_dictionary[param][quote]['tard_order'][-quote[1]:]:
                     cost_origin.append(compute_total_cost( order[0]-order[1], order[0],pair,quote))
                order_task_dictionary[param][quote]['cost_change']=np.sum(cost_origin) 
            else:
                order_task_dictionary[param][quote]['cost_change']=0
            cost_edd_change.append(order_task_dictionary[param][quote]['cost_change'])
        except Exception as e:  #处理空数据无法EDD情况
            cost_edd_change.append(0) 
            print(e)
"计算FCFS和EDD的最大化长期单位时间利润"        
max_profit_per_time=list(np.array(max_value_list)/2000)
max_profit_per_time_sim=(np.array(max_value_list)+np.array(cost_edd_change))/2000 
max_profit_per_time_sim=[round(value,3) for value in max_profit_per_time_sim]
max_profit_per_time=[round(value,3) for value in max_profit_per_time]    
max_profit_per_change=np.array(max_profit_per_time_sim)-np.array(max_profit_per_time)
     
Output=pd.DataFrame({'(κ,ζ)':list(optimal_quote.keys()),'quote':list(optimal_quote.values()),'Max_Profit_FCFS':max_profit_per_time,'Max_Profit_EDD':max_profit_per_time_sim,})                                                                 
Output.to_excel('MDP_data_with Parameter{}{}{}{}.xlsx'.format(initial_order,s[0],ksi_L,ksi_p),encoding='utf-8',sheet_name="n={},type={},ξL={},ξp={}".format(initial_order,s[0],ksi_L,ksi_p),index=False)
 
















