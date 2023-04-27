# -*- coding: utf-8 -*-
"""
"""
# -*- coding: utf-8 -*-

import random
import math
from matplotlib import pyplot as plt
import networkx as nx
import time
import numpy as np
import pandas as pd
import bokeh
from bokeh.io import show, output_notebook, output_file
from bokeh.models import Plot, Range1d, MultiLine, Circle, StaticLayoutProvider
from bokeh.models import HoverTool, BoxZoomTool, ResetTool
from bokeh.models.graphs import from_networkx
from bokeh.palettes import Category20_20,Turbo256
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, show
#Parameter
K_r = 6   #库仑斥力系数
K_s = 0.3  #引力系数
L = 5     # L是两点之间距离  只有邻接的两个点之间才计算斥力
delta_t =40
MaxLength = 20  #节点最大偏移量
iterations=200    #模拟退火采样次数
color=['red','green','blue','orange'] #颜色通道表示连通分支
Displacement_list=[]#模拟退火斥力产生的正向偏移采样列表
scale=3#采样范围

data=[['YDR277C', 'YDL194W', 'pp'], ['YDR277C', 'YJR022W', 'pp'],
         ['YPR145W', 'YMR117C', 'pp'], ['YER054C', 'YBR045C', 'pp'],
         ['YER054C', 'YER133W', 'pp'], ['YBR045C', 'YOR178C', 'pp'],
         ['YBR045C', 'YIL045W', 'pp'], ['YBL079W', 'YDL088C', 'pp'],
         ['YLR345W', 'YLR321C', 'pp'], ['YGR136W', 'YGR058W', 'pp'],
         ['YDL023C', 'YJL159W', 'pp'], ['YBR170C', 'YGR048W', 'pp'],
         ['YGR074W', 'YBR043C', 'pp'], ['YGL202W', 'YGR074W', 'pp'],
         ['YLR197W', 'YOR310C', 'pp'], ['YLR197W', 'YDL014W', 'pp'],
         ['YDL088C', 'YER110C', 'pp'], ['YMR117C', 'YPR010C', 'pp'],
         ['YMR117C', 'YCL032W', 'pp'], ['YML114C', 'YDR167W', 'pp'],
         ['YNL036W', 'YIR009W', 'pp'], ['YOR212W', 'YLR362W', 'pp'],
         ['YDR070C', 'YFL017C', 'pp'], ['YGR046W', 'YNL236W', 'pp'],
         ['YIL070C', 'YML054C', 'pp'], ['YPR113W', 'YMR043W', 'pd'],
         ['YER081W', 'YIL074C', 'pp'], ['YGR088W', 'YLR256W', 'pd'],
         ['YDR395W', 'YPR102C', 'pp'], ['YDR395W', 'YOL127W', 'pp'],
         ['YDR395W', 'YIL052C', 'pp'], ['YDR395W', 'YER056CA', 'pp'],
         ['YDR395W', 'YNL069C', 'pp'], ['YDR395W', 'YIL133C', 'pp'],
         ['YDR395W', 'YDL075W', 'pp'], ['YGR085C', 'YDR395W', 'pp'],
         ['YGR085C', 'YLR075W', 'pp'], ['YDL030W', 'YMR005W', 'pp'],
         ['YDL030W', 'YDL013W', 'pp'], ['YER079W', 'YNL154C', 'pp'],
         ['YER079W', 'YHR135C', 'pp'], ['YDL215C', 'YLR432W', 'pp'],
         ['YDL215C', 'YER040W', 'pd'], ['YPR041W', 'YMR309C', 'pp'],
         ['YPR041W', 'YOR361C', 'pp'], ['YOR120W', 'YPL248C', 'pd'],
         ['YIL074C', 'YNL311C', 'pp'], ['YDR299W', 'YJL194W', 'pd'],
         ['YHR005C', 'YLR362W', 'pp'], ['YLR452C', 'YHR005C', 'pp'],
         ['YMR255W', 'YGL122C', 'pp'], ['YBR274W', 'YMR255W', 'pp'],
         ['YHR084W', 'YFL026W', 'pd'], ['YHR084W', 'YDR461W', 'pd'],
         ['YHR084W', 'YMR043W', 'pp'], ['YBL050W', 'YOR036W', 'pp'],
         ['YBL026W', 'YOR167C', 'pp'], ['YJL194W', 'YMR043W', 'pd'],
         ['YLR258W', 'YBR274W', 'pp'], ['YLR258W', 'YIL045W', 'pp'],
         ['YGL134W', 'YLR258W', 'pp'], ['YGL134W', 'YPL031C', 'pp'],
         ['YPR124W', 'YMR021C', 'pd'], ['YNL135C', 'YDR174W', 'pp'],
         ['YER052C', 'YNL135C', 'pp'], ['YPL240C', 'YOR036W', 'pp'],
         ['YPL240C', 'YBR155W', 'pp'], ['YLR075W', 'YPR102C', 'pp'],
         ['YKL161C', 'YPL089C', 'pp'], ['YAR007C', 'YPL111W', 'pd'],
         ['YAR007C', 'YML032C', 'pp'], ['YDR142C', 'YIL160C', 'pp'],
         ['YDR142C', 'YGL153W', 'pp'], ['YDR244W', 'YDL078C', 'pp'],
         ['YDR244W', 'YDR142C', 'pp'], ['YDR244W', 'YNL214W', 'pp'],
         ['YDR244W', 'YGL153W', 'pp'], ['YDR244W', 'YLR191W', 'pp'],
         ['YDR167W', 'YLR432W', 'pp'], ['YLR175W', 'YNL307C', 'pp'],
         ['YNL117W', 'YJL089W', 'pd'], ['YOR089C', 'YDR323C', 'pp'],
         ['YNL214W', 'YGL153W', 'pp'], ['YBR135W', 'YER102W', 'pp'],
         ['YER110C', 'YML007W', 'pp'], ['YLR191W', 'YGL153W', 'pp'],
         ['YOL149W', 'YOR167C', 'pp'], ['YMR044W', 'YIL061C', 'pp'],
         ['YNL113W', 'YPR110C', 'pp'], ['YDR354W', 'YEL009C', 'pd'],
         ['YKL211C', 'YER090W', 'pp'], ['YDR146C', 'YGL035C', 'pd'],
         ['YDR146C', 'YMR043W', 'pd'], ['YER111C', 'YMR043W', 'pd'],
         ['YOR039W', 'YOR303W', 'pp'], ['YML024W', 'YNL216W', 'pd'],
         ['YIL113W', 'YHR030C', 'pp'], ['YLL019C', 'YIL113W', 'pp'],
         ['YDR009W', 'YGL035C', 'pd'], ['YML051W', 'YDR009W', 'pp'],
         ['YML051W', 'YBR020W', 'pp'], ['YPL031C', 'YHR071W', 'pp'],
         ['YML123C', 'YFR034C', 'pd'], ['YMR058W', 'YER145C', 'pp'],
         ['YML074C', 'YJL190C', 'pp'], ['YOR355W', 'YNL091W', 'pp'],
         ['YFL038C', 'YOR036W', 'pp'], ['YIL162W', 'YNL167C', 'pd'],
         ['YER133W', 'YBR050C', 'pp'], ['YER133W', 'YMR311C', 'pp'],
         ['YER133W', 'YOR315W', 'pp'], ['YER133W', 'YOR178C', 'pp'],
         ['YER133W', 'YDR412W', 'pp'], ['YFR037C', 'YOR290C', 'pp'],
         ['YFR034C', 'YBR093C', 'pd'], ['YAL040C', 'YMR043W', 'pd'],
         ['YGR048W', 'YPL222W', 'pp'], ['YMR291W', 'YGL115W', 'pp'],
         ['YGR009C', 'YBL050W', 'pp'], ['YGR009C', 'YDR335W', 'pp'],
         ['YGR009C', 'YOR327C', 'pp'], ['YGR009C', 'YAL030W', 'pp'],
         ['YMR183C', 'YGR009C', 'pp'], ['YGL161C', 'YDR100W', 'pp'],
         ['YDL063C', 'YPL131W', 'pp'], ['YNL167C', 'YOR202W', 'pd'],
         ['YHR115C', 'YOR215C', 'pp'], ['YEL041W', 'YHR115C', 'pp'],
         ['YJL036W', 'YDL113C', 'pp'], ['YBR109C', 'YFR014C', 'pp'],
         ['YBR109C', 'YOR326W', 'pp'], ['YOL016C', 'YBR109C', 'pp'],
         ['YNL311C', 'YKL001C', 'pp'], ['YLR319C', 'YLR362W', 'pp'],
         ['YNL189W', 'YPR062W', 'pp'], ['YNL189W', 'YER065C', 'pp'],
         ['YNL189W', 'YPL111W', 'pp'], ['YNL189W', 'YDL236W', 'pp'],
         ['YBL069W', 'YGL008C', 'pp'], ['YGL073W', 'YER103W', 'pd'],
         ['YGL073W', 'YHR055C', 'pd'], ['YGL073W', 'YHR053C', 'pd'],
         ['YGL073W', 'YOR178C', 'pp'], ['YBR072W', 'YGL073W', 'pd'],
         ['YLR321C', 'YDR412W', 'pp'], ['YPR048W', 'YDL215C', 'pp'],
         ['YPR048W', 'YOR355W', 'pp'], ['YNL199C', 'YPR048W', 'pp'],
         ['YPL075W', 'YOL086C', 'pd'], ['YPL075W', 'YDR050C', 'pd'],
         ['YPL075W', 'YNL199C', 'pp'], ['YPL075W', 'YHR174W', 'pd'],
         ['YPL075W', 'YGR254W', 'pd'], ['YPL075W', 'YCR012W', 'pd'],
         ['YFL039C', 'YHR179W', 'pp'], ['YFL039C', 'YCL040W', 'pp'],
         ['YDR382W', 'YFL039C', 'pp'], ['YDR382W', 'YDL130W', 'pp'],
         ['YJR066W', 'YLR116W', 'pp'], ['YNL154C', 'YKL204W', 'pp'],
         ['YNL047C', 'YIL105C', 'pp'], ['YHR135C', 'YNL116W', 'pp'],
         ['YML064C', 'YDR174W', 'pp'], ['YML064C', 'YLR284C', 'pp'],
         ['YML064C', 'YHR198C', 'pp'], ['YKL074C', 'YGL035C', 'pp'],
         ['YDL081C', 'YLR340W', 'pp'], ['YGL166W', 'YHR055C', 'pd'],
         ['YGL166W', 'YHR053C', 'pd'], ['YLL028W', 'YGL166W', 'pp'],
         ['YDR335W', 'YDR174W', 'pp'], ['YMR021C', 'YLR214W', 'pd'],
         ['YJL089W', 'YLR377C', 'pd'], ['YJL089W', 'YKR097W', 'pd'],
         ['YJL089W', 'YER065C', 'pd'], ['YHR030C', 'YER111C', 'pp'],
         ['YHR030C', 'YLL021W', 'pp'], ['YPL089C', 'YHR030C', 'pp'],
         ['YGL115W', 'YGL208W', 'pp'], ['YLR310C', 'YER103W', 'pp'],
         ['YNL098C', 'YLR310C', 'pp'], ['YER040W', 'YGR019W', 'pd'],
         ['YER040W', 'YPR035W', 'pd'], ['YGL008C', 'YMR043W', 'pd'],
         ['YOR036W', 'YGL161C', 'pp'], ['YOR036W', 'YDR100W', 'pp'],
         ['YDR323C', 'YOR036W', 'pp'], ['YBL005W', 'YJL219W', 'pd'],
         ['YBR160W', 'YGR108W', 'pp'], ['YBR160W', 'YMR043W', 'pd'],
         ['YKL101W', 'YBR160W', 'pp'], ['YOL156W', 'YBL005W', 'pd'],
         ['YLL021W', 'YLR362W', 'pp'], ['YJL203W', 'YOL136C', 'pp'],
         ['YJL157C', 'YOR212W', 'pp'], ['YJL157C', 'YAL040C', 'pp'],
         ['YNL145W', 'YHR084W', 'pd'], ['YNL145W', 'YCL067C', 'pd'],
         ['YGR108W', 'YBR135W', 'pp'], ['YMR043W', 'YFL026W', 'pd'],
         ['YMR043W', 'YJL157C', 'pd'], ['YMR043W', 'YNL145W', 'pd'],
         ['YMR043W', 'YDR461W', 'pd'], ['YMR043W', 'YGR108W', 'pd'],
         ['YMR043W', 'YKR097W', 'pd'], ['YMR043W', 'YJL159W', 'pd'],
         ['YMR043W', 'YIL015W', 'pd'], ['YKL109W', 'YGL035C', 'pd'],
         ['YKL109W', 'YJR048W', 'pd'], ['YKL109W', 'YBL021C', 'pp'],
         ['YKL109W', 'YGL237C', 'pp'], ['YBR217W', 'YNR007C', 'pp'],
         ['YHR171W', 'YNR007C', 'pp'], ['YHR171W', 'YDR412W', 'pp'],
         ['YPL149W', 'YBR217W', 'pp'], ['YPL149W', 'YHR171W', 'pp'],
         ['YDR311W', 'YKL028W', 'pp'], ['YBL021C', 'YJR048W', 'pd'],
         ['YGL237C', 'YJR048W', 'pd'], ['YLR256W', 'YEL039C', 'pd'],
         ['YLR256W', 'YJR048W', 'pd'], ['YLR256W', 'YML054C', 'pd'],
         ['YJR109C', 'YOR303W', 'pp'], ['YGR058W', 'YOR264W', 'pp'],
         ['YLR229C', 'YJL157C', 'pp'], ['YDR309C', 'YLR229C', 'pp'],
         ['YLR116W', 'YKL012W', 'pp'], ['YNL312W', 'YPL111W', 'pd'],
         ['YML032C', 'YNL312W', 'pp'], ['YKL012W', 'YKL074C', 'pp'],
         ['YNL236W', 'YKL012W', 'pp'], ['YNL091W', 'YNL164C', 'pp'],
         ['YDR184C', 'YLR319C', 'pp'], ['YIL143C', 'YDR311W', 'pp'],
         ['YIR009W', 'YNL091W', 'pp'], ['YIR009W', 'YDR184C', 'pp'],
         ['YIR009W', 'YIL143C', 'pp'], ['YIR009W', 'YKR099W', 'pp'],
         ['YPL248C', 'YBR018C', 'pd'], ['YPL248C', 'YLR081W', 'pd'],
         ['YPL248C', 'YBR020W', 'pd'], ['YPL248C', 'YML051W', 'pp'],
         ['YPL248C', 'YML051W', 'pd'], ['YPL248C', 'YGL035C', 'pd'],
         ['YPL248C', 'YJR048W', 'pd'], ['YPL248C', 'YBR019C', 'pd'],
         ['YBR020W', 'YGL035C', 'pd'], ['YGL035C', 'YIL162W', 'pd'],
         ['YGL035C', 'YLR377C', 'pd'], ['YGL035C', 'YLR044C', 'pd'],
         ['YOL051W', 'YBR018C', 'pd'], ['YOL051W', 'YPL248C', 'pp'],
         ['YOL051W', 'YLR081W', 'pd'], ['YOL051W', 'YBR020W', 'pd'],
         ['YBR019C', 'YGL035C', 'pd'], ['YBR019C', 'YOL051W', 'pd'],
         ['YJR060W', 'YPR167C', 'pd'], ['YDR103W', 'YLR362W', 'pp'],
         ['YLR362W', 'YPL240C', 'pp'], ['YLR362W', 'YMR186W', 'pp'],
         ['YLR362W', 'YER124C', 'pp'], ['YCL032W', 'YDR103W', 'pp'],
         ['YCL032W', 'YLR362W', 'pp'], ['YCL032W', 'YDR032C', 'pp'],
         ['YMR138W', 'YLR109W', 'pp'], ['YMR138W', 'YHR141C', 'pp'],
         ['YOL058W', 'YNL189W', 'pp'], ['YEL009C', 'YMR300C', 'pd'],
         ['YEL009C', 'YOL058W', 'pd'], ['YEL009C', 'YBR248C', 'pd'],
         ['YEL009C', 'YCL030C', 'pd'], ['YEL009C', 'YOR202W', 'pd'],
         ['YEL009C', 'YMR108W', 'pd'], ['YMR186W', 'YBR155W', 'pp'],
         ['YOR326W', 'YGL106W', 'pp'], ['YMR309C', 'YNL047C', 'pp'],
         ['YOR361C', 'YDR429C', 'pp'], ['YOR361C', 'YMR309C', 'pp'],
         ['YER179W', 'YIL105C', 'pp'], ['YER179W', 'YLR134W', 'pp'],
         ['YER179W', 'YLR044C', 'pp'], ['YDL014W', 'YOR310C', 'pp'],
         ['YPR119W', 'YMR043W', 'pd'], ['YLR117C', 'YBR190W', 'pp'],
         ['YGL013C', 'YOL156W', 'pd'], ['YGL013C', 'YJL219W', 'pd'],
         ['YCR086W', 'YOR264W', 'pp'], ['YDR412W', 'YPR119W', 'pp'],
         ['YDR412W', 'YLR117C', 'pp'], ['YDR412W', 'YGL013C', 'pp'],
         ['YDR412W', 'YCR086W', 'pp'], ['YER062C', 'YPL201C', 'pp'],
         ['YOR327C', 'YER143W', 'pp'], ['YAL030W', 'YER143W', 'pp'],
         ['YCL030C', 'YKR099W', 'pd'], ['YCR012W', 'YJR060W', 'pd'],
         ['YNL216W', 'YOL086C', 'pd'], ['YNL216W', 'YDR050C', 'pd'],
         ['YNL216W', 'YOL127W', 'pd'], ['YNL216W', 'YAL038W', 'pd'],
         ['YNL216W', 'YIL069C', 'pd'], ['YNL216W', 'YER074W', 'pd'],
         ['YNL216W', 'YBR093C', 'pd'], ['YNL216W', 'YDR171W', 'pp'],
         ['YNL216W', 'YCL030C', 'pd'], ['YNL216W', 'YNL301C', 'pd'],
         ['YNL216W', 'YOL120C', 'pd'], ['YNL216W', 'YLR044C', 'pd'],
         ['YNL216W', 'YIL133C', 'pd'], ['YNL216W', 'YHR174W', 'pd'],
         ['YNL216W', 'YGR254W', 'pd'], ['YNL216W', 'YCR012W', 'pd'],
         ['YAL038W', 'YPL075W', 'pd'], ['YNL307C', 'YAL038W', 'pp'],
         ['YER116C', 'YDL013W', 'pp'], ['YNR053C', 'YDL030W', 'pp'],
         ['YNR053C', 'YJL203W', 'pp'], ['YLR264W', 'YBL026W', 'pp'],
         ['YLR264W', 'YOL149W', 'pp'], ['YLR264W', 'YER112W', 'pp'],
         ['YEL015W', 'YML064C', 'pp'], ['YNR050C', 'YMR138W', 'pp'],
         ['YJR022W', 'YNR053C', 'pp'], ['YJR022W', 'YLR264W', 'pp'],
         ['YJR022W', 'YOR167C', 'pp'], ['YJR022W', 'YEL015W', 'pp'],
         ['YJR022W', 'YNL050C', 'pp'], ['YJR022W', 'YNR050C', 'pp'],
         ['YER112W', 'YOR167C', 'pp'], ['YCL067C', 'YFL026W', 'pd'],
         ['YCL067C', 'YDR461W', 'pd'], ['YCL067C', 'YMR043W', 'pp'],
         ['YCL067C', 'YIL015W', 'pd'], ['YCR084C', 'YCL067C', 'pp'],
         ['YCR084C', 'YBR112C', 'pp'], ['YIL061C', 'YLR153C', 'pp'],
         ['YIL061C', 'YNL199C', 'pp'], ['YIL061C', 'YDL013W', 'pp'],
         ['YGR203W', 'YIL061C', 'pp'], ['YJL013C', 'YGL229C', 'pp'],
         ['YJL030W', 'YGL229C', 'pp'], ['YGR014W', 'YJL013C', 'pp'],
         ['YGR014W', 'YJL030W', 'pp'], ['YPL211W', 'YGR014W', 'pp'],
         ['YOL123W', 'YGL044C', 'pp'], ['YFL017C', 'YOR362C', 'pp'],
         ['YFL017C', 'YER102W', 'pp'], ['YFL017C', 'YOL059W', 'pp'],
         ['YDR429C', 'YFL017C', 'pp'], ['YMR146C', 'YDR429C', 'pp'],
         ['YLR293C', 'YGL097W', 'pp'], ['YBR118W', 'YAL003W', 'pp'],
         ['YPR080W', 'YAL003W', 'pp'], ['YLR249W', 'YBR118W', 'pp'],
         ['YLR249W', 'YPR080W', 'pp'], ['YGL097W', 'YOR204W', 'pp'],
         ['YGR218W', 'YGL097W', 'pp'], ['YGL122C', 'YOL123W', 'pp'],
         ['YKR026C', 'YGL122C', 'pp']]

class Force_Directed_Layout():
   def __init__(self,Node_num,max_pos_x,max_pos_y,node_list):
       self.Node_num=Node_num
       self.max_posx=max_pos_x
       self.max_posy=max_pos_y
       self.Edge=[]  #边
       self.Node_force={}  #引力
       self.Node_position={}  #坐标
       self.Node_degree=[]   #节点的度用来反映节点大小
       self.Node_name=node_list
   def init_edge_force(self):
       
       for i in range(0,self.Node_num):#生成点坐标,初始化斥力
           #np.random.seed(42) #固定坐标
           posx=random.uniform(0,self.max_posx)
           posy=random.uniform(0,self.max_posy)#初始化点坐标和受力
           self.Node_position[i]=(posx,posy)
           self.Node_force[i]=(0,0)
           node_name=self.Node_name[i]
           node_adj=adj_node[node_name]
           index=[self.Node_name.index(node) for node in node_adj]
           for num in index:
             self.Edge.append((i,num))  #初始化边

   def compute_repulsion(self):#计算每两个点之间的斥力
        for i in range(0,self.Node_num):
            for j in range(i+1,self.Node_num):
                dx=self.Node_position[j][0]-self.Node_position[i][0]
                dy=self.Node_position[j][1]-self.Node_position[i][1]
                if dx!=0 or dy!=0:
                    distanceSquared=dx*dx+dy*dy
                    distance=math.sqrt(distanceSquared)
                    R_force=K_r/distanceSquared
                    fx=R_force*dx/distance
                    fy=R_force*dy/distance#更新受力
                    fi_x=self.Node_force[i][0]
                    fi_y=self.Node_force[i][1]
                    self.Node_force[i]=(fi_x-fx,fi_y-fy)
                    fj_x=self.Node_force[j][0]
                    fj_y=self.Node_force[j][1]
                    self.Node_force[j]=(fj_x+fx,fj_y+fy)

   def compute_string(self):
            for i in range(0,self.Node_num):#取出其邻居
                neighbors=[n for n in G[i]]#对每一个邻居，计算斥力
                for j in neighbors:
                    if i < j:
                        dx=self.Node_position[j][0]-self.Node_position[i][0]
                        dy=self.Node_position[j][1]-self.Node_position[i][1]
                        if dx!=0 or dy!=0:
                            distance=math.sqrt(dx*dx+dy*dy)
                            S_force=K_s*(distance-L)
                            fx=S_force*dx/distance
                            fy=S_force*dy/distance#更新受力
                            fi_x=self.Node_force[i][0]
                            fi_y=self.Node_force[i][1]
                            self.Node_force[i]=(fi_x+fx,fi_y+fy)
                            fj_x=self.Node_force[j][0]
                            fj_y=self.Node_force[j][1]
                            self.Node_force[j]=(fj_x-fx,fj_y-fy)
                
   def update_position(self,times):
    '''
    更新坐标  模拟退火策略
    移动的最大步长设置为关于迭代次数的减函数， y = 1/x 
    初期节点允许移动的最大距离很大，后期很小
    节点的偏移总量进行连续采样，图布局变化不大时结束迭代
    '''
    Displacement_sum=0
    for i in range(0,self.Node_num):
        dx = delta_t*self.Node_force[i][0]
        dy = delta_t*self.Node_force[i][1]
        displacementSquard = dx*dx + dy*dy
        #随迭代次数增加，MaxLength逐渐减小；
        current_MaxLength = MaxLength/(times+0.1)

        if( displacementSquard >current_MaxLength):
            s=math.sqrt(current_MaxLength/displacementSquard)
            dx=dx*s
            dy=dy*s
        (newx,newy) = (self.Node_position[i][0]+dx, self.Node_position[i][1]+dy)
        Displacement_sum += math.sqrt(dx*dx + dy*dy) 
        self.Node_position[i]=(newx,newy)
    return Displacement_sum
#得到节点和节点关系并画出F
def get_node_edge(data):
    
    node_list=[]
    for node_edge in data:
        for element in node_edge:
            if element not in node_list and element not in ['pp','pd']:
                node_list.append(element)
    adj_node={key:[] for  key in node_list}
    for edge_tuple in data:
        adj_node[edge_tuple[0]].append(edge_tuple[1])
        adj_node[edge_tuple[1]].append(edge_tuple[0])
    return node_list,adj_node

def plot_node_img(data,G):
    node_list,adj_node=get_node_edge(data) #获取节点
    node_num=len(node_list)
    G_layout=Force_Directed_Layout(node_num,60,40,node_list)
    G.add_nodes_from(list(range(0,G_layout.Node_num)))
    # G.add_nodes_from(node_list_uni)
    G_layout.init_edge_force()#初始化节点坐标和边
    G.add_edges_from(G_layout.Edge)
    
    #获得节点的度
    for i in range(0,G_layout.Node_num):
        G_layout.Node_degree.append(pow(G.degree(i),2))
   
    #获得联通子图
    connected_num=0
    connected_subgraph=[]
    for c in nx.connected_components(G):
        connected_num += 1
        nodeSet = G.subgraph(c)
        connected_subgraph.append(nodeSet) 
    
    
    #原图
    nx.draw_networkx_nodes(G, pos =G_layout.Node_position, node_size=20, node_color = 'red',alpha=0.8)
    nx.draw_networkx_edges(G, pos =G_layout.Node_position, edge_color='lightblue',width=0.5) 
    plt.show()

    #Force图
    fig=plt.figure('不同迭代次数force-direction_layout')
    start =time.perf_counter()
    iteration_time=0
    for times in range(0,1+iterations):
        for i in range(0,G_layout.Node_num):
            G_layout.Node_force[i]=(0,0)
        G_layout.compute_repulsion()
        G_layout.compute_string()
        #记录本次迭代移动距离：
        Displacement_sum = G_layout.update_position(times)
        Displacement_list.append(Displacement_sum)
        print(Displacement_sum)
        if len(Displacement_list)>scale:
            last = np.mean(Displacement_list[times-4:times-1])
            now = np.mean(Displacement_list[times-3:times])
            if (last-now)/last < 0.01:
                break
        iteration_time=times
    end = time.perf_counter()

    print('Running time: %s Seconds'%(end-start))
    print('最终迭代次数:',iteration_time)
    index=0
    for subgrap in connected_subgraph:
        sub_position = dict({i for i in G_layout.Node_position.items() if i[0] in list(subgrap)})
        sub_degree = [degree+5 for (point,degree) in enumerate(G_layout.Node_degree) if point in list(subgrap) ] 
        nx.draw_networkx_nodes(subgrap, pos = sub_position, node_size=sub_degree, node_color = color[index%4],alpha=0.8)
        nx.draw_networkx_edges(subgrap, pos = sub_position,edge_color='lightblue',width=0.5) 
        index+=1
    plt.show()
    graph_renderer = from_networkx(G, nx.spring_layout)
    # 画布
    plot = Plot(title="Graph layout demonstration",plot_width=800, plot_height=500,
                x_range=Range1d(-1.1, 1.1), y_range=Range1d(-1.1, 1.1),background_fill_color="#efefef",)
    graph_renderer.edge_renderer.glyph = MultiLine(line_alpha=0.5, line_width=1.5,line_color='blue')
    graph_renderer.node_renderer.data_source.data['colors']=Turbo256
    graph_renderer.node_renderer.glyph.update(size=8, fill_color="colors")     
    graph_renderer.node_renderer.data_source.data['name'] = node_list                                     
    # 工具条
    hover = HoverTool(tooltips=[('name', '@name')])  #显示标签
    plot.add_tools(hover, BoxZoomTool(), ResetTool())  #显示工具栏
    # 绘图
    plot.renderers.append(graph_renderer)
    output_file('test_graph.html')
    # 显示
    show(plot)

    
if __name__ == '__main__':
    G=nx.Graph()
    node_list,adj_node=get_node_edge(data) #函数
    plot_node_img(data,G)


