# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 17:24:21 2022

@author: JK
"""

from random import *
import matplotlib.pyplot as plt

n=200
x_values=[]
y_values=[]

for i in range(n):
    choose=randrange(2)
    x=0
    y=0
    
    if choose==0:
        x=randint(0,30)
        y=randint(0,30)
        
    else:
        x=randint(30,50)
        y=randint(30,50)
        
    x_values.append(x)
    y_values.append(y)
    
n_cluster=2

center_x=[]
center_y=[]

for i in range(n_cluster):
    cx=randint(0,50)
    cy=randint(0,50)
    
    center_x.append(cx)
    center_y.append(cy)



color = ['#9467bd','#bcbd22','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','7f7f7f','#bcbd22','#17becf']


while True:
    move=0
    
    cluster=[[[],[]] for i in range(n_cluster)]

    for i in range(n):
        cluster_num=-1 
        min_d=float("inf") 

        for c in range(n_cluster):
          
            distance=abs(x_values[i]-center_x[c])**2+abs(y_values[i]-center_y[c])**2

            if min_d>distance:
                min_d=distance
                cluster_num=c

       
        cluster[cluster_num][0].append(x_values[i])
        cluster[cluster_num][1].append(y_values[i])
    
  
    for c in range(n_cluster):
        ncenter_x=sum(cluster[c][0])//len(cluster[c][0])
        ncenter_y=sum(cluster[c][1])//len(cluster[c][1])
    
        if ncenter_x!=center_x[c] or ncenter_y!=center_y[c]:
            move=1
            center_x[c]=ncenter_x
            center_y[c]=ncenter_y    

    if move==0:
        break

for c in range(n_cluster):
    plt.scatter(cluster[c][0],cluster[c][1],c=color[c])
plt.scatter(center_x,center_y, c='red',marker='x')
plt.show()
        
        
        
        
        
    