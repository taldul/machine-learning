# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

class Node:
    
    def __init__(self,dataset,condition,gini,num_of_samples,values,class_value):
        self.dataset=dataset
        self.left = None
        self.right = None
        self.condition=condition
        self.gini=gini
        self.num_of_samples=num_of_samples
        self.values=values
        self.class_value=class_value
        
        
class Tree:
    def __init__(self,root):
        self.root=root
        
real_estate=pd.read_csv('data.csv', index_col=[0])
classfier_arr=np.array(real_estate['area_type'])
real_estate=real_estate.iloc[:,1:]


def gini(attr_array,splitter, classffier_arr):
    groupA=np.array([0,0])
    groupB=np.array([0,0])
    for i in range(len(attr_array)):
        if attr_array[i]<splitter:
            if classffier_arr[i]==str('B'):
                groupA[0]+=1
            else:
                groupA[1]+=1
        else:
            if classffier_arr[i]==str('B'):
                groupB[0]+=1
            else:
                groupB[1]+=1
    
    
    
    
    giniA=(1-(groupA[0]/(groupA[0]+groupA[1]))**2-(groupA[1]/(groupA[0]+groupA[1]))**2)
    giniB=(1-(groupB[0]/(groupB[0]+groupB[1]))**2-(groupB[1]/(groupB[0]+groupB[1]))**2)
    total_gini=(np.sum(groupA)/len(attr_array))*giniA+(np.sum(groupB)/len(attr_array))*giniB
    return total_gini
print(gini(real_estate['availability'],0.5,classfier_arr))


def best_gini(attr_array):
    unique_attr_array=np.unique(attr_array)
    min_gini=1
    splitter=0
    for i in range((len(unique_attr_array)-1)):
        temp_gini= gini(attr_array,np.mean(unique_attr_array[i:i+2]),classfier_arr)
        if temp_gini<min_gini:
            min_gini=temp_gini
            splitter=np.mean(unique_attr_array[i:i+2])
        
    return min_gini,splitter
print(best_gini(real_estate['availability']))        


def choose_col(df):
    col=''
    splitter=0
    min_gini=1
    df.apply(best_gini, axis=0, broadcast=None, raw=False, 
                    reduce=None, result_type=None)
    for column in df:
        temp_gini,temp_splitter=best_gini(df[column])
        if temp_gini<min_gini:
            min_gini=temp_gini
            splitter=temp_splitter
            col=column.name
        print(col)
        print(splitter)
        print(min_gini)
     