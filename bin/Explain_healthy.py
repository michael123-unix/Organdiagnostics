#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 16:54:56 2026

@author: michael
"""

import argparse
import copy
import json
import pickle
import graphviz
import seaborn as sn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as st
from matplotlib_venn import venn3
import lightgbm as lgb
import shap
import os
import psutil

#%%Function for memory usage tracking
def reduce_mem_usage(df):
    for col in df.columns:
        if df[col].dtype == "float64":
            df[col]= df[col].astype(np.float32)
        if df[col].dtype == "int64":
            df[col]=df[col].astype(np.int32)
    return(df)
            
#%prepare data
if __name__== "__main__":
    parser = argparse.ArgumentParser(description="Explain the LGB model on the healthy individual to extract important features.")
    parser.add_argument("--train_data", help="Path to the training data CSV file", required=True)   
    parser.add_argument("--model", help="Path to the trained LightGBM model file", required=True)
    parser.add_argument("--interactions", action="store_true", help="Whether to calculate gene-gene interactions for the top 50 genes for each age group. This can be computationally intensive, so use with caution.")
    args = parser.parse_args()

#%%load data
# 1. Prepare your data
train_data = pd.read_csv(args.train_data)

train_data.index = train_data["index"]
train_data= train_data.drop(columns="index")
train_data.columns

reduce_mem_usage(train_data)

#prepare the train data for generating attributions and interactions
Y_train = train_data.pop("Age")
X_data_train = train_data
X_data_train.columns
Full_train_data = X_data_train.assign(Age = Y_train).copy()

#load model
gbm= lgb.Booster(model_file=args.model)

#%%shap values and overview importance plots
explainer = shap.TreeExplainer(gbm)

shap_values = explainer(X_data_train)
X_data_train.columns
shap_values.feature_names

#for explanations on each age group, subset the data and calculate averages of SHAP values for all features

    #generate masks for subsetting the Explainer object
mask_3= (Full_train_data["Age"] == 3).values
mask_18= (Full_train_data["Age"] == 18).values
mask_24= (Full_train_data["Age"] == 24).values
type(mask_3)

if len(Full_train_data["Age"]) != mask_3.sum() + mask_18.sum() + mask_24.sum():
    raise ValueError("The sum of the masks does not equal the length of the training data. Please check the age values in the training data.")  

    #generate age slices
shap_3m = shap_values[mask_3]
shap_18m = shap_values[mask_18]
shap_24m = shap_values[mask_24]
type(shap_3m)

    #generate mean shap values. Note: to reconstruct a shap explanation object for plotting,
    #the mean for all subarrays have to be calculated: base, data and shap value
    #value means
mean_shap_3m_value = shap_3m.values.mean(axis=0)
mean_shap_18m_value = shap_18m.values.mean(axis=0)
mean_shap_24m_value = shap_24m.values.mean(axis=0)

    #base (background) means
mean_shap_3m_base = shap_3m.base_values.mean(axis=0)
mean_shap_18m_base = shap_18m.base_values.mean(axis=0)
mean_shap_24m_base = shap_24m.base_values.mean(axis=0)

    #data means
mean_shap_3m_data = shap_3m.data.mean(axis=0)
mean_shap_18m_data = shap_18m.data.mean(axis=0)
mean_shap_24m_data = shap_24m.data.mean(axis=0)

    #generating new explanation objects
mean_shap_3m = shap.Explanation(values=mean_shap_3m_value, base_values=mean_shap_3m_base, data= mean_shap_3m_data, 
                                feature_names=shap_3m.feature_names)
mean_shap_18m = shap.Explanation(values=mean_shap_18m_value, base_values=mean_shap_18m_base, data= mean_shap_18m_data, 
                                feature_names=shap_18m.feature_names)
mean_shap_24m = shap.Explanation(values=mean_shap_24m_value, base_values=mean_shap_24m_base, data= mean_shap_24m_data, 
                                feature_names=shap_24m.feature_names)


#waterfall plot of average values 
waterfall_3m, ax= plt.subplots()
ax=shap.plots.waterfall(mean_shap_3m, show=False)
#waterfall_3m.savefig("/Users/michael/Python/Organage/Facs_only/LGB/Results/Analysis/Plots/Waterfall_plot_run1_3m.png")
waterfall_3m.savefig("waterfall_3m.png")
plt.close()

waterfall_18m, ax= plt.subplots()
ax=shap.plots.waterfall(mean_shap_18m, show=False)
#waterfall_18m.savefig("/Users/michael/Python/Organage/Facs_only/LGB/Results/Analysis/Plots/Waterfall_plots_run1_18m.png")
waterfall_18m.savefig("waterfall_18m.png")
plt.close()

waterfall_24m, ax= plt.subplots()
ax=shap.plots.waterfall(mean_shap_24m, show=False)
#waterfall_24m.savefig("/Users/michael/Python/Organage/Facs_only/LGB/Results/Analysis/Plots/Waterfall_plots_run1_24m.png")
waterfall_24m.savefig("waterfall_24m.png")
plt.close()


#beeswarm mplots for shap value to value interactions:
beeswarm_3m, ax = plt.subplots()  
ax=shap.plots.beeswarm(shap_3m, show=False)
#beeswarm_3m.savefig("/Users/michael/Python/Organage/Facs_only/LGB/Results/Analysis/Plots/beeswarm_run1_3m")
beeswarm_3m.savefig("beeswarm_3m.png")
plt.close()

beeswarm_18m, ax = plt.subplots()  
ax=shap.plots.beeswarm(shap_18m, show=False)
#beeswarm_18m.savefig("/Users/michael/Python/Organage/Facs_only/LGB/Results/Analysis/Plots/beeswarm_run1_18m")
beeswarm_18m.savefig("beeswarm_18m.png")
plt.close()

beeswarm_24m, ax = plt.subplots()  
ax=shap.plots.beeswarm(shap_24m, show=False)
#beeswarm_24m.savefig("/Users/michael/Python/Organage/Facs_only/LGB/Results/Analysis/Plots/beeswarm_run1_24m")
beeswarm_24m.savefig("beeswarm_24m.png")
plt.close()


#%%Top 50 features across all sets
#extract 50 most important features globally and for age groups
#global
idx_shap_top_50 = np.argsort(np.abs(shap_values.values.mean(axis=0)))[::-1][:50] #::-1 reverses the array

top_50_genes = X_data_train.columns[idx_shap_top_50]
top_50_values = shap_values.values.mean(axis=0)[idx_shap_top_50]

top_genes = pd.DataFrame({"Top_50_Genes": top_50_genes,                         
                          "Values": top_50_values})
top_genes.to_csv("top_50_genes_global.csv")

#top genes for age groups
idx_shap_top_50_3m = np.argsort(np.abs(mean_shap_3m.values))[::-1][:50] #::-1 reverses the array
idx_shap_top_50_18m = np.argsort(np.abs(mean_shap_18m.values))[::-1][:50] #::-1 reverses the array
idx_shap_top_50_24m = np.argsort(np.abs(mean_shap_24m.values))[::-1][:50] #::-1 reverses the array

#3m
top_50_genes_3m = X_data_train.columns[idx_shap_top_50_3m]
top_50_values_3m = shap_values.values.mean(axis=0)[idx_shap_top_50_3m]

top_genes_3m = pd.DataFrame({"Top_50_Genes": top_50_genes_3m,
                             "Values": top_50_values_3m})
top_genes_3m.to_csv("top_50_genes_3m.csv")

#18m
top_50_genes_18m = X_data_train.columns[idx_shap_top_50_18m]
top_50_values_18m = shap_values.values.mean(axis=0)[idx_shap_top_50_18m]

top_genes_18m = pd.DataFrame({"Top_50_Genes": top_50_genes_18m,
                          "Values": top_50_values_18m})
top_genes_18m.to_csv("top_50_genes_18m.csv")

#24m
top_50_genes_24m = X_data_train.columns[idx_shap_top_50_24m]
top_50_values_24m = shap_values.values.mean(axis=0)[idx_shap_top_50_24m]

top_genes_24m = pd.DataFrame({"Top_50_Genes": top_50_genes_24m,
                          "Values": top_50_values_24m})
top_genes_24m.to_csv("top_50_genes_24m.csv")

#venn diagrams of most important genes 
plt.figure()
venn3((set(top_50_genes_3m), set(top_50_genes_18m), set(top_50_genes_24m)), ("3m", "18m", "24m"))
plt.savefig("venn_top_genes.png")
plt.close()

#%%gene gene interactions

if args.interactions:

    #generate X_df subsets based on age 
    Full_3m = Full_train_data[Full_train_data["Age"]==3]
    Full_18m = Full_train_data[Full_train_data["Age"]==18]
    Full_24m = Full_train_data[Full_train_data["Age"]==24]

    X_3m = Full_3m.drop(columns=["Age"])
    X_18m = Full_18m.drop(columns=["Age"])
    X_24m = Full_24m.drop(columns=["Age"])

    #interaction plots
    #global dependence plots of 20 most important genes

    for i in range(20):
        shap.plots.scatter(shap_values[:, i], color=shap_values, show=False)
        #plt.savefig(f"/Users/michael/Python/Organage/Results/interaction{i}.png")
        plt.savefig(f"global_dependence_plot_gene_{i}.png")
        plt.close()

    #for age groups: calculate top three interaction partners for top 50 genes for each age group
    #Note: approximation with shap.utils.approximate_interactions, gives index of most likely interaction partner

    Interactions_3m = pd.DataFrame()
    for i,j in zip(top_50_genes_3m, range(len(top_50_genes_3m))):
        sorted_idx = shap.utils.approximate_interactions(i, shap_3m.values, X_3m)
        top_idx = sorted_idx[0:3]
        interactors = X_3m.columns[top_idx]
        Interactions_3m.insert(loc=j, column=i, value=interactors)
        #Interactions_3m.to_csv("/Users/michael/Python/Organage/Results/Top_genes_and_interactors_3m.csv")
        Interactions_3m.to_csv("top_50_interactions_3m.csv")

    Interactions_18m = pd.DataFrame()
    for i,j in zip(top_50_genes_18m, range(len(top_50_genes_18m))):
        sorted_idx = shap.utils.approximate_interactions(i, shap_18m.values, X_18m)
        top_idx = sorted_idx[0:3]
        interactors = X_18m.columns[top_idx]
        Interactions_18m.insert(loc=j, column=i, value=interactors)
        #Interactions_18m.to_csv("/Users/michael/Python/Organage/Results/Top_genes_and_interactors_18m.csv")
        Interactions_18m.to_csv("top_50_interactions_18m.csv")

    Interactions_24m = pd.DataFrame()
    for i,j in zip(top_50_genes_24m, range(len(top_50_genes_24m))):
        sorted_idx = shap.utils.approximate_interactions(i, shap_24m.values, X_24m)
        top_idx = sorted_idx[0:3]
        interactors = X_24m.columns[top_idx]
        Interactions_24m.insert(loc=j, column=i, value=interactors)
        #Interactions_24m.to_csv("/Users/michael/Python/Organage/Results/Top_genes_and_interactors_24m.csv")
        Interactions_24m.to_csv("top_50_interactions_24m.csv")

    print("Gene-gene interactions calculated and saved to file.")

else:
    print("pipeline finished successfully")















