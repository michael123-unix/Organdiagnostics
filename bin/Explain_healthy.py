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
from matplotlib_venn import venn3, venn2

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
    parser.add_argument("--ranges", help="Path to a file specifying age ranges for analysis.", required=True)
    parser.add_argument("--num_attributions", help="Number of top feature attributions to extract as csv file", required=True, type= int)
    parser.add_argument("--num_genes_int", help = "number of genes for which to extract interactions. Number has to be smaller or equal to the number of extracted top attributions", type=int)
    parser.add_argument("--num_int", help= "number or interaction parterns to extract for a given gene", type=int)
    args = parser.parse_args()

#%%load data
print("loading and preparing data and model")

train_data = pd.read_csv(args.train_data)
ranges = pd.read_csv(args.ranges)

train_data.index = train_data["index"]
train_data= train_data.drop(columns="index")
train_data.columns

reduce_mem_usage(train_data)

#prepare the train data for generating attributions and interactions
Y_train = train_data.pop("Age")
X_data_train = train_data
Full_train_data = X_data_train.assign(Age = Y_train).copy()

#load model
gbm= lgb.Booster(model_file=args.model)

#%%shap values 
print("claculating shap values")

explainer = shap.TreeExplainer(gbm)
shap_values = explainer(X_data_train)

print("shap values have been calculated")

#%%subset data based on age windows
print("prepare data for age window specific analysis")

age_window_frames = {}
for i,r in ranges.iterrows():
    name = r["Name"]
    Lower = r["Lower"]
    Upper = r["Upper"]
    mask = Full_train_data["Age"].between(Lower, Upper, inclusive="both").values

    age_window_frames[name] = shap_values[mask]

length = 0
for i in age_window_frames.values():
    length += i.shape[0]

if length == Full_train_data.shape[0]:
    print("data was split successfully")

#%% calculate means over all patients

mean_shaps_by_age_window = {}
for n,i in zip(age_window_frames.keys(), age_window_frames.values()):
    value_mean = i.values.mean(axis=0)
    base_mean = i.base_values.mean(axis=0)
    data_mean = i.data.mean(axis=0)
    shap_mean = shap.Explanation(values=value_mean, base_values=base_mean, 
                                 data = data_mean, feature_names = i.feature_names)
    mean_shaps_by_age_window[n] = shap_mean

#%% Waterfall plots
print("generate waterfall plot for age groups")

for n,i in zip(mean_shaps_by_age_window.keys(), mean_shaps_by_age_window.values()):
               waterfall_plot, ax=plt.subplots()
               ax = shap.plots.waterfall(i, show=False)
               plt.savefig(f"waterfall_plot_{n}.png")
               plt.close()

#%%beeswarm plots
print("generate beewsarm plots for age groups")

for n,i in zip(age_window_frames.keys(), age_window_frames.values()):
               beeswarm_plot, ax=plt.subplots()
               ax = shap.plots.beeswarm(i, show=False)
               plt.savefig(f"beeswarm_plot_{n}.png")
               plt.close()

#%%Top k attributions global:
print("extracting top k attributions")

idx_shap_top_50 = np.argsort(np.abs(shap_values.values.mean(axis=0)))[::-1][:50] #::-1 reverses the array

top_50_genes = X_data_train.columns[idx_shap_top_50]
top_50_values = shap_values.values.mean(axis=0)[idx_shap_top_50]

top_genes = pd.DataFrame({"Top_50_Genes": top_50_genes,                         
                          "Values": top_50_values})
top_genes.to_csv("top_50_genes_global.csv")

#%%top k attributions for age groups
num_attributions = args.num_attributions
top_k_attributions = {}
for n,i in zip(mean_shaps_by_age_window.keys(), mean_shaps_by_age_window.values()):
     idx = np.argsort(np.abs(i.values))[::-1][:num_attributions]
     top_genes = X_data_train.columns[idx]
     top_values = i.values[idx]
     top_k_attributions[n] = pd.DataFrame({"gene": top_genes,
                                           "value": top_values})
     top_k_attributions[n].to_csv(f"Top_{num_attributions}_attributions_{n}.csv")
     
#%%venn diagram, extract feature sets
print("generate venn diagrams of top features for age groups")

no_venns = sum("Venn" in col for col in ranges.columns)
sets_venn = {}
for i in range(1,(no_venns+1)):
    set_names = list(ranges[ranges[f"Venn_{i}"]=="y"]["Name"])
    venn_subsets = {}
    for j in set_names:
        gene_set = set(top_k_attributions[j]["gene"])
        venn_subsets[j] = gene_set
    sets_venn[f"venn_{i}"] = venn_subsets

#%%venn diagram, plot

for i in range(1,no_venns+1):
    subsets = tuple(sets_venn[f"venn_{i}"].values())
    labels = tuple(sets_venn[f"venn_{i}"].keys())
    plt.figure()
    if len(subsets) == 3:
         venn3(subsets, labels)
    elif len(subsets) == 2:
         venn2(subsets, labels)
    else:
         print("wrong number of sets provided for matplotlib-venn. use either 2 or 3 sets for each plot")
    plt.savefig(f"venn_{i}.png")
    plt.close()              

#%%gene gene interactions

if args.num_genes_int:
    print("Start analyzing feature interactions")


#%%dependence plot for top 10 features globally
    print("generating shap dependence- plots")

    for i in top_genes["Top_50_Genes"][0:9]:
        shap.plots.scatter(shap_values[:, i], color=shap_values, show=False)
        plt.savefig(f"global_dependence_plot_gene_{i}.png")
        plt.close()

#%%dependence plots for age groups

    for n in age_window_frames.keys():
         for i in top_k_attributions[n]["gene"][0:9]:
            shap.plots.scatter(age_window_frames[n][:,i], color = age_window_frames[n], show=False)
            plt.savefig(f"dependence_plot_{i}_{n}")
            plt.close()

    #%% generate X_df subsets based on age 

    X_data_for_interactions = {}
    for i,r in ranges.iterrows():
        name = r["Name"]
        Lower = r["Lower"]
        Upper = r["Upper"]
        mask = Full_train_data["Age"].between(Lower, Upper, inclusive="both").values

        X_data_for_interactions[name] = Full_train_data[mask].drop(columns = "Age")

#%% extract top k interactions
    print("extracting top k interactions")
    int_num = args.num_int
    num_for_int = args.num_genes_int 
    interactions_by_age = {}
    for n in age_window_frames.keys():
         
        age_window = []
        for g in top_k_attributions[n]["gene"][0:num_for_int]:
            sorted_idx = shap.utils.approximate_interactions(g, age_window_frames[n].values, 
                                                             X_data_for_interactions[n])
            top_5_idx = sorted_idx[0:int_num]
            interactors = X_data_for_interactions[n].columns[top_5_idx]
            row_data = {"gene":g}

            for i, j in enumerate(interactors):
                 row_data[f"interactor_{i}"] = j
            
            age_window.append(row_data)
        
        df = pd.DataFrame(age_window)
        interactions_by_age[n] = df
        df.to_csv(f"top_{int_num}_interactions_{n}.csv")

    print("pipeline finished successfully")

else:
    print("pipeline finished successfully")















