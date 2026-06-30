#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 13:21:47 2026

@author: michael
"""

#%% import libraries
import argparse
import copy
import json
import pickle
import graphviz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
import os
import seaborn as sn

#%%
#def function to reduce memory usage
def reduce_mem_usage(df):
    for col in df.columns:
        if df[col].dtype == "float64":
            df[col]= df[col].astype(np.float32)
        if df[col].dtype == "int64":
            df[col]=df[col].astype(np.int32)
    return(df)

print("Loading and preparing data...")
# load or create dataset

#%% define arguments
if __name__== "__main__":
    parser = argparse.ArgumentParser(description="Train a LightGBM model on the provided dataset.")
    parser.add_argument("--train_data", help="Path to the training data CSV file", required=True)
    parser.add_argument("--best_params", help="Path to the JSON file containing the best hyperparameters", required=True)   
    parser.add_argument("--test_data", help="Path to the evaluation data CSV file", required=True)
    parser.add_argument("--organ", type=str, help="Organ name for naming the output")

    args = parser.parse_args()

#Prepare data
df = pd.read_csv(args.train_data)

#%%remove index column
df.index = df["index"]
df_1 = df.drop(columns="index")

reduce_mem_usage(df_1)

#check if first column is "Age"
if df_1.columns[0] != "Age":
    raise ValueError("The first column of the dataframe must be 'Age'.")   

#train test split

np.random.seed(42)
indices = df_1.index.to_list()
np.random.shuffle(indices)
indices_train = indices[0:int(len(indices)*0.8)]
indices_test = indices[int(len(indices)*0.8):len(indices)]
len(indices_test)

df_train = df_1.loc[indices_train,:]
df_train.shape
df_train.columns
df_test = df_1.loc[indices_test,:]

y_train = df_train["Age"]
y_test = df_test["Age"]
X_train = df_train.drop("Age", axis=1)
X_test = df_test.drop("Age", axis=1)

# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

#defining the static parameters
static_params = {
   "objective": "regression",
   "metric": ["l1","rmse"],
   "verbosity": -1,
   "boosting_type": "gbdt",
   "random_state": 42}

#%%load the best hyperparameters from the JSON file

with open(args.best_params, "r") as f:
    opt_params = json.load(f)

#with open("/Users/michael/Python/Organage/work/2c/b33b8e4d10f4ca37ce32ebd90c81c4/final_best_params.json", "r") as f:
    #opt_params = json.load(f)

params = {**static_params, **opt_params}

#-
#for recording evaluation results
    #this generates an emty dictionary for the callback function reccording the metrics
    #a callback function runs at the end of each itteration
    #the callback function "record_evaluation" collects evaluation metrics for each itteration
    #the recorded metrics are the ones specified in the params under "metric", more than one are possible such as mae and rmse

evals_result_m1 = {} 
#-

#%% train

    #Note: log_evaluation prints the evaluation metric to the console for live monitoring
print("Starting training model 1...")
gbm = lgb.train(
    params, lgb_train, num_boost_round=200, valid_sets=lgb_eval, 
    callbacks=[lgb.early_stopping(stopping_rounds=5), lgb.log_evaluation(10), lgb.record_evaluation(evals_result_m1)]
)

feature_importance_m1 = list(gbm.feature_importance())
best_mae_m1 = gbm.best_score
best_iter_m1 = gbm.best_iteration


#%%plot parameter importances
print("Plotting metrics recorded during training...")
fig,ax = plt.subplots(2,2,figsize= (15,12))

lgb.plot_metric(evals_result_m1, metric="l1",ax=ax[0,0])
lgb.plot_importance(gbm, importance_type="gain", max_num_features=10, ax=ax[0,1])
lgb.plot_split_value_histogram(gbm, feature="Lars2", bins="auto", ax =ax[1,0])
lgb.plot_tree(gbm, tree_index=1, figsize=(15, 15), show_info=["split_gain"], ax=ax[1,1])

ax[0,0].set_title("Error Reduction")
ax[0,0].set_xlabel("Iteration")
ax[0,0].set_ylabel("PRMSE")

ax[0,1].set_title(" Feature importances")
ax[0,1].set_xlabel("total gain")
ax[0,1].set_ylabel("Feature")

ax[1,0].set_title("Split Values")
ax[1,0].set_xlabel("Feature split value Lars2")
ax[1,0].set_ylabel("Clunt")

ax[1,1].set_title("Tree 1")
fig.savefig(f"model_metrics_m1_{args.organ}.png") 


#%% predict on train set and correct data
Age_all_data = df_1.pop("Age")

preds = gbm.predict(df_1)
Full_data = df_1.copy().assign(Age = Age_all_data, Age_predicted = preds)
Full_data["Error"] = Full_data["Age"] - Full_data["Age_predicted"]
Full_data["z_score_error"] = (Full_data["Error"]- np.mean(Full_data["Error"]))/np.std(Full_data["Error"])
                              
Corrected_data = Full_data[abs(Full_data["z_score_error"])<2]

#%%retrain on the corrected data

np.random.seed(42)
indices_2 = Corrected_data.index.to_list()
np.random.shuffle(indices_2)
indices_train_2 = indices_2[0:int(len(indices_2)*0.8)]
indices_test_2 = indices_2[int(len(indices_2)*0.8):len(indices_2)]

df_train_2 = Corrected_data.loc[indices_train_2,:]
df_test_2 = Corrected_data.loc[indices_test_2,:]

y_train_2 = df_train_2["Age"]
y_test_2 = df_test_2["Age"]
X_train_2 = df_train_2.drop(["Age", "Age_predicted", "Error", "z_score_error"], axis=1)
X_test_2 = df_test_2.drop(["Age", "Age_predicted", "Error", "z_score_error"], axis=1)

# create dataset for lightgbm
lgb_train_2 = lgb.Dataset(X_train_2, y_train_2)
lgb_eval_2 = lgb.Dataset(X_test_2, y_test_2, reference=lgb_train_2)

#%%
evals_result_m2 = {}

print("Starting training model 2...")
gbm_2 = lgb.train(
    params, lgb_train_2, num_boost_round=200, valid_sets=lgb_eval_2, 
    callbacks=[lgb.early_stopping(stopping_rounds=5), lgb.log_evaluation(10), lgb.record_evaluation(evals_result_m2)]
)

feature_importance_m2 = list(gbm_2.feature_importance())
best_mae_m2 = gbm_2.best_score
best_iter_m2 = gbm_2.best_iteration

# save model to file
print("Saving model...")
gbm_2.save_model(f"model_{args.organ}.txt")

with open(f"model_{args.organ}.json", "w") as f:
   json.dump(gbm_2.dump_model(), f)

#%%plot parameter importances
print("Plotting metrics recorded during training model_2...")
fig,ax = plt.subplots(2,2,figsize= (15,12))

lgb.plot_metric(evals_result_m2, metric="l1",ax=ax[0,0])
lgb.plot_importance(gbm_2, importance_type="gain", max_num_features=10, ax=ax[0,1])
lgb.plot_split_value_histogram(gbm_2, feature="Lars2", bins="auto", ax =ax[1,0])
lgb.plot_tree(gbm_2, tree_index=1, figsize=(15, 15), show_info=["split_gain"], ax=ax[1,1])

ax[0,0].set_title("Error Reduction")
ax[0,0].set_xlabel("Iteration")
ax[0,0].set_ylabel("PRMSE")

ax[0,1].set_title(" Feature importances")
ax[0,1].set_xlabel("total gain")
ax[0,1].set_ylabel("Feature")

ax[1,0].set_title("Split Values")
ax[1,0].set_xlabel("Feature split value Lars2")
ax[1,0].set_ylabel("Count")

ax[1,1].set_title("Tree 1")
fig.savefig(f"model_metrics_m2_{args.organ}.png") 

#%%evaluate the model on independent evaluation data
print("Training complete. Start evaluation on provided evaluation data...")

eval_data = pd.read_csv(args.test_data)

eval_data.index = eval_data["index"]
eval_data = eval_data.drop(columns="index")
if eval_data.columns[0] != "Age":
    raise ValueError("The first column of the evaluation dataframe must be 'Age'.")

y_eval = eval_data["Age"]
X_eval = eval_data.drop(columns="Age")

preds_eval = gbm_2.predict(X_eval)
rmse_eval = (np.sqrt(sum((preds_eval - y_eval)**2)/len(preds_eval)))
print(f"RMSE on evaluation data: {rmse_eval}")

#%%save full data
Full_eval_data = eval_data.assign(Age = y_eval, Age_predicted = preds_eval)

#%% plot true vs predicted age 
ages = [Full_eval_data[Full_eval_data["Age"]==i]["Age_predicted"] for i in (3,18,24)]

#%%
plt.boxplot(ages)
plt.title("Predicted vs true ages")
plt.xlabel("True age")
plt.ylabel("predicted ages")
plt.text(1,25, f"RMSE = {round(rmse_eval,2)}")
plt.savefig(f"True_vs_predicted_age_{args.organ}.png")
plt.close

print("Pipeline finished successfully!")

