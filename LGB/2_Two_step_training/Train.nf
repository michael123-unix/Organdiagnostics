#!/usr/bin/env nextflow

/*
 * Defenition of the train process for LGB, two models are trained, 
 * the second on a dataset which excludes all samples with a prediction error exceeding a 
 * z score of 2
 */

 process train {

conda "$projectDir/LGB/Envs/Organage.yaml"

input:
tuple path(train_data), path(test_data)
path best_params


output:
path "model.txt", emit: model
path "model.json"
path "model_metrics_m1.png"
path "model_metrics_m2.png"
path "True_vs_predicted_age.png"

script:
"""
Training.py --train_data ${train_data} --best_params ${best_params} --test_data ${test_data}
"""

 }