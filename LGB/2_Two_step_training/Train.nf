#!/usr/bin/env nextflow

/*
 * Defenition of the train process for LGB, two models are trained, 
 * the second on a dataset which excludes all samples with a prediction error exceeding a 
 * z score of 2
 */

 process train {

conda "$projectDir/LGB/Envs/Organage.yaml"

input:
tuple val(organ), path(train_data), path(test_data), path(best_params)

output:
tuple val(organ), path("model_${organ}.txt"), emit: model
path "model_${organ}.json"
path "model_metrics_m1_${organ}.png"
path "model_metrics_m2_${organ}.png"
path "True_vs_predicted_age_${organ}.png"

script:
"""
Training.py --train_data ${train_data} --best_params ${best_params} --test_data ${test_data} --organ ${organ}
"""

 }