#!/usr/bin/env nextflow

/*
 * Nextflow process defining the explanation of the LGB model on healthy patients for biomarker discovery. Optionally calculates the top 50 gene gene interactions.
 */

process explain_healthy {

conda "$projectDir/LGB/Envs/Organage.yaml"

input:
path train_data 
path model

output:
path "*.png", emit: importance_plots  //generates one channel containing all the .png
path "*.csv", emit: top_genes_and_interactions


script:
def interaction_flag = params.interactions ? "--interaction" : ""

"""
Explain_healthy.py --train_data $train_data --model $model ${interaction_flag}
"""

}