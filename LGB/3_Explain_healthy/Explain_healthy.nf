#!/usr/bin/env nextflow

/*
 * Nextflow process defining the explanation of the LGB model on healthy patients for biomarker discovery. Optionally calculates the top 50 gene gene interactions.
 */

process explain_healthy {

conda "$projectDir/LGB/Envs/Organage.yaml"

input:
tuple val(organ), path(train_data), path(model)
path ranges
tuple val(num_attributions), val(num_genes_int), val(num_int)


output:
path "*_${organ}.png", emit: importance_plots  //generates one channel containing all the .png
path "*_${organ}.csv", emit: top_genes_and_interactions


script:
def genes_arg = num_genes_int ? "--num_genes_int $num_genes_int" : ""
def int_arg = num_int ? "--num_int $num_int" : ""

"""
Explain_healthy.py --train_data $train_data --model $model --ranges $ranges \
--num_attributions $num_attributions $genes_arg $int_arg --organ ${organ}
"""

}