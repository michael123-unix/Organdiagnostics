#!/usr/bin/env nextflow

/*
 * This is a Nextflow script defining the optimisation process for LGB
 */

 process optimise {

    conda "$projectDir/LGB/Envs/Organage.yaml" 

     input:
     path train_data

     output:
     path "param_importances.png"
     path "opt_history.png"
     path "best_params.csv"      
     path "final_best_params.json", emit: best_params

     script:
     """
    Optimisation.py --train_data ${train_data}
     """

 }