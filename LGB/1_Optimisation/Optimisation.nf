#!/usr/bin/env nextflow

/*
 * This is a Nextflow script defining the optimisation process for LGB
 */

 process optimise {

    conda "$projectDir/LGB/Envs/Organage.yaml" 

     input:
     tuple val(organ), path(train_data)

     output:
      path("param_importances_${organ}.png")
      path("opt_history_${organ}.png")
      path("best_params_${organ}.csv")   
      tuple val(organ), path("final_best_params_${organ}.json"), emit: best_params

     script:
     """
    Optimisation.py --train_data ${train_data} --organ ${organ}
     """

 }