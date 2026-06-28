#!/usr/bin/env nextflow

include { optimise } from "./LGB/1_Optimisation/Optimisation.nf"
include { train } from "./LGB/2_Two_step_training/Train.nf"
include { explain_healthy } from "./LGB/3_Explain_healthy/Explain_healthy.nf"

/*
 *Pipeline Parameters
 */

params.train_data = "$baseDir/Data/Model_train_data/Train_data.csv"
params.test_data = "$baseDir/Data/Model_train_data/Test_data.csv"
params.ranges = "$baseDir/Data/Ranges.csv"
params.num_attributions = 50
params.num_genes_int = null
params.num_int = null
    
    workflow {
      	main:
        in_channel = channel.fromPath(params.train_data)

        optimise(in_channel)

            test_channel = channel.fromPath(params.test_data)

        train(in_channel, optimise.out.best_params, test_channel)

            ranges_channel = channel.fromPath(params.ranges) 
            params_channel = channel.value([params.num_attributions, params.num_genes_int, 
            params.num_int])
            
        explain_healthy(in_channel, train.out.model, ranges_channel, params_channel)

    	publish:
        optimise_out = optimise.out[0].mix(optimise.out[1],
            optimise.out[2],
            optimise.out[3])     //mix generates one channel from multiple output channels, Ech output specified in a process is its own channel apparently.

        train_out = train.out[0].mix(train.out[1],
            train.out[2],
            train.out[3],
            train.out[4],)

        explain_healthy_out = explain_healthy.out[0].mix(explain_healthy.out[1])

    }
    output {
      optimise_out {path "LGB/1_Optimisation"}
      train_out {path "LGB/2_Two_step_training"}
      explain_healthy_out {path "LGB/3_Explain_healthy"}

    }