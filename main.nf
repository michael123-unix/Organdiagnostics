#!/usr/bin/env nextflow

include { optimise } from "./LGB/1_Optimisation/Optimisation.nf"
include { train } from "./LGB/2_Two_step_training/Train.nf"
include { explain_healthy } from "./LGB/3_Explain_healthy/Explain_healthy.nf"

/*
 *Pipeline Parameters
 */

params.samples = "$baseDir/Data/Sample_sheet.csv"
params.ranges = "$baseDir/Data/Ranges.csv"
params.num_attributions = 50
params.num_genes_int = null
params.num_int = null
    
    workflow {
      	main:
        in_channel = channel.fromPath(params.samples).splitCsv(header: true)
        .map{row -> [file(row.Train_data), file(row.Test_data), row.Organ]
        }

        opt_in = in_channel.map{list -> list[0]}
        train_in = in_channel.map{list -> tuple(list[0], list[1])}
        explain_in = in_channel.map{list -> list[0]}

        optimise(opt_in)


        train(train_in, optimise.out.best_params)

            ranges_channel = channel.fromPath(params.ranges) 
            params_channel = channel.value([params.num_attributions, params.num_genes_int, 
            params.num_int])
            
        explain_healthy(explain_in, train.out.model, ranges_channel, params_channel)

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