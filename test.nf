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
        }.view()
        

        opt_in = in_channel.map{list -> tuple(list[0], list[2])}.view()
        train_in = in_channel.map{list -> tuple(list[0], list[1], list[2])}.view()
        explain_in = in_channel.map{list -> tuple(list[0], list[2])}.view()

    }