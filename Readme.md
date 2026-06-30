# Description
Work in progress - follow up project from the Organage_prequal repository. This project utilized the gradient boosted decision tree model lightGBM to infer age based on gene or protein data. The nextflow- directed pipeline includes model hyperparameter optimisation, a two step training process and model explenation for relevant biomarker extraction employing the SHAP framework. Further downstream modules for calculating organ specific age acceleration on diseased individual will be added:  
+ Age gap calculation for diseased individuals
+ Multi- group disease risk analysis based on age gaps
+ Analysis of disease and age associated biomarkers of high risk individuals 
+ Biomarker - disease hallmark matching to further test their association

# Usage
## Input:
* Sample_sheet.csv : contains Organ, Train_data (Path to train data), Test_data(Path to test data). See example in /Data
* Ranges.csv: contains Name, Lower, Upper, Venn_1... where 'Name' identifies the age range, 'Lower' and 'Upper' contain integers defining the age ranges for analysis and 'Venn' contains either n or y, where y includes the age range into a comparison with one or two other ranges regarding the overlap of top features. Only two and three group comparisons are supported. Several different venn diagrams can be made, just enter Venn_1 in the first and Venn_2 in the subsequent column. Example: see /Data
* Train and Test data: must be present in the location described in the sample sheet.
* Parameters:
    * num_attributions: number of top features to write to a csv for each age window
    * num_genes_int: optional, number of top attributions for which to calculate interactions, must be <= to num_attributions
    * num_int: optional, number of interactions to calculate for each gene


## Example usage
nextflow run main.nf --samples Data/Sample_sheet.csv --ranges Data/Ranges.csv --num_attributions = 50