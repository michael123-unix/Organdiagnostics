# Description
Work in progress - follow up project from the Organage_prequal repository. This project utilized the gradient boosted decision tree model lightGBM to infer age based on gene or protein data. The nextflow- directed pipeline includes model hyperparameter optimisation, a two step training process and model explenation for relevant biomarker extraction employing the SHAP framework. Further downstream modules for calculating organ specific age acceleration on diseased individual will be added:  
+ Age gap calculation for diseased individuals
+ Multi- group disease risk analysis based on age gaps
+ Analysis of disease and age associated biomarkers of high risk individuals 
+ Biomarker - disease hallmark matching to further test their association
