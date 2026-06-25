# Description
Work in progress - follow up project from the Organage_prequal repository. This project utilized the gradient boosted desition tree model lightGBM to infer age to a target dataset. The nextflow- directed pipeline includes model hyperparameter optimisation, a two step training process and model explenation for relevant biomarker extraction employing the SHAP framework. Further downstream modules for calculating organ specific age acceleration on diseased individual will be added:  
+ Age gap calculation for diseased individuals
+ Multi- group disease risk analys based on age gaps
+ Analysis of disease and age associated biomarkers of high risk individuals 
+ Biomarker - disease hallmark matching to further test their association
