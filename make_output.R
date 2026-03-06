##### SET WORKING DIRECTORY TO MMFRAMEWORK #####
setwd("~/Documents/siri/nonImage/production/MMFramework/")

##### LOAD FRAMEWORK #####
source("MMFramework.R")

# --------------------------------------------------------
# 1. COMPUTE ModelPerformance.csv for each input type
# --------------------------------------------------------
compute_model_performance(folder_path="LN/integrated") #for integrated data
compute_model_performance(folder_path="LN/clinical") #for clinical data
compute_model_performance(folder_path="LN/metabolomics") #for metabolomics data

## For other data, if exist
#compute_model_performance(folder_path="LN/DataX") #for other data

# --------------------------------------------------------
# 2. Plot comparative performance heatmap
# --------------------------------------------------------
input_list = c("I"="LN/integrated","C"="LN/clinical","M"="LN/metabolomics")
plot_comparative_heatmap(input_list, title="Control vs LN")

# --------------------------------------------------------
# 3. Plot cross-modal rank comparison
# --------------------------------------------------------
input_list = c("I"="LN/integrated","C"="LN/clinical","M"="LN/metabolomics")
featureFile = "log2_rf_feature.csv"
plot_cross_modal_rank(input_list, featureFile, title="Control vs LN")

# --------------------------------------------------------
# 4. Plot combined rankings
# --------------------------------------------------------
input_list = c("LN/integrated/auto_plsda_feature.csv","LN/clinical/auto_plsda_feature.csv","LN/metabolomics/auto_plsda_feature.csv")
combine_ranking(input_list, n_perm=1000, out_folder="LN")

# --------------------------------------------------------
# 5. Plot accuracy curves across training rounds
# --------------------------------------------------------
folder_path = "LN/clinical"
plot_cross_model_accuracy(folder_path, title="Control vs LN (Clinical modal)")

# --------------------------------------------------------
# 6. Plot model performance comparison
# --------------------------------------------------------
folder_path = "LN/clinical/"
plot_cross_model_performance(folder_path, title="Control vs LN (Clinical modal)")
