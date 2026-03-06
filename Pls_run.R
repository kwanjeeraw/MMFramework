library(metabox2)
library(caret)
library(ropls)
library(pROC)
library(dplyr)
library(optparse)

# -----------------------------
# 00. Parse cmd arguments
# -----------------------------
option_list <- list(
  make_option(
    "--wkpath", type = "character", help = "Working directory path of core codes e.g. /path/to/code_directory (default: getcwd())"
  ),
  make_option(
    "--proc", type = "character", default = "log2", help = "Processing method e.g. log2,raw,auto,minmax (default: log2)"
  ),
  make_option(
    "--input", type = "character", help = "Input file in CSV format e.g. input.csv"
  ),
  make_option(
    "--outpath", type = "character", help = "Output directory path e.g. OUT"
  ),
  make_option(
    "--train", type = "character", help = "Text file containing training indices e.g. TrainList.txt"
  ),
  make_option(
    "--test", type = "character", help = "Text file containing test indices e.g. TestList.txt"
  )
)
opt <- parse_args(
  OptionParser(option_list = option_list),
  positional_arguments = TRUE
)
wkpath  <- opt$options$wkpath
proc  <- opt$options$proc
input <- opt$options$input
outpath <- opt$options$outpath
train <- opt$options$train
test <- opt$options$test

# ---- Checks ----
required_opts <- c("wkpath", "proc", "input", "outpath", "train", "test")
missing_opts <- required_opts[sapply(opt$options[required_opts], function(x) is.null(x) || x == "")]
if (length(missing_opts) > 0) {
  stop("Missing required arguments: ", paste(paste0("--", missing_opts), collapse = ", "),call. = FALSE)
}

# -----------------------------
# 01. Apply arguments
# -----------------------------
setwd(wkpath)
method = proc
if (!dir.exists(outpath)) {
  dir.create(outpath, recursive = TRUE)
}

# -----------------------------
# 0. Initialize
# -----------------------------
source("MMFramework.R") #general functions
filenamepath = paste0(outpath,"/",method,"_plsda_");
################## Setup input ##################
dt = read.csv(input,stringsAsFactors = T) #load input data Index|Class|FeaturesXx
#training sets
trainlines <- readLines(train)
TrainList <- lapply(trainlines, function(x) {
  parts <- strsplit(x, " : ")[[1]]
  as.numeric(strsplit(parts[2], ",\\s*")[[1]])
})
names(TrainList) <- sub(" :.*", "", trainlines)
#test sets
testlines <- readLines(test)
TestList <- lapply(testlines, function(x) {
  parts <- strsplit(x, " : ")[[1]]
  as.numeric(strsplit(parts[2], ",\\s*")[[1]])
})
names(TestList) <- sub(" :.*", "", testlines)

idCol=1; classCol=2; xCol=3; #input data Index|Class|FeaturesXx
binary = ifelse(nlevels(dt[,2]) > 2, FALSE, TRUE) #classification type

## 1. PLS-DA model
set.seed(14)
out_model = list(); out_perform = data.frame(); out_imp = data.frame(); imp_list = list(); stat_list = list();
for(i in 1:length(TrainList)){
  cat("Running ..", names(TrainList)[i], "..")
  #setup data
  inTraining_raw = dt[dt$Index %in% TrainList[[i]],]
  inTraining_raw[] = lapply(inTraining_raw, function(x) if(is.factor(x)) factor(x) else x)
  inTesting_raw = dt[dt$Index %in% TestList[[i]],]
  inTesting_raw[] = lapply(inTesting_raw, function(x) if(is.factor(x)) factor(x) else x)
  inTraining = process.data(inTraining_raw, method=method,idCol,classCol,xCol) #process train
  inTraining[is.na(inTraining)] = 0 #manage NaN
  inTraining[,classCol] = factor(make.names(inTraining[,classCol])) #make valid names
  inTesting = process.data(inTesting_raw, method=method,idCol,classCol,xCol) #process test
  inTesting[is.na(inTesting)] = 0 #manage NaN
  inTesting[,classCol] = factor(make.names(inTesting[,classCol])) #make valid names
  mbobj_train = set_input_obj(inTraining,idCol,classCol,xCol) #Training
  mbobj_test = set_input_obj(inTesting,idCol,classCol,xCol) #Testing
  #calculate
  predmodel = multiv_analyze(mbobj_train,method = "pls", scale = "none") #pls
  if(length(predmodel$vip_val)>0){
    train.pred = predict(predmodel$model, mbobj_train$X) #Training prediction
    train.acc = mean(train.pred == mbobj_train$Y) #Training accuracy
    test.pred = predict(predmodel$model, mbobj_test$X) #Testing prediction
    out.cf = confusionMatrix(test.pred, mbobj_test$Y) #cf - the first level will be used as the positive
    test.numeric = as.numeric(mbobj_test$Y); pred.numeric = as.numeric(test.pred)
    multi.auc = multiclass.roc(test.numeric, pred.numeric) #auc
    if(nlevels(mbobj_train$Y) > 2){#multi-class
      class.stats = as.data.frame(out.cf$byClass)
      class.stats = class.stats %>% mutate(Class = gsub("Class: ","",rownames(out.cf$byClass))) %>%
        select(Class, Sensitivity, Specificity, F1, 'Balanced Accuracy')
      class.stats = class.stats %>% rename(Balanced.Accuracy = 'Balanced Accuracy')
      class.avg = cbind(Class="Average",as.data.frame(t(apply(class.stats[,-1], 2, function(x) {mean(x,na.rm=T)})))) #avg across all classes
      pred_probs = data.frame(matrix(NA, nrow = nrow(mbobj_test$X), ncol = length(levels(mbobj_train$Y))))
      colnames(pred_probs) = levels(mbobj_train$Y)
      auc_class=list();imp_class=list();
      for(class in levels(mbobj_train$Y)) {#this class vs rest
        y_binary = ifelse(mbobj_train$Y == class, class, paste0("not_", class))
        plsda_model = opls(mbobj_train$X,y_binary,orthoI = 0,scaleC = "none",fig.pdfC = 'none') #plsda
        if(length(plsda_model@modelDF) > 0){
          pred = predict(plsda_model, mbobj_test$X)
          prob_class = as.numeric(pred == class)
          pred_probs[[class]] = prob_class
          binary_label = ifelse(mbobj_test$Y == class, 1, 0)
          roc_obj = roc(binary_label, prob_class)
          auc_class[[class]] = auc(roc_obj) #auc
          imp_class[[class]] = plsda_model@vipVn #vip
        }else{
          auc_class[[class]] = 0 #auc
          imp_class[[class]] = setNames(rep(0, ncol(mbobj_train$X)), colnames(mbobj_train$X)) #vip
        }
      }
      auc_df = data.frame(Class=names(auc_class),AUC=unlist(auc_class))
      imp_list[[names(TrainList)[i]]] = bind_rows(imp_class, .id = "Class") #important features
      stat_list[[names(TrainList)[i]]] = merge(class.stats,auc_df,by="Class") #performance per class
    }else{#binary class
      class.avg = as.data.frame(t(out.cf$byClass))
      class.avg = class.avg %>% mutate(Class = out.cf$positive) %>%
        select(Class, Sensitivity, Specificity, F1, 'Balanced Accuracy')
      class.avg = class.avg %>% rename(Balanced.Accuracy = 'Balanced Accuracy')
      imp_list[[names(TrainList)[i]]] = data.frame()
      stat_list[[names(TrainList)[i]]] = data.frame()
    }
    #overall outputs
    out_perform = rbind(out_perform,c(class.avg,
                                      data.frame(Train.Accuracy=train.acc, Test.Accuracy=out.cf$overall["Accuracy"], Overall.AUC=multi.auc$auc)))
    out_imp = rbind(out_imp,t(data.frame(predmodel$vip_val))) #vip
  }else{#if no PC found
    clname = ifelse(binary,levels(mbobj_train$Y)[1],"Average")
    noperform = data.frame(Class=clname,Sensitivity=0,Specificity=0,F1=0,Balanced.Accuracy=0,
                                         Train.Accuracy=0,Test.Accuracy=0,Overall.AUC=0)
    noimp = data.frame(matrix(0, nrow = 1, ncol = ncol(mbobj_train$X)))
    colnames(noimp) = colnames(mbobj_train$X)
    out_perform = rbind(out_perform,noperform)
    out_imp = rbind(out_imp,noimp)
    imp_list[[names(TrainList)[i]]] = data.frame()
    stat_list[[names(TrainList)[i]]] = data.frame()
  }
  out_model[[names(TrainList)[i]]] = predmodel
}
row.names(out_perform) = row.names(out_imp) = names(TrainList)

## 2. Export outputs
export_model_outputs(binary=binary,out_pf=out_perform,out_imt=out_imp,stat_ls=stat_list,imp_ls=imp_list,
                     filename=filenamepath,width=9.5,height=(8+ncol(out_imp)*0.03))
cat("#### DONE!! ####\n")