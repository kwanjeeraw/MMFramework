import pandas as pd
import numpy as np
import seaborn as sns
import shap
import os
import random
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, label_binarize
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, ReLU, Dropout
from tensorflow.keras.utils import to_categorical
import sys
import argparse
import MMFramework

# -----------------------------
# 00. Parse cmd arguments
# -----------------------------
parser = argparse.ArgumentParser(description="Run CNN SHAP")

parser.add_argument(
    "--wkpath",type=str,default="",help="Working directory path of core codes e.g. /path/to/code_directory (default: os.getcwd())"
)
parser.add_argument(
    "--proc",type=str,default="log2",help="Processing method e.g. log2,raw,auto,minmax (default: log2)"
)
parser.add_argument(
    "--input",type=str,default="input.csv",help="Input file in CSV format e.g. input.csv"
)
parser.add_argument(
    "--outpath",type=str,default="OUT",help="Output directory path e.g. OUT"
)
parser.add_argument(
    "--train",type=str,default="TrainList.txt",help="Text file containing training indices e.g. TrainList.txt"
)
parser.add_argument(
    "--test",type=str,default="TestList.txt",help="Text file containing test indices e.g. TestList.txt"
)
parser.add_argument(
    "--epoch",type=int,default=200,help="Number of training epochs"
)
parser.add_argument(
    "--batch",type=int,default=32,help="Number of training batch size"
)
args = parser.parse_args()

# -----------------------------
# 01. Apply arguments
# -----------------------------
sys.path.append(args.wkpath)
pmethod = args.proc
outpath = args.outpath
os.makedirs(outpath, exist_ok=True)
epochs = args.epoch
batchs = args.batch

# -----------------------------
# 0. Initialize
# -----------------------------
#input
input_df = pd.read_csv(args.input) #load input data Index|Class|FeaturesXx
input_excluded = input_df #setup input
#train/test sets
trainlist_dict = MMFramework.read_list_file(args.train)
testlist_dict = MMFramework.read_list_file(args.test)

filenamepath = f"{outpath}/{pmethod}_shap_"
CNN_model = {}
CNN_model_per_class = {}
CNN_eval = {}
CNN_feature = {}
CNN_feature_per_class = {}

    # -----------------------------
    # 1. Load Data
    # -----------------------------
for m in range(len(trainlist_dict)):

    dataset_name = list(trainlist_dict.keys())[m]
    print("###Loading data: ",dataset_name)
    train_indices = list(trainlist_dict.values())[m]
    train_df = input_excluded[input_excluded['Index'].isin(train_indices)] #train df: Index|Class|FeaturesXx
    test_indices = list(testlist_dict.values())[m]
    test_df = input_excluded[input_excluded['Index'].isin(test_indices)] #test df: Index|Class|FeaturesXx

    # Select features and Class column
    X_train = train_df.drop(train_df.columns[[0, 1]], axis=1).values #train
    y_train = train_df["Class"].values
    X_test = test_df.drop(test_df.columns[[0, 1]], axis=1).values #test
    y_test = test_df["Class"].values
    num_ft = X_train.shape[1]
    
    # Encode target labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    # Initial setup
    feature_names = list(train_df.columns[2:])
    num_features = len(feature_names)
    ls_classes = np.unique(y_test)
    num_classes = len(np.unique(y_train_enc))

    # Convert to one-hot encoded matrix
    y_train_cat = to_categorical(y_train_enc, num_classes)
    y_test_cat = to_categorical(y_test_enc, num_classes)

    # -----------------------------
    # 2. Process features
    # -----------------------------
    if pmethod == "raw":
        # No scale
        X_train_scaled = X_train
        X_test_scaled = X_test
    elif pmethod == "auto":
        # Auto scaling
        mscaler = StandardScaler()
        X_train_scaled = mscaler.fit_transform(X_train)
        X_test_scaled = mscaler.fit_transform(X_test)
    elif pmethod == "minmax":
        # Min-Max
        mscaler = MinMaxScaler()
        X_train_scaled = mscaler.fit_transform(X_train)
        X_test_scaled = mscaler.fit_transform(X_test)
    else:
        # Log2 transformation
        X_train_scaled = np.log2(X_train)
        X_test_scaled = np.log2(X_test)

    # -----------------------------
    # 3. Convert tabular data for 1D CNN
    # -----------------------------
    hmsize = math.ceil(math.sqrt(num_ft)) #a heatmap of size AxA
    X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0],X_train_scaled.shape[1],1)
    X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0],X_test_scaled.shape[1],1)

    # ---------------------------
    # 4. Build 1D CNN -- 2 layers
    # ---------------------------
    seed = 14
    MMFramework.set_all_seeds(seed)

    print("###Calculating CNN: ","...")
    inputs = Input(shape=(X_train_cnn.shape[1], 1))
    x = Conv1D(32, kernel_size=2, padding='same')(inputs)
    x = ReLU()(x)
    x = MaxPooling1D(pool_size=2, padding='same')(x)
    x = Conv1D(64, kernel_size=2, padding='same')(x)
    x = ReLU()(x)
    x = MaxPooling1D(pool_size=2, padding='same')(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    # -----------------------------
    # 5. Train Model
    # -----------------------------
    cnn_model = model.fit(X_train_cnn, y_train_cat,validation_split=0.3,epochs=epochs,batch_size=batchs) #about 30% validation

    # ---------------------------
    # 6. Evaluate model
    # ---------------------------
    y_pred_prob = model.predict(X_test_cnn) #predicted probabilities
    y_pred = np.argmax(y_pred_prob, axis=1) #predicted classes

    # Confusion matrix
    cfmatrix = confusion_matrix(y_test_enc, y_pred)
    TP = np.diag(cfmatrix)
    FP = cfmatrix.sum(axis=0) - TP
    FN = cfmatrix.sum(axis=1) - TP
    TN = cfmatrix.sum() - (FP + FN + TP)

    # ---------------------------
    # Per-class metrics
    # ---------------------------
    # Accuracy
    acc_per_class = (TP + TN) / cfmatrix.sum()
    # Sensitivity
    sensitivity_per_class = recall_score(y_test_enc, y_pred, average=None)
    # Specificity
    specificity_per_class = []
    for i in range(num_classes):
        tn = cfmatrix.sum() - (cfmatrix[i,:].sum() + cfmatrix[:,i].sum() - cfmatrix[i,i])
        fp = cfmatrix[:,i].sum() - cfmatrix[i,i]
        specificity = tn / (tn + fp)
        specificity_per_class.append(specificity)

    # F1
    f1_per_class = f1_score(y_test_enc, y_pred, average=None)

    # ---------------------------
    # Overall metrics
    # ---------------------------
    # Accuracy
    acc = accuracy_score(y_test_enc, y_pred)
    # Sensitivity
    sensitivity_macro = recall_score(y_test_enc, y_pred, average='macro')
    # Specificity
    specificity_macro = sum(specificity_per_class)/len(specificity_per_class)
    # F1
    f1_macro = f1_score(y_test_enc, y_pred, average='macro')

    # AUC
    if num_classes > 2:#Multi-class
        auc_macro = roc_auc_score(y_test, y_pred_prob, multi_class='ovr', average='macro')
        auc_per_class = roc_auc_score(y_test, y_pred_prob, multi_class='ovr', average=None)
    else:#Binary class
        auc_macro = roc_auc_score(y_test_enc, y_pred_prob[:,1], average=None)
        auc_per_class = np.repeat(auc_macro, 2)

    # Collect performance metrics
    metrics_dict = {#average
        "Accuracy": [acc],
        "Sensitivity": [sensitivity_macro],
        "Specificity": [specificity_macro],
        "F1-score": [f1_macro],
        "AUC": [auc_macro]
    }
    metrics_dict_per_class = {#per_class
        "Accuracy": [acc_per_class],
        "Sensitivity": [sensitivity_per_class],
        "Specificity": [specificity_per_class],
        "F1-score": [f1_per_class],
        "AUC": [auc_per_class]
    }
    cnn_metrics = pd.DataFrame(metrics_dict) #overall performance
    # Collect accuracy for each epoch
    epoch_metrics = pd.DataFrame(cnn_model.history)
    epoch_metrics['epoch'] = range(1, len(epoch_metrics) + 1)
    epoch_metrics = epoch_metrics.iloc[:, [4,0,1,2,3]] #train-test performance

    # Collect performacne per class
    cnn_metrics_per_class = pd.DataFrame({
        metric: values[0] for metric, values in metrics_dict_per_class.items()
    }, index=ls_classes).T
    cnn_metrics_per_class = cnn_metrics_per_class.reset_index().rename(columns={"index": "Metric"})

    # -----------------------------
    # 7. Compute SHAP
    # -----------------------------
    explainer = shap.GradientExplainer(model, X_train_cnn)
    shap_values_list = explainer.shap_values(X_train_cnn) #shap scores from the training set
    shap_flat = shap_values_list.reshape(len(shap_values_list), num_features, num_classes) #from the training set
    shap_values_list_t = explainer.shap_values(X_test_cnn) #shap scores from the test set
    shap_flat_t = shap_values_list_t.reshape(len(shap_values_list_t), num_features, num_classes) #from the test set

    # Average overall shap scores across samples
    avg_importance = np.mean(np.abs(shap_flat), axis=(0,2))
    shap_scores = pd.DataFrame({
        "Feature": feature_names,
        "Feature_Importance": avg_importance
    }).sort_values("Feature_Importance", ascending=False)

    # Average shap scores per class
    shap_df = np.mean(np.abs(shap_flat), axis=0)
    shap_per_class = pd.DataFrame(shap_df,columns=ls_classes)
    shap_per_class.insert(0, "Feature", feature_names)
    print(cnn_metrics);print("###Exporting output data: ",dataset_name)

    # ---------------------------
    # 8. Collect SHAP outputs
    # ---------------------------
    CNN_model[dataset_name] = cnn_metrics #collect cnn_metrics
    CNN_model_per_class[dataset_name] = cnn_metrics_per_class #collect cnn_metrics_per_class
    CNN_eval[dataset_name] = epoch_metrics #collect epoch_metrics
    CNN_feature[dataset_name] = shap_scores #collect shap_scores
    CNN_feature_per_class[dataset_name] = shap_per_class #collect shap_per_class
    fig_fname = f"{filenamepath}{dataset_name}.pdf" 

    # -----------------------------
    # 9. Plots
    # -----------------------------
    with PdfPages(fig_fname) as pdf:
        # Overall
        #train-test-validation accuracy
        test_acc = cnn_metrics['Accuracy'].iloc[0]
        train_acc = epoch_metrics['accuracy'].mean()
        val_acc = epoch_metrics['val_accuracy'].mean()
        plt.axhline(y=test_acc, color='#D11E00', linestyle='--', label=f'Test Accuracy (mean): {test_acc:.3f}', zorder=3)
        plt.plot(epoch_metrics['epoch'], epoch_metrics['accuracy'], label=f'Training Accuracy (mean): {train_acc:.3f}', color="#00429D", alpha=0.8)
        plt.plot(epoch_metrics['epoch'], epoch_metrics['val_accuracy'], label=f'Validation Accuracy (mean): {val_acc:.3f}', color="#383838", alpha=0.8)
        plt.title("Accuracy Curves During Model Training")
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        pdf.savefig(dpi=300, bbox_inches='tight')
        plt.close()
        #train-test-validation loss
        plt.plot(epoch_metrics['epoch'], epoch_metrics['loss'], label='Training Loss', color="#00429D", alpha=0.8)
        plt.plot(epoch_metrics['epoch'], epoch_metrics['val_loss'], label='Validation Loss', color="#383838", alpha=0.8)
        plt.title("Loss Curves During Model Training")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        pdf.savefig(dpi=300, bbox_inches='tight')
        plt.close()
        #importance scores
        MMFramework.plot_importance_barplot(shap_scores, figsize=(7,hmsize*1.5))
        pdf.savefig(dpi=300, bbox_inches='tight')
        plt.close()
        
        # Per class
        #model performance
        cnn_metrics_melt = cnn_metrics_per_class.melt(id_vars="Metric", var_name="Class", value_name="Value")
        ax = sns.barplot(data=cnn_metrics_melt, x="Metric", y="Value", hue="Class", palette="Dark2", alpha=0.8)
        # Add values to bars
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f",label_type="edge",fontsize=9)        
        plt.title("Class-wise Performance Metrics")
        plt.ylabel("Performance")
        plt.xlabel("")
        plt.legend(title=None, loc='center left', bbox_to_anchor=(1.02, 0.5))
        plt.xticks(rotation=45)
        pdf.savefig(dpi=300, bbox_inches='tight')
        plt.close()    
        #feature importance
        df_imp = shap_per_class.set_index("Feature")[ls_classes]
        df_imp = df_imp.div(df_imp.sum(axis=1), axis=0) * 100  #scale to 100%
        MMFramework.stack_plot_importance_scores(df_imp, figsize=(7,hmsize*1.5))
        pdf.savefig(dpi=300, bbox_inches='tight')
        plt.close()
        X_flat = X_train_cnn.reshape(X_train_cnn.shape[0], X_train_cnn.shape[1])
        X_flat_t = X_test_cnn.reshape(X_test_cnn.shape[0], X_test_cnn.shape[1])
        # Create figure
        for i, cls in enumerate(ls_classes):
            df_tmp = shap_per_class[["Feature", cls]]
            df_tmp = df_tmp.rename(columns={cls: "Feature_Importance"})
            MMFramework.plot_importance_barplot(df_tmp.sort_values("Feature_Importance", ascending=False), title=f"Average Feature Importance - {cls}", figsize=(7,hmsize*1.5))
            pdf.savefig(dpi=300, bbox_inches='tight')
            plt.close() 
            plt.figure(figsize=(8,hmsize*1.5))
            shap.summary_plot(shap_flat[:, :, i], X_flat, feature_names=feature_names, show=False)
            plt.xlabel("SHAP value")
            plt.title(f"SHAP Summary Plot - {cls} (training set)", fontsize=14)
            pdf.savefig(dpi=300, bbox_inches='tight')
            plt.close()
            plt.figure(figsize=(8,hmsize*1.5))
            shap.summary_plot(shap_flat_t[:, :, i], X_flat_t, feature_names=feature_names, show=False)
            plt.xlabel("SHAP value")
            plt.title(f"SHAP Summary Plot - {cls} (test set)", fontsize=14)
            pdf.savefig(dpi=300, bbox_inches='tight')
            plt.close()
       
# ---------------------------
# 10. Export outputs
# ---------------------------
# Save to CSV
# Overall
combined = pd.concat(CNN_model, names=["FoldRep"]) #model performance
combined = combined.reset_index(level=0).rename(columns={"FoldRep": "Fold.Rep"})
combined.to_csv(f"{filenamepath}model.csv", index=False)
combined = pd.concat(CNN_eval, names=["FoldRep"]) #train-test-validation
combined = combined.reset_index(level=0).rename(columns={"FoldRep": "Fold.Rep"})
combined.to_csv(f"{filenamepath}eval.csv", index=False)
combined = pd.concat(CNN_feature, names=["FoldRep"]) #feature importance
combined = combined.reset_index(level=0).rename(columns={"FoldRep": "Fold.Rep"})
combined.to_csv(f"{filenamepath}feature.csv", index=False)
combined = MMFramework.compute_rank_stability(CNN_feature) #ranked feature
combined.to_csv(f"{filenamepath}feature_rank.csv", index=False)
# Per class
combined = pd.concat(CNN_model_per_class, names=["FoldRep"]) #model performance
combined = combined.reset_index(level=0).rename(columns={"FoldRep": "Fold.Rep"})
combined.to_csv(f"{filenamepath}model_per_class.csv", index=False)
combined = pd.concat(CNN_feature_per_class, names=["FoldRep"]) #feature importance
combined = combined.reset_index(level=0).rename(columns={"FoldRep": "Fold.Rep"})
combined.to_csv(f"{filenamepath}feature_per_class.csv", index=False)

# ---------------------------
# 11. Plot model outputs
# ---------------------------
# Overall
CNN_summary = pd.DataFrame({"Mean": pd.concat(CNN_model, names=["FoldRep"]).mean(), "SD": pd.concat(CNN_model, names=["FoldRep"]).std()})
EVAL_summary = pd.DataFrame({"Mean": pd.concat(CNN_eval, names=["FoldRep"]).mean(), "SD": pd.concat(CNN_eval, names=["FoldRep"]).std()})
EVAL_stat = pd.DataFrame([{"Resample": key, "train_accuracy": df["accuracy"].mean()}
                             for key, df in CNN_eval.items()])
eval_all = pd.concat(CNN_eval.values(), ignore_index=True)
EPOCH_stat = eval_all.groupby("epoch")[["accuracy", "val_accuracy"]].mean()
feature_df = pd.concat(CNN_feature, names=["FoldRep"]).reset_index()
FEATURE_summary = feature_df.groupby("Feature")["Feature_Importance"].agg(["mean", "std"]).sort_values("mean", ascending=True)
# Combine all folds per class
#model
model_all_folds = []
for fold_name, df in CNN_model_per_class.items():
    df = df.melt(id_vars="Metric", var_name="Class", value_name="Value")
    df["Fold"] = fold_name
    model_all_folds.append(df)
df_model_all_folds = pd.concat(model_all_folds, ignore_index=True)
#feature
feature_all_folds = []
for fold_name, df in CNN_feature_per_class.items():
    df = df.melt(id_vars="Feature", var_name="Class", value_name="Value")
    df["Fold"] = fold_name
    feature_all_folds.append(df)
df_feature_all_folds = pd.concat(feature_all_folds, ignore_index=True)
#average across folds
df_avg = df_feature_all_folds.groupby(['Feature', 'Class'], sort=False)['Value'].mean().unstack(fill_value=0)
df_percent = df_avg.div(df_avg.sum(axis=1), axis=0) * 100 #scale to 100%

# Convert dict of dicts, one per class
class_importance_dicts = {}
for cls in ls_classes:
    per_class_dict = {}
    for fold, df in CNN_feature_per_class.items():
        tmp = df[["Feature", cls]].copy()
        tmp = tmp.rename(columns={cls: "Feature_Importance"})
        tmp = tmp.sort_values("Feature_Importance", ascending=False).reset_index(drop=True)
        per_class_dict[fold] = tmp
    class_importance_dicts[cls] = per_class_dict
    
with PdfPages(f"{filenamepath}summary.pdf") as pdf:
    # Overall
    #resample
    mean_test = round(CNN_summary.loc["Accuracy","Mean"],3)
    mean_train = round(EVAL_summary.loc["accuracy", "Mean"], 3)
    mean_val = round(EVAL_summary.loc["val_accuracy", "Mean"], 3)
    plt.plot(range(1, (EVAL_stat.shape[0]+1)), EVAL_stat["train_accuracy"], linestyle="-", color="#00429D", alpha=0.6)
    plt.axhline(mean_train, linestyle="--", color="#00429D", alpha=0.8, label=f'Training Accuracy (mean): {mean_train:.3f}')
    plt.axhline(mean_val, linestyle="--", color="#383838", alpha=0.8, label=f'Validation Accuracy (mean): {mean_val:.3f}')
    plt.axhline(mean_test, linestyle="--", color="#D11E00", alpha=0.8, label=f'Test Accuracy (mean): {mean_test:.3f}')
    plt.title("Accuracy Curves Across Training Iterations")
    plt.xlabel("Round");plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    pdf.savefig(dpi=300, bbox_inches='tight')
    plt.close()
    #bar
    bars = plt.bar(CNN_summary.index, CNN_summary["Mean"], yerr=CNN_summary["SD"], capsize=5, color="#2B47AD")
    plt.ylabel("")
    plt.title("Overall Model Performance (Mean ± SD)")
    plt.xticks(rotation=45)
    for bar, mean, sd in zip(bars, CNN_summary["Mean"], CNN_summary["SD"]):
        plt.text(bar.get_x() + bar.get_width()/2, mean + sd + 0.002, f'{mean:.3f} ± {sd:.2f}', 
                 ha='center', va='bottom')
    pdf.savefig(dpi=300, bbox_inches='tight')
    plt.close()

##    #epoch
##    plt.plot(range(1, (EPOCH_stat.shape[0]+1)), EPOCH_stat["accuracy"], linestyle="--", color="#00429D", alpha=0.6)
##    plt.axhline(mean_train, linestyle="-", color="#00429D", alpha=0.8, label=f'Train Accuracy = {mean_train:.3f}')
##    plt.axhline(mean_val, linestyle="-", color="#383838", alpha=0.8, label=f'Validation Accuracy = {mean_val:.3f}')
##    plt.axhline(mean_test, linestyle="-", color="#D11E00", alpha=0.8, label=f'Test Accuracy = {mean_test:.3f}')
##    plt.title("Overall Model Performance")
##    plt.xlabel("Round");plt.ylabel("Accuracy")
##    plt.legend()
##    plt.grid(True)
##    pdf.savefig(dpi=300, bbox_inches='tight')
##    plt.close()
    
    # Overall features
    plt.figure(figsize=(7,hmsize*1.5))
    bars = plt.barh(FEATURE_summary.index, FEATURE_summary["mean"], xerr=FEATURE_summary["std"], capsize=5, color="#7F90E3")
    plt.xlabel("Feature Importance")
    plt.title("Average Feature Importance")
    for bar, mean, sd in zip(bars, FEATURE_summary["mean"], FEATURE_summary["std"]):
        plt.text(mean + sd + 0.002, bar.get_y() + bar.get_height()/2, f'{mean:.3f} ± {sd:.2f}', 
                 va='center', ha='left')
    pdf.savefig(dpi=300, bbox_inches='tight')
    plt.close()
    MMFramework.plot_importance_heatmap(CNN_feature, title="Overall Feature Importance Map", figsize=(7,hmsize*1.5))
    pdf.savefig(dpi=300, bbox_inches='tight')
    plt.close()   
    MMFramework.plot_rank_stability(CNN_feature, figsize=(8,hmsize*1.5))
    pdf.savefig(dpi=300, bbox_inches='tight')
    plt.close()
    # Overall per class
    ax = sns.barplot(data=df_model_all_folds, x="Metric", y="Value", hue="Class", palette="Dark2",errorbar=None,alpha=0.8)
    # Add values to bars
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f",label_type="edge",fontsize=9)
    plt.title("Overall Class-wise Performance Metrics")
    plt.ylabel("Performance")
    plt.xlabel("")
    plt.xticks(rotation=45)
    plt.legend(title=None, loc='center left', bbox_to_anchor=(1.02, 0.5))
    pdf.savefig(dpi=300, bbox_inches='tight')
    plt.close()
    MMFramework.stack_plot_importance_scores(df_percent, figsize=(7,hmsize*1.5))
    pdf.savefig(dpi=300, bbox_inches='tight')
    plt.close()
    rank_tables = []
    for cls, results_dict in class_importance_dicts.items():
        MMFramework.plot_importance_heatmap_barplot(results_dict, title=f"Feature Importance Summary - {cls}", figsize=(18,hmsize*1.5))
        pdf.savefig(dpi=300, bbox_inches='tight')
        plt.close()
        MMFramework.plot_rank_stability(results_dict, title=f"Rank Stability Plot - {cls}", figsize=(8,hmsize*1.5))
        pdf.savefig(dpi=300, bbox_inches='tight')
        plt.close()
        rank_summary = MMFramework.compute_rank_stability(results_dict) #ranked feature
        # Keep only median rank and rename column to class name
        tmp = rank_summary[["Feature", "Rank_median"]].rename(
            columns={"Rank_median": cls}
        )
        rank_tables.append(tmp)
    rank_feature_summary_perclass = rank_tables[0]
    for df in rank_tables[1:]:
        rank_feature_summary_perclass = rank_feature_summary_perclass.merge(df, on="Feature", how="left")
        rank_feature_summary_perclass.to_csv(f"{filenamepath}feature_rank_per_class.csv", index=False) #ranked feature
print("#### DONE!! ####")
