import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
import shap
import os
import random
import math
import dill
import sys
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, label_binarize
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, ReLU
from tensorflow.keras.utils import to_categorical

# -----------------------------
# Utility functions
# -----------------------------
# Set seed
def set_all_seeds(seed_value):
    # Set the Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    # Set the random seed for Python's built-in random module
    random.seed(seed_value)

    # Set the random seed for NumPy
    np.random.seed(seed_value)

    # Set the global random seed for TensorFlow
    tf.random.set_seed(seed_value)

# Reads a text file containing lists
def read_list_file(file_path):
    lists_dict = {}
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if ':' in line:
                name, values_str = line.split(':', 1)
                name = name.strip()
                values = [int(x.strip()) for x in values_str.split(',')]
                lists_dict[name] = values
                
    return lists_dict

# Get active features by threshold
def get_active_features(feature_importance, feature_names, threshold=0.5):
    active_idx = np.where(feature_importance >= threshold)[0]
    active_features = [(feature_names[i], feature_importance[i]) for i in active_idx]
    return active_features

# Plot rank stability with median±IQR error bars
def compute_rank_stability(results_dict):
    # Convert dictionary into long DataFrame
    df_list = []
    for fold, data in results_dict.items():
        tmp = data.copy()
        tmp["Fold"] = fold
        df_list.append(tmp)
    df = pd.concat(df_list, ignore_index=True)

    # Rank features within each fold from 1
    df["Rank"] = df.groupby("Fold")["Feature_Importance"].rank(
        ascending=False, method="average"
    )

    # Compute mean ± std rank per feature
##    rank_summary = (
##        df.groupby("Feature")["Rank"]
##        .agg(["mean", "std"])
##        .reset_index()
##        .sort_values("mean", ascending=True)
##    )

    rank_summary = (
        df.groupby("Feature")["Rank"]
        .agg(
            Rank_median="median",
            Rank_IQR=lambda x: x.quantile(0.75) - x.quantile(0.25)
            )
        .reset_index()
        .sort_values("Rank_median", ascending=True)
    )
    return rank_summary

# Plot stacked horizontal bars of importance scores per class
def stack_plot_importance_scores(input_importance, figsize=(8,9), title=None):
    # Plot stacked horizontal bars
    df_long = input_importance
    colors = cm.get_cmap('Dark2').colors
    ax = df_long.plot(kind='barh', stacked=True, figsize=figsize, color=colors, alpha=0.8, width=0.9)

    # Add text labels
    for i, row in enumerate(df_long.values):
        start = 0
        for value, cls in zip(row, df_long.columns):
            if value > 0:
                ax.text(start + value/2, i, f"{value:.1f}", ha='center', va='center', color='black', fontsize=10)
            start += value

    # Customize labels and title
    ax.set_xlabel("Percentage of Importance (%)", fontsize=14)
    ax.set_ylabel("", fontsize=9)
    ax.set_title("Class-wise Feature Importance Distribution", fontsize=16)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(title=None, fontsize=12, loc='center left', bbox_to_anchor=(1.02, 0.5))
    #plt.tight_layout()
    #plt.show()

# Plot a barplot of importance scores
def plot_importance_barplot(score_df, feature_col="Feature", score_col="Feature_Importance", title="Average Feature Importance", figsize=(8,5)):
    plt.figure(figsize=figsize)
    ax = sns.barplot(data=score_df,x=score_col,y=feature_col,color="#7F90E3")

    # Add values inside bars
    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%.3f",
            label_type="edge",
            padding=3,color="black",
            fontsize=9
        )
        
    plt.title(title)
    plt.ylabel("")
    plt.xlabel("Feature Importance")
    #plt.tight_layout()
    #plt.show()

# Plot importance summary
def plot_importance_heatmap(results_dict, title="Feature Importance Summary", figsize=(8,5)):
    # Convert dictionary to long DataFrame
    df_list = []
    for fold, data in results_dict.items():
        tmp = data.copy()
        tmp["Fold"] = fold
        df_list.append(tmp)
    df = pd.concat(df_list, ignore_index=True)

    # Compute mean and std per feature
    summary = (
        df.groupby("Feature")["Feature_Importance"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )

    # Order features for heatmap and barplot
    feature_order = summary["Feature"].tolist()
    df["Feature"] = pd.Categorical(df["Feature"], categories=feature_order, ordered=True)

    # Heatmap
    heatmap_data = df.pivot(index="Feature", columns="Fold", values="Feature_Importance")

    # Create figure
    plt.figure(figsize=figsize)

    # Heatmap
    ax = sns.heatmap(
        heatmap_data,cmap="PuRd",cbar_kws={"label": "Importance"}
    )
    # Change font sizes
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=14)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    # Colorbar formatting
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("Feature Importance", fontsize=14)   
    #plt.tight_layout()
    #plt.show()

# Plot importance summary
def plot_importance_heatmap_barplot(results_dict, title="Feature Importance Summary", figsize=(18,8)):
    # Convert dictionary to long DataFrame
    df_list = []
    for fold, data in results_dict.items():
        tmp = data.copy()
        tmp["Fold"] = fold
        df_list.append(tmp)
    df = pd.concat(df_list, ignore_index=True)

    # Compute mean and std per feature
    summary = (
        df.groupby("Feature")["Feature_Importance"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )

    # Order features for heatmap and barplot
    feature_order = summary["Feature"].tolist()
    df["Feature"] = pd.Categorical(df["Feature"], categories=feature_order, ordered=True)

    # Heatmap
    heatmap_data = df.pivot(index="Feature", columns="Fold", values="Feature_Importance")

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [2, 1]})

    # Heatmap
    sns.heatmap(
        heatmap_data,cmap="PuRd",cbar_kws={"label": "Importance"},
        ax=axes[0]
    )
    # Change font sizes
    axes[0].set_title("Feature Importance Map", fontsize=14)
    axes[0].set_xlabel(None)
    axes[0].set_ylabel(None)
    axes[0].tick_params(axis="x", labelsize=12)
    axes[0].tick_params(axis="y", labelsize=14)
    plt.setp(axes[0].get_xticklabels(), rotation=35, ha="right")
    # Colorbar formatting
    cbar = axes[0].collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("Feature Importance", fontsize=14)


    # Barplot with error bars
    bars = axes[1].barh(
        summary["Feature"],
        summary["mean"],
        xerr=summary["std"],
        color="#7F90E3"
    )
    # Change font sizes
    axes[1].tick_params(axis="x", labelsize=14)
    axes[1].tick_params(axis="y", labelsize=14)
    axes[1].set_title("Average Feature Importance", fontsize=14)
    axes[1].set_xlabel("Feature Importance", fontsize=14)
    axes[1].set_ylabel(None)
    axes[1].invert_yaxis()  #Top feature appears first

    # Add labels (mean ± std)
    offset = max(summary["mean"]) * 0.05  #5% as spacing
    for bar, mean_val, std_val in zip(bars, summary["mean"], summary["std"]):
        axes[1].text(
            bar.get_width() + std_val + offset,   #Shift beyond error bar
            bar.get_y() + bar.get_height() / 2,
            f"{mean_val:.3f} ± {std_val:.2f}",    
            va="center",ha="left",fontsize=12,color="black"
        )
    fig.suptitle(title, fontsize=16, fontweight="bold")
    #plt.tight_layout()
    #plt.show()

# Plot rank stability with median±IQR error bars
def plot_rank_stability(results_dict, title="Rank Stability Plot", cutoff=None, figsize=(10,8)):
    # Get rank
    rank_summary = compute_rank_stability(results_dict)
    # Use median rank as a cutoff
    if cutoff is None:
        cutoff = rank_summary["Rank_median"].median()
        cutoff_label = "Median Rank"
    else:
        cutoff_label = f"Rank ≤ {cutoff}"

    # Which features are above the cutoff
    rank_summary["highlight"] = rank_summary["Rank_median"] <= cutoff

    # Define marker size range
    ms_min, ms_max = 3, 18
    # Normalize Rank_median
    ranks = rank_summary["Rank_median"]
    sizes = ms_max - (ranks - ranks.min()) / (ranks.max() - ranks.min()) * (ms_max - ms_min)
    rank_summary["markersize"] = 10 #sizes
    
    # Plot
    plt.figure(figsize=figsize)

    # Plot median ± IQR with different colors for highlighted features
    for idx, row in rank_summary.iterrows():
        color = "#D65FA1" if row["highlight"] else "#007BFF"
        plt.errorbar(
            row["Rank_median"],row["Feature"],xerr=row["Rank_IQR"] / 2,
            fmt="o",color=color,ecolor="black",elinewidth=1,capsize=4,markersize=row["markersize"]
        )

    # Cutoff line
    plt.axvline(cutoff, color="#D35400", linestyle="--", label=cutoff_label, alpha=0.8)

    plt.xlabel("Rank", fontsize=14)
    plt.ylabel(None)
    plt.title("Point: Median Rank; Bar: IQR", fontsize=10)
    plt.suptitle(title, fontsize=14, y=0.98)
    plt.gca().invert_yaxis()  # Best ranks appear at top
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tick_params(axis="both", labelsize=12)

    # Create legend    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Top Features', 
               markerfacecolor='#D65FA1', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Other Features', 
               markerfacecolor='#007BFF', markersize=10),
        Line2D([0], [0], color='red', lw=2, linestyle='--', label=cutoff_label)
    ]
    # Legend position
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14)
    #plt.tight_layout(rect=[0, 0, 0.85, 1])  
    #plt.show()

# -----------------------------
# Convert features to heatmaps
# -----------------------------
# Convert features into a heatmap of size AxA (15**0.5)
def features_to_heatmap(features, size=(4,4)):
    arr = np.zeros(size[0]*size[1])
    arr[:len(features)] = features
    arr = arr.reshape(size)
    return arr

# -----------------------------
# Plot heatmap functions
# -----------------------------
# Set features into a heatmap of size AxA (15**0.5)
def plot_heatmap_with_labels(ax, heatmapimg, label, feature_names, size):
    hmap = plt.cm.summer_r
    ax.imshow(heatmapimg, cmap=hmap, aspect="auto")
    ax.set_title(f"Class {label}")
    ax.axis("off")
    
    idx = 0 #add feature labels
    for i in range(size[0]):
        for j in range(size[1]):
            if idx < len(feature_names):
                ax.text(j, i, feature_names[idx],
                         ha="center", va="center",
                         color="black", fontsize=10)
                idx += 1

# Plot multiple heatmaps in an nrow × ncol
def plot_multiple_heatmaps(heatmaps, labels, feature_names, size=(4,4), ncol=3, figsize=(14, 8), title="Example feature heatmaps"):
    n = len(heatmaps)
    nrow = math.ceil(n / ncol)  #auto rows

    fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
    axes = np.array(axes).reshape(-1)

    for i, (heatmap, label) in enumerate(zip(heatmaps, labels)):
        plot_heatmap_with_labels(axes[i], heatmap, label, feature_names, size=size)

    # Hide unused subplots if not a perfect fit
    for j in range(len(heatmaps), nrow*ncol):
        axes[j].axis("off")
    
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.95)
    #plt.tight_layout()
    #plt.show()

# ---------------------------
# Grad-CAM functions
# ---------------------------
# Compute grad-cam
def grad_cam(model, image, class_index=None, layer_name=None):
    input_tensor = tf.convert_to_tensor(np.expand_dims(image, axis=0), dtype=tf.float32)
    if layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, Conv2D):
                layer_name = layer.name
                break
    conv_layer = model.get_layer(layer_name)

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_tensor, training=False)
        if class_index is None:
            class_index = tf.argmax(predictions[0])
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    weights = tf.reduce_mean(grads, axis=(0,1,2))
    cam = tf.reduce_sum(tf.multiply(weights, conv_outputs[0]), axis=-1)
    cam = tf.nn.relu(cam)
    cam = (cam - tf.reduce_min(cam)) / (tf.reduce_max(cam) - tf.reduce_min(cam) + 1e-8)
    cam = tf.image.resize(cam[..., tf.newaxis], (image.shape[0], image.shape[1]))
    return cam.numpy().squeeze()

# ---------------------------
# Grad-CAM outputs
# ---------------------------

# Plot a heatmap of Grad-CAM importance with feature names and values highlight by threshold
def plot_avg_gradcam_heatmap(avg_importance, feature_names, threshold=0.5, size=(4,4), figsize=(8,8), title=""):
    # Reshape importance to size heatmap
    heatmap = np.zeros(size)
    idx = 0
    for r in range(size[0]):
        for c in range(size[1]):
            if idx < len(avg_importance):
                heatmap[r, c] = avg_importance[idx]
                idx += 1

    plt.figure(figsize=figsize)
    im = plt.imshow(heatmap, cmap="PuRd", zorder=1)
    plt.axis("off")
    ax = plt.gca()

    # Add colorbar
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label("Grad-CAM Importance", rotation=270, labelpad=15)
    
    idx = 0
    for r in range(size[0]):
        for c in range(size[1]):
            if idx < len(feature_names):
                val = heatmap[r, c]
                if val >= threshold:
                    # Highlight active cells
                    rect = plt.Rectangle(
                        (c-0.5, r-0.5), 1, 1,
                        linewidth=4, edgecolor="#000000", facecolor="none", zorder=2
                    )
                    ax.add_patch(rect)
                    text_color = "white"
                else:
                    text_color = "white"
                
                # Add feature name and value
                plt.text(c, r-0.15, feature_names[idx], ha="center", va="center", color=text_color, fontsize=11, fontweight="bold", zorder=3)
                plt.text(c, r+0.15, f"{val:.3f}", ha="center", va="center", color=text_color, fontsize=10, zorder=3)
                idx += 1

    plt.title(f"Grad-CAM Importance Map (threshold={threshold}) - {title}")
    #plt.show()

# Plot a heatmap of Grad-CAM importance with feature names and values highlight by top n features
def plot_avg_gradcam_heatmap_topn(avg_importance, feature_names, top_n=5, size=(4,4), figsize=(8,8), title=""):    
    # Reshape importance to size heatmap
    heatmap = np.zeros(size)
    idx = 0
    for r in range(size[0]):
        for c in range(size[1]):
            if idx < len(avg_importance):
                heatmap[r, c] = avg_importance[idx]
                idx += 1

    # Determine indices of top-n features
    top_idx = np.argsort(avg_importance)[-top_n:] #indices of top n features

    plt.figure(figsize=figsize)
    im = plt.imshow(heatmap, cmap="PuRd", zorder=1)
    plt.axis("off")
    ax = plt.gca()

    # Add colorbar
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label("Grad-CAM Importance", rotation=270, labelpad=15)
    
    # Annotate cells
    idx = 0
    for r in range(size[0]):
        for c in range(size[1]):
            if idx < len(feature_names):
                val = heatmap[r, c]
                if idx in top_idx:
                    # Highlight top-n cells
                    rect = plt.Rectangle(
                        (c-0.5, r-0.5), 1, 1,
                        linewidth=4, edgecolor="#000000", facecolor="none", zorder=2
                    )
                    ax.add_patch(rect)
                    text_color = "white"
                else:
                    text_color = "white"
                
                # Add feature name and value
                plt.text(c, r-0.15, feature_names[idx], ha="center", va="center", color=text_color, fontsize=11, fontweight="bold", zorder=3)
                plt.text(c, r+0.15, f"{val:.3f}", ha="center", va="center", color=text_color, fontsize=10, zorder=3)
                idx += 1

    plt.title(f"Grad-CAM Importance Map (Top {top_n}) - {title}")
    #plt.show()

# Plot median heatmaps per class with highlighted active cells based on average Grad-CAM importance threshold
def plot_avg_heatmap_active_cells(X_data, y_data, avg_importance, feature_names, threshold=0.5, size=(4,4), ncol=2, figsize=(10,8), title=""):
    classes = np.unique(y_data)

    # Reshape avg_importance to heatmap size
    importance_map = np.zeros(size)
    idx = 0
    for r in range(size[0]):
        for c in range(size[1]):
            if idx < len(avg_importance):
                importance_map[r, c] = avg_importance[idx]
                idx += 1

    # Create subplot layout
    nrow = math.ceil(len(classes) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol*(size[0]+1),nrow*(size[1]+1)))
    fig.set_layout_engine(None)
    fig.subplots_adjust(wspace=0.01, hspace=0.01)
    axes = np.array(axes).reshape(-1)
    ims = [] #keep references for colorbar

    for ax, cls in zip(axes, classes):
        # Compute average feature values for this class
        #avg_features = np.mean(X_data[y_data == cls], axis=0)
        # Compute median feature values for this class
        avg_features = np.median(X_data[y_data == cls], axis=0)

        # Reshape into heatmap grid
        heatmap = np.zeros(size)
        idx = 0
        for r in range(size[0]):
            for c in range(size[1]):
                if idx < len(avg_features):
                    heatmap[r, c] = avg_features[idx]
                    idx += 1

        # Plot heatmap
        im = ax.imshow(heatmap, cmap="summer_r", zorder=1)
        ims.append(im)
        ax.set_title(f"{cls}")
        ax.axis("off")
        
        # Overlay labels and importance
        idx = 0
        for r in range(size[0]):
            for c in range(size[1]):
                if idx < len(feature_names):
                    imp_val = importance_map[r, c]

                    # highlight active cells
                    if imp_val >= threshold:
                        rect = plt.Rectangle((c-0.5, r-0.5), 1, 1,
                                             linewidth=4, edgecolor="#000000",facecolor="none", zorder=2)
                        ax.add_patch(rect)

                    ax.text(c, r-0.15, feature_names[idx],
                            ha="center", va="center", color="black",fontsize=10, fontweight="bold", zorder=3)
                    ax.text(c, r+0.15, f"{imp_val:.3f}",
                            ha="center", va="center", color="black",fontsize=10, zorder=3)
                    idx += 1
    
    # Hide unused axes (if odd number of classes)
    for j in range(len(classes), len(axes)):
        fig.delaxes(axes[j])

    # Add colorbar at right
    fig.subplots_adjust(bottom=0.15, hspace=0.4)
    # Match colorbar height with the first subplot
    cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.03])   # [left, bottom, width, height]
    cbar = fig.colorbar(ims[0], cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Median Feature Value", rotation=0, labelpad=5)
    fig.suptitle(f"Feature Heatmap and Actived Regions (threshold={threshold}) {title}")
    #plt.show()

# Plot median heatmaps per class with highlighted active cells based on average Grad-CAM importance top n features
def plot_avg_heatmap_active_cells_topn(X_data, y_data, avg_importance, feature_names, top_n=5, size=(4,4), ncol=2, figsize=(10,8), title=""):
    classes = np.unique(y_data)

    # Reshape avg_importance to heatmap size
    importance_map = np.zeros(size)
    idx = 0
    for r in range(size[0]):
        for c in range(size[1]):
            if idx < len(avg_importance):
                importance_map[r, c] = avg_importance[idx]
                idx += 1

    # Determine indices of top-n features
    top_idx = np.argsort(avg_importance)[-top_n:] #indices of top n features
    top_mask = np.zeros_like(avg_importance, dtype=bool)
    top_mask[top_idx] = True

    # Create subplot layout
    nrow = math.ceil(len(classes) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol*(size[0]+1),nrow*(size[1]+1)))
    fig.set_layout_engine(None)
    fig.subplots_adjust(wspace=0.01, hspace=0.01)
    axes = np.array(axes).reshape(-1)
    ims = [] #keep references for colorbar   

    for ax, cls in zip(axes, classes):
        # Compute average feature values for this class
        #avg_features = np.mean(X_data[y_data == cls], axis=0)
        # Compute median feature values for this class
        avg_features = np.median(X_data[y_data == cls], axis=0)

        # Reshape into heatmap grid
        heatmap = np.zeros(size)
        idx = 0
        for r in range(size[0]):
            for c in range(size[1]):
                if idx < len(avg_features):
                    heatmap[r, c] = avg_features[idx]
                    idx += 1

        # Plot heatmap
        im = ax.imshow(heatmap, cmap="summer_r", zorder=1)
        ims.append(im)
        ax.set_title(f"{cls}")
        ax.axis("off")

        # Overlay labels and highlight top-n features
        idx = 0
        for r in range(size[0]):
            for c in range(size[1]):
                if idx < len(feature_names):
                    imp_val = importance_map[r, c]

                    if top_mask[idx]:
                        # highlight top-n cells
                        rect = plt.Rectangle((c-0.5, r-0.5), 1, 1,
                                             linewidth=4, edgecolor="#000000", facecolor="none", zorder=2)
                        ax.add_patch(rect)

                    ax.text(c, r-0.15, feature_names[idx],
                            ha="center", va="center", color="black", fontsize=10, fontweight="bold", zorder=3)
                    ax.text(c, r+0.15, f"{imp_val:.3f}",
                            ha="center", va="center", color="black", fontsize=10, zorder=3)
                    idx += 1
    
    for j in range(len(classes), len(axes)):
        fig.delaxes(axes[j])

    # Add colorbar at right
    fig.subplots_adjust(bottom=0.15, hspace=0.4)
    # Match colorbar height with the first subplot
    cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.03])   # [left, bottom, width, height]
    cbar = fig.colorbar(ims[0], cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Median Feature Value", rotation=0, labelpad=5)
    fig.suptitle(f"Feature Heatmap and Actived Regions (Top {top_n}) {title}")
    #plt.show()

# Plot median heatmaps per class with highlighted active cells based on Grad-CAM importance threshold per class
def plot_avg_heatmap_active_cells_per_class(X_data, y_data, gradcam_per_class, feature_names, threshold=0.5, size=(4,4), ncol=2, figsize=(10,8), title=""):
    classes = np.unique(y_data)

    # Create subplot layout
    nrow = math.ceil(len(classes) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol*(size[0]+1),nrow*(size[1]+1)))
    fig.set_layout_engine(None)
    fig.subplots_adjust(wspace=0.01, hspace=0.01)
    axes = np.array(axes).reshape(-1)
    ims = [] #keep references for colorbar    

    # --- Loop over classes ---
    for i, cls in enumerate(classes):
        ax = axes[i]

        # Get GradCAM importances for this class
        importance = gradcam_per_class[cls].values

        # Compute average feature values for this class
        #avg_features = np.mean(X_data[y_data == cls], axis=0)
        # Compute median feature values for this class
        avg_features = np.median(X_data[y_data == cls], axis=0)
        
        # Create heatmap grid
        heatmap = np.zeros(size)
        idx = 0
        for r in range(size[0]):
            for c in range(size[1]):
                if idx < len(avg_features):
                    heatmap[r, c] = avg_features[idx]
                    idx += 1

        # Plot heatmap
        im = ax.imshow(heatmap, cmap="summer_r", zorder=1)
        ims.append(im)
        ax.set_title(f"{cls}", fontsize=13, fontweight="bold")
        ax.axis("off")

        # Overlay labels and importance
        idx = 0
        for r in range(size[0]):
            for c in range(size[1]):
                if idx < len(feature_names):
                    imp_val = importance[idx]
                    fname = feature_names[idx]

                    # highlight active cells
                    if imp_val >= threshold:
                        rect = plt.Rectangle(
                            (c - 0.5, r - 0.5), 1, 1,
                            linewidth=3, edgecolor="black", facecolor="none", zorder=2
                        )
                        ax.add_patch(rect)

                    # Label feature name and GradCAM value
                    ax.text(c, r - 0.15, fname, ha="center", va="center",
                            color="black", fontsize=9, fontweight="bold", zorder=3)
                    ax.text(c, r + 0.15, f"{imp_val:.3f}", ha="center", va="center",
                            color="black", fontsize=9, zorder=3)
                    idx += 1

    # Hide unused axes (if odd number of classes)
    for j in range(len(classes), len(axes)):
        fig.delaxes(axes[j])

    # Add shared colorbar
    fig.subplots_adjust(bottom=0.15, hspace=0.4)
    # Match colorbar height with the first subplot
    cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.03])   # [left, bottom, width, height]
    cbar = fig.colorbar(ims[0], cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Median Feature Value", rotation=0, labelpad=5)
    fig.suptitle(f"Feature Heatmap and Actived Regions per Class (threshold={threshold}) {title}")
    #plt.show()

# Plot median heatmaps per class with highlighted active cells based on Grad-CAM importance top n features per class
def plot_avg_heatmap_active_cells_topn_per_class(X_data, y_data, gradcam_per_class, feature_names, top_n=5, size=(4,4), ncol=2, figsize=(10,8), title=""):  
    classes = np.unique(y_data)

    # Create subplot layout
    nrow = math.ceil(len(classes) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol*(size[0]+1),nrow*(size[1]+1)))
    fig.set_layout_engine(None)
    fig.subplots_adjust(wspace=0.01, hspace=0.01)
    axes = np.array(axes).reshape(-1)
    ims = [] #keep references for colorbar 

    # --- Loop over classes ---
    for i, cls in enumerate(classes):
        ax = axes[i]

        # Get GradCAM importances for this class
        importance = gradcam_per_class[cls].values

        # Identify top-N features for this class
        top_idx = np.argsort(importance)[-top_n:]
        top_mask = np.zeros_like(importance, dtype=bool)
        top_mask[top_idx] = True

        # Compute average feature values for this class
        #avg_features = np.mean(X_data[y_data == cls], axis=0)
        # Compute median feature values for this class
        avg_features = np.median(X_data[y_data == cls], axis=0)
        
        # Create heatmap grid
        heatmap = np.zeros(size)
        idx = 0
        for r in range(size[0]):
            for c in range(size[1]):
                if idx < len(avg_features):
                    heatmap[r, c] = avg_features[idx]
                    idx += 1

        # Plot heatmap
        im = ax.imshow(heatmap, cmap="summer_r", zorder=1)
        ims.append(im)
        ax.set_title(f"{cls}", fontsize=13, fontweight="bold")
        ax.axis("off")

        # Overlay text labels and highlight top-N features
        idx = 0
        for r in range(size[0]):
            for c in range(size[1]):
                if idx < len(feature_names):
                    imp_val = importance[idx]
                    fname = feature_names[idx]

                    # Highlight top-N
                    if top_mask[idx]:
                        rect = plt.Rectangle(
                            (c - 0.5, r - 0.5), 1, 1,
                            linewidth=3, edgecolor="black", facecolor="none", zorder=2
                        )
                        ax.add_patch(rect)

                    # Label feature name and GradCAM value
                    ax.text(c, r - 0.15, fname, ha="center", va="center",
                            color="black", fontsize=9, fontweight="bold", zorder=3)
                    ax.text(c, r + 0.15, f"{imp_val:.3f}", ha="center", va="center",
                            color="black", fontsize=9, zorder=3)
                    idx += 1

    # Hide unused axes (if odd number of classes)
    for j in range(len(classes), len(axes)):
        fig.delaxes(axes[j])

    # Add shared colorbar
    fig.subplots_adjust(bottom=0.15, hspace=0.4)
    cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.03])   # [left, bottom, width, height]
    cbar = fig.colorbar(ims[0], cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Median Feature Value", rotation=0, labelpad=5)
    fig.suptitle(f"Feature Heatmap and Actived Regions per Class (Top {top_n}) {title}")
    #plt.show()

# ---------------------------
# SHAP outputs
# ---------------------------
