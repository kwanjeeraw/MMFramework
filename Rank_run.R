library(dplyr)
library(tidyr)
library(ggplot2)
library(ggtext)
library(optparse)

# -----------------------------
# 00. Parse cmd arguments
# -----------------------------
option_list <- list(
  make_option(
    "--wkpath", type = "character", help = "Working directory path of core codes e.g. /path/to/code_directory (default: getcwd())"
  ),
  make_option(
    "--outpath", type = "character", help = "Output directory path e.g. OUT"
  ),
  make_option(
    "--nperm", type = "integer", default = 100, help = "Number of permutations [default %default]"
  ),
  make_option(
    "--infiles", type = "character", help = "CSV files with feature importance scores to compute consensus ranks e.g. gradcam_feature.csv pls_feature.csv"
  )
)
opt <- parse_args(
  OptionParser(option_list = option_list),
  positional_arguments = TRUE
)

wkpath  <- opt$options$wkpath
outpath <- opt$options$outpath
n_perm <- opt$options$nperm
inputfiles <- c(opt$options$infiles, opt$args)

# ---- Checks ----
if (is.null(wkpath) || is.null(outpath)) {
  stop("Both --wkpath and --outpath must be provided")
}
if (length(inputfiles) == 0) {
  stop("No input files provided after --infiles")
}

# -----------------------------
# 01. Apply arguments
# -----------------------------
setwd(wkpath)
if (!dir.exists(outpath)) {
  dir.create(outpath, recursive = TRUE)
}

# -----------------------------
# 0. Initialize
# -----------------------------
cat("Checking input files ..\n")

expected_cols <- c("Fold.Rep", "Feature", "Feature_Importance")
df_list <- lapply(inputfiles, function(f) {
  read.csv(f, stringsAsFactors = FALSE) #read input Fold.Rep|Feature|Feature_Importance
})
## check colnames
unmatched <- which(!sapply(df_list, function(df)
  setequal(colnames(df), expected_cols)
))
if (length(unmatched) > 0) {
  message("Column mismatch in files: ", paste(unmatched, collapse = ", "),"\nExpected columns: Fold.Rep Feature Feature_Importance")
  lapply(unmatched, function(i) colnames(df_list[[i]]))
  stop("STOP! due to unmatched column names.")
}
## scale each input to 100
df_list_scaled <- lapply(df_list, function(df) {
  df$Feature_Importance <- ave(df$Feature_Importance,df$Fold.Rep,
    FUN = function(x) x / max(x, na.rm = TRUE) * 100) #scale to 100
  df
})
combined_df = do.call(rbind, df_list_scaled)

## 2. Compute rank
cat("Computing Rank product ..\n")
observed_rp_df <- combined_df %>% group_by(Fold.Rep) %>%
  mutate(r = rank(-Feature_Importance, ties.method = "average")) %>%
  ungroup() %>% group_by(Feature) %>%
  summarise(
    RankProduct  = exp(mean(log(r))), # Rank product
    Rank_median  = median(r), # Median rank
    Rank_IQR     = IQR(r), # IQR
    .groups = "drop"
  )

## 3. Permutation Test
set.seed(14)
null_rps = matrix(NA, nrow = nrow(observed_rp_df), ncol = n_perm)
for (i in 1:n_perm) {
  # Shuffle features within each fold
  perm_data <- combined_df %>% group_by(Fold.Rep) %>% 
    mutate(Feature = sample(Feature)) %>%
    ungroup()
  
  # Compute RP
  null_rps[, i] <- perm_data %>% group_by(Fold.Rep) %>%
    mutate(r = rank(-Feature_Importance, ties.method = "average")) %>%
    arrange(Feature) %>% group_by(Feature) %>%
    summarise(RP = exp(mean(log(r))), .groups = 'drop') %>%
    pull(RP)
}

## 4. Calculate P-values and FDR
## feature-specific distribution
result_rp <- observed_rp_df %>%
  mutate(
    p_value = sapply(1:nrow(observed_rp_df), function(j) {
      sum(null_rps[j, ] <= RankProduct[j]) / n_perm
    }),
    FDR = p.adjust(p_value, method = "fdr")
  ) %>% arrange(RankProduct)
## global/pooled distribution
result_rp <- observed_rp_df %>%
  mutate(
    p_value = sapply(RankProduct, function(x) {mean(null_rps <= x, na.rm = TRUE)}),
    FDR = p.adjust(p_value, method = "fdr") 
  ) %>% arrange(RankProduct)

## 5. Plot and export
cat("Exporting outputs to",paste0(wkpath,"/",outpath),"..\n")
RP_plot = ggplot(result_rp, aes(x = RankProduct, y = reorder(Feature, -RankProduct))) +
  geom_point(aes(fill = FDR < 0.05),size = 5,alpha = 0.9,shape = 21, color = "black", stroke = 0.4) +
  geom_text(aes(label = round(RankProduct, 2)),nudge_y = -0.4,size = 4.5,show.legend = FALSE) +
  scale_size_continuous(name = "Rank Product",trans = "reverse") +
  scale_fill_manual(values = c("#007BFF", "#D65FA1"), labels = c("NS", "FDR < 0.05")) +
  labs(title = "Consensus Ranks with Significance",
    subtitle = paste(n_perm, "Permutations with FDR correction"),
    x = "Rank Product", y = "", color = "Significance"
  ) + theme_bw() +
  theme(plot.title = element_text(size = 18), plot.subtitle = element_text(size = 12), 
        axis.title = element_text(size = 18), axis.text = element_text(size = 16),
        text = element_text(size = 16)
  )

# Plot
cutoff = median(result_rp$Rank_median, na.rm = TRUE)
# Define caption with colors
caption_text = paste0(
  "<span style='color:#C90000;'> Median Rank = ",cutoff,"</span>"
)
MedianRank_Plot = ggplot(result_rp, aes(x = Rank_median, y = reorder(Feature, -Rank_median))) +
  geom_errorbarh(aes(xmin = Rank_median - (Rank_IQR/2), xmax = Rank_median + (Rank_IQR/2)),
                 height = 0.2, color = "black") +
  geom_point(aes(fill = Rank_median > cutoff),size = 5,alpha = 0.9,shape = 21,
             color = "black", stroke = 0.4) +
  geom_text(aes(label = round(Rank_median, 2)),nudge_y = -0.4,size = 4.5,show.legend = FALSE) +
  geom_vline(xintercept = cutoff, linetype = "dashed", color = "#D35400", linewidth = 1, alpha = 0.8) +
  scale_size_continuous(name = "Rank",trans = "reverse") +
  scale_fill_manual(values = c("#D65FA1", "#007BFF"), labels = c(" > Median Rank","< Median Rank")) +
  labs(x = "Rank", y = "", color = "Color",
       title = "Rank Stability Plot", subtitle = "Point: Median Rank; Bar: IQR", 
       caption = caption_text) + theme_bw() +
  theme(plot.title = element_text(size = 18), plot.subtitle = element_text(size = 12), 
        axis.title = element_text(size = 18), axis.text = element_text(size = 16),
        text = element_text(size = 16), 
        plot.caption = element_markdown(size = 14, hjust = 0) #ggtext for colored caption
  )

pdf(file = paste0(outpath,"/rankplots_outputs.pdf"), width=9.5,height=(8+nrow(result_rp)*0.03)) #Overall model performance
# Overall
print(RP_plot)
print(MedianRank_Plot)
dev.off()
write.csv(result_rp,file = paste0(outpath,"/rankvalues_outputs.csv"), row.names = FALSE)
cat("#### DONE!! ####\n")