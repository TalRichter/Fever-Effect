# -----------------------------------------------------------------------------------
# Section 1: Setup and Data Preparation
# -----------------------------------------------------------------------------------

# # Ensure necessary packages are installed and loaded
# if (!requireNamespace("BiocManager", quietly = TRUE))
#   install.packages("BiocManager")
# BiocManager::install(c("DESeq2", "apeglm", "BiocParallel", "rtracklayer"))

library(DESeq2)
library(BiocParallel)
library(rtracklayer)

# Register parallel workers for improved performance
register(SnowParam(4))

# Load preprocessed data
load("RNASeq_PCA_OutlierAnalysis_Workspace.RData")

# Define paths for saving outputs
results_path <- "G://My Drive//XGBoost_Fever_effect//deseq2//DE_results2//"
plots_path <- "G://My Drive//XGBoost_Fever_effect//deseq2//plots2//"
# Ensure output directories exist
dir.create(results_path, recursive = TRUE, showWarnings = FALSE)
dir.create(plots_path, recursive = TRUE, showWarnings = FALSE)

# Import and process GTF for gene symbol conversion
gtf_path <-  "G://.shortcut-targets-by-id//10NLX5IbYP3TzUvd0Zqw06Zn4dZ2tlQRK//SSC_RNAseq//RNASeq_Large//resourceFiles//refAnnot//gencode.v34.chr_patch_hapl_scaff.annotation.gtf.gz"
gtf_data <- rtracklayer::import(gtf_path)
gene_info <- subset(gtf_data, type == "gene")
ensembl_ids <- sapply(gene_info$gene_id, function(x) strsplit(x, ";")[[1]][1])
gene_symbols <- sapply(gene_info$gene_name, function(x) strsplit(x, ";")[[1]][1])
id_to_symbol_mapping <- setNames(gene_symbols, ensembl_ids)

# Function to filter txi data for a given set of sample IDs
filterTxiForSamples <- function(txi, sampleTable_filtered) {
  filtered_txi <- txi
  filtered_txi$counts <- filtered_txi$counts[, rownames(sampleTable_filtered)]
  filtered_txi$abundance <- filtered_txi$abundance[, rownames(sampleTable_filtered)]
  filtered_txi$length <- filtered_txi$length[, rownames(sampleTable_filtered)]
  return(filtered_txi)
}

# Replace missing verbal_iq and nonverbal_iq values before any preprocessing
mean_verbal_iq <- mean(sampleTable_no_outliers$verbal_iq, na.rm = TRUE)
sampleTable_no_outliers$verbal_iq[is.na(sampleTable_no_outliers$verbal_iq)] <- mean_verbal_iq

mean_nonverbal_iq <- mean(sampleTable_no_outliers$nonverbal_iq, na.rm = TRUE)
sampleTable_no_outliers$nonverbal_iq[is.na(sampleTable_no_outliers$nonverbal_iq)] <- mean_nonverbal_iq

# -----------------------------------------------------------------------------------
# Section 2: Analysis for All Samples
# -----------------------------------------------------------------------------------

# Prepare DESeqDataSet
dds_all <- DESeqDataSetFromTximport(txi_no_outliers,
                                    colData = sampleTable_no_outliers,
                                    design = ~ sex + race + verbal_iq + Target)

# Run DESeq analysis
dds_all <- DESeq(dds_all, parallel = FALSE)

# Obtain DE results
results_de_all <- results(dds_all, name="Target_yes_vs_no")
rownames(results_de_all) <- id_to_symbol_mapping[rownames(results_de_all)]
write.csv(as.data.frame(results_de_all), file = paste0(results_path, "DE_results_all_samples.csv"))

# MA Plot before LFC shrinkage
png(file = paste0(plots_path, "all_samples_MA_plot_before_LFC.png"))
plotMA(results_de_all, main = "All Samples Before LFC")
dev.off()

# LFC Shrinkage
results_lfc_all <- lfcShrink(dds_all, coef="Target_yes_vs_no", type="apeglm", parallel = TRUE)
rownames(results_lfc_all) <- id_to_symbol_mapping[rownames(results_lfc_all)]
write.csv(as.data.frame(results_lfc_all), file = paste0(results_path, "LFC_results_all_samples.csv"))

# MA Plot after LFC shrinkage
png(file = paste0(plots_path, "all_samples_MA_plot_after_LFC.png"))
plotMA(results_lfc_all, main = "All Samples After LFC")
dev.off()

# -----------------------------------------------------------------------------------
# Section 3: Analysis for GI Samples
# -----------------------------------------------------------------------------------
sampleTable_GI <- sampleTable_no_outliers[ sampleTable_no_outliers$any_gastro_disorders_proband == "yes", ]

# Filter data
txi_gi <- filterTxiForSamples(txi_no_outliers, sampleTable_GI)


# Prepare DESeqDataSet for GI samples
dds_GI <- DESeqDataSetFromTximport(txi_gi,
                                   colData = sampleTable_GI,
                                   design = ~ sex + race + verbal_iq + Target)

# Run DESeq analysis for GI samples
dds_GI <- DESeq(dds_GI, parallel = FALSE)

# Obtain DE results for GI samples
results_de_GI <- results(dds_GI, name="Target_yes_vs_no")
rownames(results_de_GI) <- id_to_symbol_mapping[rownames(results_de_GI)]
write.csv(as.data.frame(results_de_GI), file = paste0(results_path, "DE_results_GI_samples.csv"))

# MA Plot before LFC shrinkage for GI samples
png(file = paste0(plots_path, "GI_samples_MA_plot_before_LFC.png"))
plotMA(results_de_GI, main = "GI Samples Before LFC")
dev.off()

# LFC Shrinkage for GI samples
results_lfc_GI <- lfcShrink(dds_GI, coef="Target_yes_vs_no", type="apeglm", parallel = FALSE)
rownames(results_lfc_GI) <- id_to_symbol_mapping[rownames(results_lfc_GI)]
write.csv(as.data.frame(results_lfc_GI), file = paste0(results_path, "LFC_results_GI_samples.csv"))

# MA Plot after LFC shrinkage for GI samples
png(file = paste0(plots_path, "GI_samples_MA_plot_after_LFC.png"))
plotMA(results_lfc_GI, main = "GI Samples After LFC")
dev.off()
