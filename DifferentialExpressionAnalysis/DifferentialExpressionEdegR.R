################################################################################
# Comprehensive Analysis and Visualization for GI Sample Differential Expression
################################################################################

# This script encompasses the workflow for processing RNA-Seq data, performing differential expression analysis,
# and visualizing results with a focus on GI samples and the gene "IL2RA".

# -----------------------------------------------------------------------------------
# Section 1: Setup and Data Preparation
# -----------------------------------------------------------------------------------
# Load necessary libraries for analysis and visualization
library(edgeR)
library(limma)
library(tximport)
library(ggplot2)
library(readr)
library(rtracklayer)
library(gridExtra)

# Load the previously saved workspace with essential variables
load("RNASeq_PCA_OutlierAnalysis_Workspace.RData")

# Define the GTF path and results directory
gtf_path <- "G://.shortcut-targets-by-id//10NLX5IbYP3TzUvd0Zqw06Zn4dZ2tlQRK//SSC_RNAseq//RNASeq_Large//resourceFiles//refAnnot//gencode.v34.chr_patch_hapl_scaff.annotation.gtf.gz"
results_directory <- "G://My Drive//XGBoost_Fever_effect//limma_edgeR//results_and_files"

# Create the directory if it doesn't exist
if (!dir.exists(results_directory)) {
  dir.create(results_directory, recursive = TRUE)
}
# Function to filter txi data for a given set of sample IDs
filterTxiForSamples <- function(txi, sampleTable_filtered) {
  filtered_txi <- txi
  filtered_txi$counts <- filtered_txi$counts[, rownames(sampleTable_filtered)]
  filtered_txi$abundance <- filtered_txi$abundance[, rownames(sampleTable_filtered)]
  filtered_txi$length <- filtered_txi$length[, rownames(sampleTable_filtered)]
  return(filtered_txi)
}

# -----------------------------------------------------------------------------------
# Section 2: Preprocessing and Normalization
# -----------------------------------------------------------------------------------
# Handle missing IQ values by imputing with the mean

mean_nonverbal_iq <- mean(sampleTable_no_outliers$nonverbal_iq, na.rm = TRUE)
sampleTable_no_outliers$nonverbal_iq[is.na(sampleTable_no_outliers$nonverbal_iq)] <- mean_nonverbal_iq

mean_verbal_iq <- mean(sampleTable_no_outliers$verbal_iq, na.rm = TRUE)
sampleTable_no_outliers$verbal_iq[is.na(sampleTable_no_outliers$verbal_iq)] <- mean_verbal_iq

# Import and process GTF for gene symbol conversion
gtf_data <- rtracklayer::import(gtf_path)
gene_info <- subset(gtf_data, type == "gene")
ensembl_ids <- sapply(gene_info$gene_id, function(x) strsplit(x, ";")[[1]][1])
gene_symbols <- sapply(gene_info$gene_name, function(x) strsplit(x, ";")[[1]][1])
id_to_symbol_mapping <- setNames(gene_symbols, ensembl_ids)

# Normalize data and prepare for differential expression analysis
y <- DGEList(counts = txi_no_outliers$counts)
keep <- rowSums(cpm(y) >= 3) >= 306
y <- y[keep,]
y <- calcNormFactors(y)
logCpmData <- cpm(y, log = TRUE, prior.count = 2)
rownames(logCpmData) <- id_to_symbol_mapping[rownames(logCpmData)]

# Save log-transformed CPM data
write.table(t(logCpmData), file.path(results_directory, "counts_LogCPM_Allsamples.csv"), sep = "\t")

# -----------------------------------------------------------------------------------
# Section 3: Differential Expression Analysis
# -----------------------------------------------------------------------------------
# Prepare the design matrix and perform differential expression analysis
designMatrix <- model.matrix(~0 + Target + race + sex + verbal_iq, data = sampleTable_no_outliers)
v <- voom(y, designMatrix, plot = FALSE)
fit <- lmFit(v, designMatrix)
contrastMatrix <- makeContrasts(yes_vs_no = Targetyes - Targetno, levels = designMatrix)
fit <- contrasts.fit(fit, contrastMatrix)
fit <- eBayes(fit)
deResultsAll <- topTable(fit, sort.by = "P", n = Inf)
deResultsAll$GeneSymbol <- id_to_symbol_mapping[rownames(deResultsAll)]

# Save differential expression results for all samples
write.csv(deResultsAll, file.path(results_directory, "Differential_Expression_Results_AllSamples.csv"))

# -----------------------------------------------------------------------------------
# Section 4: Focused Analysis on GI Samples
# -----------------------------------------------------------------------------------
# Filter data for GI samples and repeat differential expression analysis
# Filter for GI samples
sampleTable_GI <- sampleTable_no_outliers[ sampleTable_no_outliers$any_gastro_disorders_proband == "yes", ]
# Filter data
txi_gi <- filterTxiForSamples(txi_no_outliers, sampleTable_GI)
y_gi <- DGEList(counts = txi_gi$counts)
keep <- rowSums(cpm(y_gi) >= 1) >= 167
y_gi <- y_gi[keep,]
y_gi <- calcNormFactors(y_gi)
logCpmDataGI <- cpm(y_gi, log = TRUE, prior.count = 2)
rownames(logCpmDataGI) <- id_to_symbol_mapping[rownames(logCpmDataGI)]

# Save log-transformed CPM data for GI samples
write.csv(t(logCpmDataGI), file.path(results_directory, "counts_LogCPM_GIsamples.csv"))

designMatrixGI <- model.matrix(~0 + Target + race + sex + verbal_iq , data = sampleTable_GI)
vGI <- voom(y_gi, designMatrixGI, plot = FALSE)
fitGI <- lmFit(vGI, designMatrixGI)
contrastMatrixGI <- makeContrasts(yes_vs_no = Targetyes - Targetno, levels = designMatrixGI)
fitGI <- contrasts.fit(fitGI, contrastMatrixGI)
fitGI <- eBayes(fitGI)
deResultsGI <- topTable(fitGI, sort.by = "P", n = Inf)
deResultsGI$GeneSymbol <- id_to_symbol_mapping[rownames(deResultsGI)]

# Save differential expression results for GI samples
write.csv(deResultsGI, file.path(results_directory, "Differential_Expression_Results_GISamples.csv"))

# -----------------------------------------------------------------------------------
# Section 5: Visualization 
# -----------------------------------------------------------------------------------
# Volcano plot visualization
volcanoData <- with(deResultsGI, data.frame(logFC = logFC, PValue = P.Value, Gene = rownames(deResultsGI)))
volcanoData$logP = -log10(volcanoData$PValue)
volcanoPlot <- ggplot(volcanoData, aes(x = logFC, y = logP)) +
  geom_point(aes(color = PValue < 0.05), alpha = 0.5) +
  geom_text(data = subset(volcanoData, Gene == geneSymbol), aes(label = Gene), vjust = 2, color = "red") +
  theme_minimal() +
  labs(title = "Volcano Plot", x = "Log Fold Change", y = "-Log10 P-value") +
  scale_color_manual(values = c("TRUE" = "red", "FALSE" = "grey")) +
  guides(color = FALSE)

volcanoData$GeneSymbol <- deResultsGI$GeneSymbol
volcanoData$adjPVal <- deResultsGI$adj.P.Val
volcanoData$logAdjP <- -log10(volcanoData$adjPVal)
significantGenes <- subset(volcanoData, adjPVal < 0.05)

volcanoPlot <- ggplot(volcanoData, aes(x = logFC, y = logP)) +
  geom_point(aes(color = adjPVal < 0.05), alpha = 0.5) +
  geom_text(data = significantGenes, aes(label = paste(GeneSymbol, formatC(adjPVal, format = 'e', digits = 2), sep = "\n")), vjust = 1.5, color = "red", size = 5) +
  theme_minimal() +
  labs(title = "Volcano Plot", x = "Log Fold Change", y = "-Log10 P-value") +
  scale_color_manual(values = c("TRUE" = "red", "FALSE" = "grey")) +
  guides(color = FALSE)+
  theme(
    axis.title.x = element_text(size = 16, face = "bold"), # X axis title
    axis.title.y = element_text(size = 16, face = "bold"), # Y axis title
    axis.text.x = element_text(size = 14), # X axis text (numbers)
    axis.text.y = element_text(size = 14), # Y axis text (numbers)
  )




ggsave("volcano_plot_with_gene_symbols_and_adjP.pdf", plot = volcanoPlot, width = 10, height = 8, dpi = 300)
