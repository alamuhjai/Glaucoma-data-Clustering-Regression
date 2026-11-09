# glaucomaHrtAnalysis.R
# Runs the whole analysis from the repository root and writes every figure used
# in README.md to figures/. Usage: Rscript glaucomaHrtAnalysis.R

suppressPackageStartupMessages({
  library(mclust)
  library(ggplot2)
  library(RColorBrewer)
  library(gridExtra)
})

for (f in c("preprocess", "dimension", "clustering", "classmap")) {
  source(file.path("R", paste0(f, ".R")))
}

# Plot theme used by the plotting functions in R/ (defined here if missing)
if (!exists("themeBiostat")) {
  themeBiostat = function(baseSize = 12) {
    theme_bw(base_size = baseSize) +
      theme(
        plot.title       = element_text(face = "bold", hjust = 0.5, size = baseSize + 1),
        plot.subtitle    = element_text(hjust = 0.5, colour = "grey25", size = baseSize - 1),
        axis.title       = element_text(face = "bold", size = baseSize),
        axis.text        = element_text(size = baseSize - 1, colour = "grey15"),
        panel.grid.minor = element_blank(),
        legend.position  = "bottom",
        legend.box       = "vertical",
        legend.title     = element_text(face = "bold", size = baseSize - 1),
        legend.text      = element_text(size = baseSize - 1),
        strip.background = element_rect(fill = "#f0f4f8", colour = "grey75"),
        strip.text       = element_text(face = "bold", size = baseSize - 1),
        plot.margin      = margin(10, 14, 10, 14)
      )
  }
}

seed   = 734985
figDir = "figures"
dir.create(figDir, showWarnings = FALSE)

savePlot = function(p, fileName, width = 8.2, height = 5.4) {
  ggsave(file.path(figDir, fileName), plot = p, width = width, height = height,
         dpi = 200, bg = "white")
}

# 1. Partition and robust scaling ---------------------------------------------
if (!file.exists("data/GlauData.txt"))
  stop("Dataset not found: place your copy at data/GlauData.txt (see README).")
df     = read.table("data/GlauData.txt", header = TRUE)
cohort = splitCohort(df, nDiscovery = 140, seed = seed)
std    = standardize(cohort$xDiscovery, cohort$xReference)

diagDisc = scaleDiag(cohort$xDiscovery)
diagFull = scaleDiag(cohort$xFull)

scaleSummary = data.frame(
  Cohort   = c("Discovery (n = 140)", "Complete (n = 167)"),
  minSD    = c(min(diagDisc$sd), min(diagFull$sd)),
  maxSD    = c(max(diagDisc$sd), max(diagFull$sd)),
  ratioSD  = c(max(diagDisc$sd) / min(diagDisc$sd), max(diagFull$sd) / min(diagFull$sd)),
  stable   = c(sum(diagDisc$ratio <= 1), sum(diagFull$ratio <= 1)),
  inflated = c(sum(diagDisc$ratio > 1), sum(diagFull$ratio > 1)),
  below05  = c(sum(diagDisc$ratio < 0.5), sum(diagFull$ratio < 0.5))
)
cat("\n== Section 1: scale diagnostics ==\n"); print(scaleSummary, digits = 6)

scalePlots = plotScales(diagDisc, "Discovery Sample (n = 140)")
savePlot(scalePlots$sdVsMad,    "sdVsMad.png",    width = 8.0, height = 5.2)
savePlot(scalePlots$scaleRatio, "scaleRatio.png", width = 8.2, height = 5.2)

# 2. Collinearity and distance geometry ---------------------------------------
mcFull = collinearity(cohort$xFull,      "Complete cohort")
mcDisc = collinearity(cohort$xDiscovery, "Discovery sample")
cat("\n== Section 2: collinearity ==\n"); print(rbind(mcFull$summary, mcDisc$summary), digits = 6)

pwDist    = as.numeric(dist(std$zDiscovery))
qqDf      = mahalanobisQq(std$zDiscovery)
geomPlots = plotDistances(pwDist, qqDf)
ggsave(file.path(figDir, "distQq.png"),
       arrangeGrob(geomPlots$distanceHist, geomPlots$mahalanobisQq, ncol = 2),
       width = 14, height = 5.4, dpi = 200, bg = "white")
savePlot(plotHeatmap(mcDisc$corMatrix), "corHeatmap.png", width = 8.0, height = 7.2)

# 3. PCA and Ladle --------------------------------------------------------------
pca    = fitPca(std$zDiscovery, std$zNew, rankK = 7)
varTab = pca$varianceTable
cat("\n== Section 3.1: PCA (center = FALSE) ==\n"); print(varTab[1:10, ], digits = 6)
cat("Total variance:", round(sum(varTab$Eigenvalue), 2), "\n")

ladle = ladle(std$zDiscovery, nBoot = 400, maxDim = 15,
              plotMaxK = 10, seed = seed)
cat("\n== Section 3.2: Ladle (maxDim = 15) d_hat =", ladle$estimatedD, "==\n")
print(ladle$table, digits = 4)
savePlot(plotLadle(ladle), "ladlePlot.png", width = 10.5, height = 4.4)

# 4. Clustering and reference-observation mapping -----------------------------
bench = clusterBench(pca$scoresDiscovery, seed = seed)
km2   = bench$kmeansModels[["2"]]
cat("\n== Section 4.1: clustering comparison ==\n"); print(bench$comparisonTable, digits = 7)
cat("Sizes K = 2: K-means", table(km2$cluster),
    "| hierarchical", table(bench$hcClusters[["2"]]),
    "| EM", table(bench$gmmModels[["2"]]$classification), "\n")

varPct = varTab$ProportionPercent[1:2]
chTab  = bench$comparisonTable
chOf   = function(k, m) round(chTab$CH[chTab$k == k & chTab$Method == m], 2)

savePlot(plotClusters(pca$scoresDiscovery, km2$cluster, pca$scoresReference,
                      cohort$yReference, varPct,
                      paste0("K-Means (K = 2, CH = ", chOf(2, "K-means"), ")")),
         "kmeansK2.png", width = 8.0, height = 5.4)
savePlot(plotClusters(pca$scoresDiscovery, bench$hcClusters[["2"]], pca$scoresReference,
                      cohort$yReference, varPct,
                      paste0("Complete-Linkage Hierarchical (K = 2, CH = ",
                             chOf(2, "Hierarchical (Complete)"), ")")),
         "hcK2.png", width = 8.0, height = 5.4)
savePlot(plotClusters(pca$scoresDiscovery, bench$gmmModels[["2"]]$classification,
                      pca$scoresReference, cohort$yReference, varPct,
                      paste0("EM Gaussian Mixture (G = 2, CH = ",
                             chOf(2, "Gaussian Mixture (EM)"), ")")),
         "gmmK2.png", width = 8.0, height = 5.4)

map = mapReference(pca$scoresDiscovery, km2$cluster, km2$centers,
                   pca$scoresReference, cohort$yReference)
cat("\n== Section 4.2: reference observations by nearest K-means centroid ==\n")
print(map$confusionMatrix)

# Label clusters by size (Cluster 1 = larger, Cluster 2 = smaller)
sizes      = table(km2$cluster)
largerId   = names(sizes)[which.max(sizes)]
sizeLabel  = ifelse(km2$cluster == largerId,
                    paste0("Larger cluster (n = ", max(sizes), ")"),
                    paste0("Smaller cluster (n = ", min(sizes), ")"))
sizeLabel  = factor(sizeLabel, levels = c(paste0("Larger cluster (n = ", max(sizes), ")"),
                                          paste0("Smaller cluster (n = ", min(sizes), ")")))
cat("Larger cluster id:", largerId,
    "| mean PC1 larger:", round(mean(pca$scoresDiscovery[km2$cluster == largerId, 1]), 2),
    "| mean PC1 smaller:", round(mean(pca$scoresDiscovery[km2$cluster != largerId, 1]), 2), "\n")

savePlot(plotClusters(pca$scoresDiscovery, sizeLabel, pca$scoresReference,
                      cohort$yReference, varPct,
                      "K-Means Clusters (K = 2) with the 12 Reference Observations",
                      groupLabel = "K-means cluster"),
         "phenotypes.png", width = 8.2, height = 5.4)

# 5. Class map ---------------------------------------------------------------------
cmHrt = classmap(pca$scoresDiscovery, sizeLabel)
cat("\n== Section 5: class map (K-means clusters) ==\n"); print(cmHrt$summary, digits = 4)
savePlot(plotClassmap(cmHrt, "Class Map of the K-Means Clusters in 7-PC Space (n = 140)"),
         "classmapHrt.png")

cat("\nR", as.character(getRversion()), "| mclust", as.character(packageVersion("mclust")),
    "| ggplot2", as.character(packageVersion("ggplot2")), "\n")
