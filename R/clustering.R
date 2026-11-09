# R/clustering.R
# Subspace clustering (K-means, complete-linkage hierarchical, EM Gaussian mixtures),
# WSS/BSS/Calinski-Harabasz validation, nearest-centroid clinical mapping, and PCA cluster plots.

# Within-cluster SS, between-cluster SS, and Calinski-Harabasz index for a partition.
clusterStats = function(X, labels) {
  n            = nrow(X)
  uniqueLabels = sort(unique(labels))
  k            = length(uniqueLabels)
  overallMean  = colMeans(X)

  centroids = t(sapply(uniqueLabels, function(cl) {
    colMeans(X[labels == cl, , drop = FALSE])
  }))

  wss = sum(sapply(seq_along(uniqueLabels), function(i) {
    cl  = uniqueLabels[i]
    pts = X[labels == cl, , drop = FALSE]
    ctr = matrix(centroids[i, ], nrow = nrow(pts), ncol = ncol(pts), byrow = TRUE)
    sum(rowSums((pts - ctr)^2))
  }))

  bss = sum(sapply(seq_along(uniqueLabels), function(i) {
    cl = uniqueLabels[i]
    nL = sum(labels == cl)
    sum((centroids[i, ] - overallMean)^2) * nL
  }))

  ch = (bss / (k - 1)) / (wss / (n - k))
  list(WSS = wss, BSS = bss, CH = ch)
}

# Compare hierarchical clustering (complete linkage), K-means, and Mclust GMMs
# for K = 2 and K = 3 in the reduced PCA score space.
clusterBench = function(scoresDiscovery, seed = 734985) {
  distMat = dist(scoresDiscovery)
  hcModel = hclust(distMat, method = "complete")
  hcK2    = cutree(hcModel, k = 2)
  hcK3    = cutree(hcModel, k = 3)

  set.seed(seed)
  kmK2 = kmeans(scoresDiscovery, centers = 2, nstart = 50)
  set.seed(seed)
  kmK3 = kmeans(scoresDiscovery, centers = 3, nstart = 50)
  set.seed(123)
  kmK4 = kmeans(scoresDiscovery, centers = 4, nstart = 25)

  set.seed(seed)
  gmmK2 = Mclust(scoresDiscovery, G = 2, verbose = FALSE)
  set.seed(seed)
  gmmK3 = Mclust(scoresDiscovery, G = 3, verbose = FALSE)

  sHc2  = clusterStats(scoresDiscovery, hcK2)
  sKm2  = clusterStats(scoresDiscovery, kmK2$cluster)
  sGmm2 = clusterStats(scoresDiscovery, gmmK2$classification)

  sHc3  = clusterStats(scoresDiscovery, hcK3)
  sKm3  = clusterStats(scoresDiscovery, kmK3$cluster)
  sGmm3 = clusterStats(scoresDiscovery, gmmK3$classification)

  compTable = data.frame(
    k      = c(2, 2, 2, 3, 3, 3),
    Method = c("Hierarchical (Complete)", "K-means", "Gaussian Mixture (EM)",
               "Hierarchical (Complete)", "K-means", "Gaussian Mixture (EM)"),
    WSS    = c(sHc2$WSS, sKm2$WSS, sGmm2$WSS,
               sHc3$WSS, sKm3$WSS, sGmm3$WSS),
    BSS    = c(sHc2$BSS, sKm2$BSS, sGmm2$BSS,
               sHc3$BSS, sKm3$BSS, sGmm3$BSS),
    CH     = c(sHc2$CH,  sKm2$CH,  sGmm2$CH,
               sHc3$CH,  sKm3$CH,  sGmm3$CH),
    stringsAsFactors = FALSE
  )

  list(
    comparisonTable = compTable,
    kmeansModels    = list("2" = kmK2, "3" = kmK3, "4" = kmK4),
    hcModel         = hcModel,
    hcClusters      = list("2" = hcK2, "3" = hcK3),
    gmmModels       = list("2" = gmmK2, "3" = gmmK3)
  )
}

# Assign each of the 12 projected reference observations to its closest cluster centroid
# in the 7-PC subspace and label each cluster by majority reference diagnosis.
mapReference = function(scoresDiscovery, clusterDiscovery,
                        centroids, scoresReference, yReference) {
  clFactor   = factor(clusterDiscovery)
  nearestIdx = apply(scoresReference, 1, function(x) {
    which.min(colSums((t(centroids) - x)^2))
  })
  refCluster = factor(nearestIdx, levels = levels(clFactor))

  clusterToDiag = sapply(levels(clFactor), function(cl) {
    idx = which(refCluster == cl)
    if (length(idx) == 0) return("unassigned")
    names(sort(table(yReference[idx]), decreasing = TRUE))[1]
  })

  confMat = table(Cluster = refCluster, Diagnosis = yReference)
  confDf  = as.data.frame.matrix(confMat)
  alignmentTable = data.frame(
    Cluster         = paste0("Cluster ", rownames(confDf)),
    MappedDiagnosis = as.character(clusterToDiag),
    DiscoverySize   = as.integer(table(clFactor)),
    confDf,
    row.names   = NULL,
    check.names = FALSE
  )

  mappedDiscovery = factor(
    paste0("Cluster ", clFactor, " (", clusterToDiag[as.character(clFactor)], ")")
  )

  list(
    referenceCluster   = refCluster,
    clusterToDiagnosis = clusterToDiag,
    confusionMatrix    = confMat,
    alignmentTable     = alignmentTable,
    discoveryDiagnosis = mappedDiscovery
  )
}

# PC1-PC2 subspace scatter with projected reference observations
plotClusters = function(scoresDiscovery, groupDiscovery,
                        scoresReference, yReference,
                        varPct, titleText, groupLabel = "Cluster") {
  tab = table(groupDiscovery)
  groupFactor = if (!all(grepl("\\(n =", names(tab)))) {
    factor(groupDiscovery, levels = names(tab),
           labels = paste0("Cluster ", names(tab), " (n = ", tab, ")"))
  } else {
    factor(groupDiscovery)
  }

  dfDisc = data.frame(
    PC1   = scoresDiscovery[, 1],
    PC2   = scoresDiscovery[, 2],
    group = groupFactor
  )
  dfRef = data.frame(
    PC1       = scoresReference[, 1],
    PC2       = scoresReference[, 2],
    diagnosis = factor(yReference)
  )

  nGroups = length(levels(dfDisc$group))
  pal = brewer.pal(min(max(nGroups, 3), 8), "Set1")[seq_len(nGroups)]

  ggplot() +
    geom_point(data = dfDisc, aes(x = PC1, y = PC2, colour = group),
               size = 2.8, alpha = 0.75) +
    scale_colour_manual(values = pal) +
    geom_point(data = dfRef, aes(x = PC1, y = PC2, shape = diagnosis),
               size = 4.2, colour = "black", stroke = 1.1) +
    labs(
      title    = titleText,
      subtitle = "Coloured: n = 140 undiagnosed observations | Black symbols: n = 12 diagnosed reference observations",
      x        = paste0("PC1 (", round(varPct[1], 1), "% variance)"),
      y        = paste0("PC2 (", round(varPct[2], 1), "% variance)"),
      colour   = groupLabel,
      shape    = "Reference Diagnosis"
    ) +
    themeBiostat()
}
