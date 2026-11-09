# R/dimension.R
# Collinearity diagnostics, Mahalanobis Q-Q geometry, robust PCA projection,
# and the bootstrap Ladle estimator (Luo & Li, 2016, Biometrika 103:875-887).

collinearity = function(X, label = "Cohort") {
  R = cor(X)
  upperVals = abs(R[upper.tri(R)])
  eigVals   = eigen(R, symmetric = TRUE, only.values = TRUE)$values

  totalPairs = ncol(X) * (ncol(X) - 1) / 2
  kappaVal   = max(eigVals) / max(min(eigVals), .Machine$double.eps)
  dEff       = sum(eigVals > 1)

  summaryDf = data.frame(
    Cohort         = label,
    n              = nrow(X),
    p              = ncol(X),
    TotalPairs     = totalPairs,
    PairsGt05      = sum(upperVals > 0.5),
    PctGt05        = 100 * mean(upperVals > 0.5),
    PairsGt08      = sum(upperVals > 0.8),
    PctGt08        = 100 * mean(upperVals > 0.8),
    LambdaMax      = max(eigVals),
    EffectiveDim   = dEff,
    ConditionKappa = kappaVal,
    stringsAsFactors = FALSE
  )

  list(
    summary     = summaryDf,
    corMatrix   = R,
    eigenvalues = eigVals
  )
}

# Ordered squared Mahalanobis distances against theoretical chi-square quantiles
mahalanobisQq = function(Z, seed = 734985) {
  d2       = mahalanobis(Z, colMeans(Z), cov(Z))
  chisqRef = qchisq(ppoints(nrow(Z)), df = ncol(Z))
  data.frame(
    chisqQuantile = chisqRef,
    mahalanobisD2 = sort(d2)
  )
}

fitPca = function(zDiscovery, zReference = NULL, rankK = 7) {
  pcaFit  = prcomp(zDiscovery, center = FALSE, scale. = FALSE)
  eigVals = pcaFit$sdev^2
  propVar = eigVals / sum(eigVals)
  cumVar  = cumsum(propVar)

  scoresDisc = pcaFit$x[, seq_len(rankK), drop = FALSE]
  scoresRef  = if (!is.null(zReference)) {
    zReference %*% pcaFit$rotation[, seq_len(rankK), drop = FALSE]
  } else {
    NULL
  }

  varTable = data.frame(
    Component         = paste0("PC", seq_along(eigVals)),
    Eigenvalue        = eigVals,
    ProportionPercent = 100 * propVar,
    CumulativePercent = 100 * cumVar,
    stringsAsFactors  = FALSE
  )

  list(
    pcaModel        = pcaFit,
    scoresDiscovery = scoresDisc,
    scoresReference = scoresRef,
    scores          = scoresDisc,
    refScores       = scoresRef,
    varianceTable   = varTable,
    varTable        = varTable
  )
}

ladle = function(Z, nBoot = 400, maxDim = 15,
                 plotMaxK = 10, seed = 734985) {
  set.seed(seed)
  nRow   = nrow(Z)
  nCol   = ncol(Z)
  maxDim = min(maxDim, nCol - 1)

  covMat = cov(Z)
  eig    = eigen(covMat, symmetric = TRUE)
  ord    = order(eig$values, decreasing = TRUE)
  eigValues  = eig$values[ord]
  eigVectors = eig$vectors[, ord, drop = FALSE]

  signalSubspace = function(V, k) {
    if (k <= 0) matrix(0, nrow(V), 0)
    else V[, seq_len(k), drop = FALSE]
  }

  refSubspaces = lapply(seq_len(maxDim), function(k) signalSubspace(eigVectors, k))
  subspaceVar  = numeric(maxDim)

  for (kLoop in seq_len(maxDim)) {
    discrepancy = numeric(nBoot)
    refBasis    = refSubspaces[[kLoop]]

    for (b in seq_len(nBoot)) {
      bootIdx   = sample.int(nRow, nRow, replace = TRUE)
      covBoot   = cov(Z[bootIdx, , drop = FALSE])
      eigBoot   = eigen(covBoot, symmetric = TRUE)
      bootOrd   = order(eigBoot$values, decreasing = TRUE)
      bootVecs  = eigBoot$vectors[, bootOrd, drop = FALSE]
      bootBasis = signalSubspace(bootVecs, kLoop)

      discrepancy[b] = 1 - abs(det(t(refBasis) %*% bootBasis))
    }

    subspaceVar[kLoop] = mean(discrepancy)
  }

  insK  = subspaceVar / (1 + sum(subspaceVar))
  eigK  = eigValues[2:(maxDim + 1)] / (1 + sum(eigValues[1:(maxDim + 1)]))
  ladle = c(
    eigValues[1] / (1 + sum(eigValues[1:(maxDim + 1)])),
    insK + eigK
  )

  kGrid   = 0:plotMaxK
  idxGrid = seq_len(plotMaxK + 1)
  estDim  = kGrid[which.min(ladle[idxGrid])]

  ladleDf = data.frame(
    k              = kGrid,
    Eigenvalue     = eigValues[idxGrid],
    BootstrapVarFn = c(0, insK[seq_len(plotMaxK)]),
    NormalizedPhi  = eigValues[idxGrid] / (1 + sum(eigValues[1:(maxDim + 1)])),
    LadleGn        = ladle[idxGrid]
  )

  list(
    estimatedD = estDim,
    table      = ladleDf
  )
}

# Pairwise Euclidean distance distribution and Mahalanobis Q-Q plot
plotDistances = function(pwDist, qqDf) {
  meanVal = round(mean(pwDist), 2)
  medVal  = round(median(pwDist), 2)

  pDist = ggplot(data.frame(d = pwDist), aes(x = d)) +
    geom_histogram(bins = 30, fill = "#a6cee3", colour = "#1f78b4", linewidth = 0.35) +
    geom_vline(xintercept = meanVal, colour = "#e31a1c", linetype = "dashed", linewidth = 0.9) +
    geom_vline(xintercept = medVal,  colour = "#33a02c", linetype = "dashed", linewidth = 0.9) +
    annotate("text", x = max(pwDist) * 0.72, y = Inf, vjust = 2.0, hjust = 0,
             label = paste0("Mean = ", meanVal, " (red)\nMedian = ", medVal, " (green)"),
             size = 4.0, fontface = "bold", colour = "grey20") +
    labs(
      title    = "Pairwise Euclidean Distances (n = 140)",
      subtitle = "Robustly standardised 61-variable space",
      x        = "Pairwise Euclidean Distance",
      y        = "Frequency"
    ) +
    themeBiostat()

  pQq = ggplot(qqDf, aes(x = chisqQuantile, y = mahalanobisD2)) +
    geom_abline(intercept = 0, slope = 1, colour = "#e31a1c", linetype = "dashed", linewidth = 0.9) +
    geom_point(colour = "#1f78b4", size = 2.6, alpha = 0.85) +
    labs(
      title    = "Mahalanobis Chi-Square Q-Q Plot (df = 61)",
      subtitle = "Dashed line: y = x. Points above the line in the upper tail flag extreme observations.",
      x        = expression(chi[61]^2 ~ "Theoretical Quantiles"),
      y        = "Ordered Squared Mahalanobis Distance"
    ) +
    themeBiostat()

  list(distanceHist = pDist, mahalanobisQq = pQq)
}

# Clustered correlation matrix heatmap
plotHeatmap = function(corMatrix) {
  hcOrd = hclust(as.dist(1 - corMatrix), method = "complete")$order
  Rord  = corMatrix[hcOrd, hcOrd]
  p     = ncol(Rord)
  gridDf = expand.grid(Row = seq_len(p), Col = seq_len(p))
  gridDf$r = as.vector(Rord)

  tickIdx    = seq(1, p, by = 5)
  tickLabels = colnames(Rord)[tickIdx]

  ggplot(gridDf, aes(x = Col, y = p - Row + 1, fill = r)) +
    geom_tile() +
    scale_fill_gradient2(
      low = "#b2182b", mid = "white", high = "#2166ac",
      midpoint = 0, limits = c(-1, 1), name = "Correlation (r)"
    ) +
    scale_x_continuous(breaks = tickIdx, labels = tickLabels, expand = c(0, 0)) +
    scale_y_continuous(breaks = p - tickIdx + 1, labels = tickLabels, expand = c(0, 0)) +
    coord_fixed() +
    labs(
      title    = "Clustered Correlation Matrix of 61 Variables (n = 140)",
      subtitle = "Variables reordered by complete-linkage hierarchical clustering (41.0% of pairs have |r| > 0.5)",
      x        = "Variable (Clustered Order)",
      y        = "Variable (Clustered Order)"
    ) +
    themeBiostat() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 9),
          axis.text.y = element_text(size = 9))
}

# Bootstrap Ladle objective and eigenvalue scree plot
plotLadle = function(ladleRes) {
  lt   = ladleRes$table
  dHat = ladleRes$estimatedD

  dfLong = rbind(
    data.frame(
      k     = lt$k,
      value = lt$Eigenvalue,
      panel = "(a) Sample Eigenvalue Scree"
    ),
    data.frame(
      k     = lt$k[-1],
      value = lt$BootstrapVarFn[-1],
      panel = "(b) Eigenvector Instability fn(k)"
    ),
    data.frame(
      k     = lt$k,
      value = lt$LadleGn,
      panel = paste0("(c) Ladle Objective gn(k) [Min d = ", dHat, "]")
    )
  )
  dfLong$panel = factor(dfLong$panel, levels = unique(dfLong$panel))

  minPt = data.frame(
    k     = dHat,
    value = lt$LadleGn[lt$k == dHat],
    panel = factor(paste0("(c) Ladle Objective gn(k) [Min d = ", dHat, "]"),
                   levels = levels(dfLong$panel))
  )

  ggplot(dfLong, aes(x = k, y = value, colour = panel)) +
    geom_line(linewidth = 1.0) +
    geom_point(size = 2.8) +
    geom_point(data = minPt, aes(x = k, y = value),
               colour = "#e31a1c", size = 4.5, inherit.aes = FALSE) +
    facet_wrap(~ panel, scales = "free_y", ncol = 3) +
    scale_x_continuous(breaks = seq(0, max(lt$k), by = 2)) +
    scale_colour_manual(values = c("#1f78b4", "#ff7f00", "#33a02c"), guide = "none") +
    labs(
      title    = paste0("Bootstrap Ladle Subspace Order Determination (Luo & Li, 2016; B = 400, d_hat = ", dHat, ")"),
      subtitle = "Red marker highlights the global minimum of gn(k) = fn(k) + phi_n(k) at d = 6",
      x        = "Candidate Subspace Dimension (k)",
      y        = "Criterion Value"
    ) +
    themeBiostat()
}
