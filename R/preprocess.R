# R/preprocess.R
# Cohort partitioning, median/MAD standardisation, and scale diagnostic plots.

# Separate the 12 diagnosed reference observations (Class != NA) from the 155 undiagnosed
# observations first so the reference labels never leak into scaling or PCA fitting.
splitCohort = function(data, nDiscovery = 140, seed = 734985) {
  classCol = which(names(data) == "Class")
  hasLabel = !is.na(data$Class)

  xFull       = as.matrix(data[, -classCol, drop = FALSE])
  xLabelled   = as.matrix(data[hasLabel, -classCol, drop = FALSE])
  yLabelled   = as.character(data$Class[hasLabel])
  xUnlabelled = as.matrix(data[!hasLabel, -classCol, drop = FALSE])

  set.seed(seed)
  sampleIdx  = sample(nrow(xUnlabelled), size = nDiscovery)
  xDiscovery = xUnlabelled[sampleIdx, , drop = FALSE]

  list(
    xFull       = xFull,
    xDiscovery  = xDiscovery,
    xReference  = xLabelled,
    yReference  = factor(yLabelled),
    sampleIndex = sampleIdx
  )
}

# Standardise features by median and MAD estimated on the discovery sample X.
# When out-of-sample observations (xNew) are passed, they are centred and scaled
# using the discovery medians and MADs rather than their own small-sample stats.
standardize = function(X, xNew = NULL) {
  med      = apply(X, 2, median)
  madScale = apply(X, 2, mad)

  zDiscovery = scale(X, center = med, scale = madScale)
  zNew = if (!is.null(xNew)) {
    scale(xNew, center = med, scale = madScale)
  } else {
    NULL
  }

  list(
    zDiscovery = as.matrix(zDiscovery),
    zNew       = if (!is.null(zNew)) as.matrix(zNew) else NULL,
    center     = med,
    scale      = madScale
  )
}

# R's mad() already includes the 1.4826 normal-consistency factor; r = SD / MAD.
scaleDiag = function(X, cFactor = 1) {
  sds  = apply(X, 2, sd)
  mads = apply(X, 2, mad)
  r    = sds / (cFactor * mads)

  data.frame(
    featureIndex = seq_len(ncol(X)),
    feature      = colnames(X),
    sd           = as.numeric(sds),
    mad          = as.numeric(mads),
    ratio        = as.numeric(r),
    status       = factor(
      ifelse(r > 1, "Inflated (r > 1)", "Stable (r <= 1)"),
      levels = c("Stable (r <= 1)", "Inflated (r > 1)")
    ),
    row.names = NULL
  )
}

# Scale diagnostic plots (log10 SD vs MAD, and ratio r_j)
plotScales = function(diagDf, cohortLabel) {
  p1 = ggplot(diagDf, aes(x = log10(sd), y = log10(mad), colour = status)) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", colour = "grey35", linewidth = 0.8) +
    geom_point(size = 2.8, alpha = 0.85) +
    scale_colour_manual(values = c("Stable (r <= 1)" = "#1f78b4", "Inflated (r > 1)" = "#e31a1c")) +
    labs(
      title    = paste0(cohortLabel, ": Classical SD vs Robust MAD"),
      subtitle = "Dashed line: log10(MAD) = log10(SD); points below line have outlier-inflated SD",
      x        = expression(log[10](SD)),
      y        = expression(log[10](MAD)),
      colour   = "Scale Regime"
    ) +
    themeBiostat()

  p2 = ggplot(diagDf, aes(x = featureIndex, y = ratio, colour = status)) +
    geom_hline(yintercept = 1.0, linetype = "dashed", colour = "#e31a1c", linewidth = 0.8) +
    geom_point(size = 2.8, alpha = 0.9) +
    scale_colour_manual(values = c("Stable (r <= 1)" = "#1f78b4", "Inflated (r > 1)" = "#e31a1c")) +
    labs(
      title    = paste0(cohortLabel, ": Scale Ratio Across 61 Variables"),
      subtitle = "r = SD / MAD (mad() is normal-consistent); r = 1 separates stable vs inflated features",
      x        = "Variable Index (1..61)",
      y        = "Scale Ratio r = SD / MAD",
      colour   = "Scale Regime"
    ) +
    themeBiostat()

  list(sdVsMad = p1, scaleRatio = p2)
}
