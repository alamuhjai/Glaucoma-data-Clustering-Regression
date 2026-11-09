# R/classmap.R
# Class Map implementation following Raymaekers, Rousseeuw & Hubert (2022,
# Technometrics 64:151-165): QR least-squares multiclass scores, softmax
# posterior probabilities, PAC, empirical Mahalanobis farness, and Class Map plots.

classmap = function(X, y, ridge = 1e-6) {
  y       = as.factor(y)
  labels  = levels(y)
  xScaled = scale(as.matrix(X))
  xDesign = cbind(1, xScaled)
  n       = nrow(xDesign)
  K       = length(labels)

  # One-vs-rest linear scoring via QR decomposition
  B = matrix(0, nrow = ncol(xDesign), ncol = K)
  for (k in seq_len(K)) {
    B[, k] = qr.solve(xDesign, as.numeric(y == labels[k]))
  }

  # Row-shifted softmax for numerical stability
  rawScores = xDesign %*% B
  shifted   = rawScores - apply(rawScores, 1, max)
  expScores = exp(shifted)
  posterior = expScores / rowSums(expScores)
  colnames(posterior) = labels

  # Probability of the Alternative Class (PAC): ratio of best alternative
  # class probability to (assigned class probability + best alternative)
  pac     = numeric(n)
  bestAlt = character(n)
  for (i in seq_len(n)) {
    k          = as.integer(y[i])
    altProbs   = posterior[i, -k]
    altMax     = max(altProbs)
    pac[i]     = altMax / (posterior[i, k] + altMax)
    bestAlt[i] = labels[-k][which.max(altProbs)]
  }

  # Within-class squared Mahalanobis distance (small ridge guards against
  # near-singular class covariances) mapped to empirical CDF ranks in (0, 1]
  distSq = numeric(n)
  for (i in seq_len(n)) {
    Xi        = xScaled[y == y[i], , drop = FALSE]
    covI      = cov(Xi) + diag(ridge, ncol(Xi))
    distSq[i] = mahalanobis(xScaled[i, ], colMeans(Xi), covI)
  }

  farness = numeric(n)
  for (g in labels) {
    idx          = which(y == g)
    farness[idx] = rank(distSq[idx]) / length(idx)
  }

  quadrant = ifelse(
    pac <= 0.5 & farness <= 0.75, "Typical (Core)",
    ifelse(
      pac > 0.5 & farness <= 0.75, "Boundary Overlap",
      ifelse(pac <= 0.5 & farness > 0.75, "Benign Outlier", "Far & Ambiguous")
    )
  )

  obsDf = data.frame(
    observation   = seq_len(n),
    assignedClass = y,
    alternative   = factor(bestAlt, levels = labels),
    pac           = pac,
    mahalanobisD2 = distSq,
    farness       = farness,
    zone          = factor(quadrant, levels = c(
      "Typical (Core)", "Boundary Overlap", "Benign Outlier", "Far & Ambiguous"
    ))
  )

  summaryDf = data.frame(
    Class           = labels,
    n               = as.integer(table(y)),
    MeanPAC         = as.numeric(tapply(pac, y, mean)),
    MedianFarness   = as.numeric(tapply(farness, y, median)),
    BoundaryCases   = as.integer(tapply(pac > 0.5, y, sum)),
    FarAndAmbiguous = as.integer(tapply(pac > 0.5 & farness > 0.75, y, sum)),
    stringsAsFactors = FALSE
  )

  list(
    observations = obsDf,
    summary      = summaryDf,
    posterior    = posterior
  )
}

# Class Map scatter of PAC vs within-class Mahalanobis farness
plotClassmap = function(classMapObj, titleText) {
  dfObs = classMapObj$observations
  nCls  = length(levels(dfObs$assignedClass))
  pal   = c("#1f78b4", "#e31a1c", "#33a02c", "#6a3d9a")[seq_len(nCls)]

  ggplot(dfObs, aes(x = farness, y = pac, colour = assignedClass, shape = assignedClass)) +
    geom_hline(yintercept = 0.50, linetype = "dashed", colour = "grey30", linewidth = 0.8) +
    geom_vline(xintercept = 0.75, linetype = "dotted", colour = "grey45", linewidth = 0.8) +
    geom_point(size = 3.0, alpha = 0.85) +
    scale_colour_manual(values = pal) +
    scale_x_continuous(limits = c(0, 1)) +
    scale_y_continuous(limits = c(0, 1)) +
    labs(
      title    = titleText,
      subtitle = "Dashed line: PAC = 0.5 boundary | Dotted line: 75th percentile within-cluster farness",
      x        = "Within-Cluster Mahalanobis Farness F_g(D(i, g_i))",
      y        = "Probability of Alternative Class (PAC)",
      colour   = "Cluster",
      shape    = "Cluster"
    ) +
    themeBiostat()
}
