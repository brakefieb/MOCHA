args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(flag) {
  idx <- match(flag, args)
  if (is.na(idx) || idx == length(args)) {
    stop(paste("Missing arg:", flag))
  }
  args[idx + 1]
}

str_to_bool <- function(x) {
  tolower(x) %in% c("true", "1", "yes", "y")
}

manifest_path <- get_arg("--manifest")
outdir <- get_arg("--outdir")
method_root <- get_arg("--method_root")
cohort <- get_arg("--cohort")

k <- as.integer(get_arg("--k"))
w <- as.numeric(get_arg("--w"))
n_neighbor <- as.integer(get_arg("--n_neighbor"))
f_val <- as.numeric(get_arg("--f_val"))

gene_select <- get_arg("--gene_select")
n_gene <- as.integer(get_arg("--n_gene"))
pcn <- as.integer(get_arg("--pcn"))

save_posterior_rds <- str_to_bool(get_arg("--save_posterior_rds"))
save_posterior_z <- str_to_bool(get_arg("--save_posterior_z"))
save_trace_plot <- str_to_bool(get_arg("--save_trace_plot"))
save_mu_median <- str_to_bool(get_arg("--save_mu_median"))
save_omega_median <- str_to_bool(get_arg("--save_omega_median"))
save_posterior_summary <- str_to_bool(get_arg("--save_posterior_summary"))
save_sample_predictions <- str_to_bool(get_arg("--save_sample_predictions"))
mcmc_iter <- as.integer(get_arg("--mcmc_iter"))
store_thin <- as.integer(get_arg("--store_thin"))
n_workers <- as.integer(get_arg("--n_workers"))
if (is.na(n_workers) || n_workers < 1) {
  n_workers <- 1L
}

dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages({
  library(Rcpp)
  library(jsonlite)
  library(readr)
  library(mvtnorm)
  library(scater)
  library(scran)
  library(MASS)
  library(harmony)
  library(STdeconvolve)
  library(parallel)
})

sourceCpp(file.path(method_root, "BayeSMART.cpp"))
source(file.path(method_root, "BayeSMART.R"))

# Override the method helper with a numerically stable distance calculation.
# Tiny negative values can appear from floating point cancellation and otherwise
# become NaNs after sqrt(), which can propagate into neighbor matrices.
vectorized_pdist <- function(A, B) {
  A <- as.matrix(A)
  B <- as.matrix(B)
  an <- apply(A, 1, function(rvec) crossprod(rvec, rvec))
  bn <- apply(B, 1, function(rvec) crossprod(rvec, rvec))
  m <- nrow(A)
  n <- nrow(B)
  tmp <- matrix(rep(an, n), nrow = m)
  tmp <- tmp + matrix(rep(bn, m), nrow = m, byrow = TRUE)
  sqrt(pmax(tmp - 2 * tcrossprod(A, B), 0))
}

manifest <- jsonlite::fromJSON(manifest_path, simplifyVector = TRUE)

parallel_lapply <- function(X, FUN, ..., workers = n_workers, context = "parallel_lapply") {
  workers <- min(as.integer(workers), length(X))
  if (workers > 1 && .Platform$OS.type != "windows" && length(X) > 1) {
    results <- parallel::mclapply(X, FUN, ..., mc.cores = workers, mc.preschedule = TRUE)
    failed <- vapply(results, inherits, logical(1), "try-error")
    if (any(failed)) {
      message(
        context, " failed in parallel workers; rerunning serially to surface the first error."
      )
      return(lapply(X, FUN, ...))
    }
    results
  } else {
    lapply(X, FUN, ...)
  }
}

normalize_subgroup <- function(x) {
  if (is.null(x) || length(x) == 0 || is.na(x)) {
    return(NA_character_)
  }
  as.character(x)
}

read_one_sample <- function(i) {
  counts_df <- read.csv(manifest$counts_path[i], row.names = 1, check.names = FALSE)
  coords_df <- read.csv(manifest$coords_path[i], row.names = 1, check.names = FALSE)
  gt_df <- read.csv(manifest$ground_truth_path[i], row.names = 1, check.names = FALSE)

  common_ids <- Reduce(intersect, list(rownames(counts_df), rownames(coords_df), rownames(gt_df)))
  counts_df <- counts_df[common_ids, , drop = FALSE]
  coords_df <- coords_df[common_ids, , drop = FALSE]
  gt_df <- gt_df[common_ids, , drop = FALSE]

  list(
    sample_id = manifest$sample_id[i],
    subgroup = if ("subgroup" %in% colnames(manifest)) normalize_subgroup(manifest$subgroup[i]) else NA_character_,
    counts = {
      counts <- t(as.matrix(counts_df))   # genes x spots
      colnames(counts) <- paste(manifest$sample_id[i], colnames(counts), sep = "__")
      counts
    },
    xy = as.matrix(coords_df[, c("x", "y")]),
    gt = gt_df$ground_truth,
    spot_ids = rownames(coords_df)
  )
}

build_layers <- function(list_count) {
  genes_union <- Reduce(union, lapply(list_count, rownames))
  layers <- matrix(0, nrow = length(genes_union), ncol = 0,
                   dimnames = list(genes_union, NULL))
  each_length <- integer(length(list_count))

  for (s in seq_along(list_count)) {
    M1 <- as.matrix(list_count[[s]])
    aligned <- matrix(0, nrow = length(genes_union), ncol = ncol(M1),
                      dimnames = list(genes_union, colnames(M1)))
    aligned[rownames(M1), ] <- M1
    layers <- cbind(layers, aligned)
    each_length[s] <- ncol(M1)
  }

  list(layers = layers, each_length = each_length)
}

align_list_count_genes <- function(list_count) {
  genes_union <- Reduce(union, lapply(list_count, rownames))

  lapply(list_count, function(M) {
    M <- as.matrix(M)
    aligned <- matrix(
      0,
      nrow = length(genes_union),
      ncol = ncol(M),
      dimnames = list(genes_union, colnames(M))
    )
    aligned[rownames(M), ] <- M
    aligned
  })
}

safe_batch_remove <- function(list_count, xys = NULL, gene_select = "hvgs", n_gene = 2000, pcn = 3, n_workers = 1) {
  set.seed(42)
  list_count <- align_list_count_genes(list_count)
  cnt_all <- do.call(cbind, list_count)

  cnt_all <- scater::normalizeCounts(cnt_all, log = TRUE)
  each_length <- sapply(list_count, ncol)

  if (gene_select == "sparkx") {
    genes <- parallel_lapply(seq_along(each_length), function(l) {
      if (!requireNamespace("SPARK", quietly = TRUE)) {
        stop("Package SPARK is not installed but gene_select='sparkx' was requested.")
      }
      if (is.null(xys) || length(xys) < l) {
        stop("Missing spatial coordinates for SPARKx gene selection at sample index ", l)
      }
      if (ncol(list_count[[l]]) != nrow(xys[[l]])) {
        stop(
          "SPARKx input mismatch at sample index ", l, ": ",
          "ncol(counts)=", ncol(list_count[[l]]),
          ", nrow(xy)=", nrow(xys[[l]])
        )
      }
      capture.output(sparkx.l <- SPARK::sparkx(list_count[[l]], xys[[l]]))
      if (is.null(sparkx.l$res_mtest) || nrow(sparkx.l$res_mtest) == 0) {
        stop("SPARKx returned no testing results at sample index ", l)
      }
      sparkx.l <- sparkx.l$res_mtest[order(sparkx.l$res_mtest$adjustedPval), ]
      genes_l <- head(rownames(sparkx.l), n = n_gene)
      genes_l <- genes_l[!is.na(genes_l) & genes_l != ""]
      if (length(genes_l) == 0) {
        stop("SPARKx selected zero genes at sample index ", l)
      }
      genes_l
    }, workers = n_workers, context = "SPARKx gene selection")
    genes <- unique(unlist(genes))
    cnt_all <- cnt_all[intersect(genes, rownames(cnt_all)), , drop = FALSE]
  } else if (gene_select == "hvgs") {
    dec <- scran::modelGeneVar(cnt_all)
    genes <- scran::getTopHVGs(dec, n = min(n_gene, nrow(cnt_all)))
    cnt_all <- cnt_all[genes, , drop = FALSE]
  } else if (gene_select != "none") {
    stop("Unsupported gene_select: ", gene_select)
  }

  idx_rm <- rowSums(cnt_all) == 0
  cnt_all <- cnt_all[!idx_rm, , drop = FALSE]
  if (nrow(cnt_all) == 0) {
    stop("No genes with non-zero counts remain for batch_remove.")
  }

  scaled <- t(scale(t(as.matrix(cnt_all)), center = TRUE, scale = TRUE))
  scaled[!is.finite(scaled)] <- 0
  spot_by_gene <- t(scaled)

  pcn_use <- min(pcn, nrow(spot_by_gene), ncol(spot_by_gene))
  if (pcn_use < 1) {
    stop("Cannot compute PCA in batch_remove: not enough spots or genes.")
  }

  pca <- prcomp(spot_by_gene, center = FALSE, scale. = FALSE)
  cnt_all_f <- pca$x[, seq_len(pcn_use), drop = FALSE]
  if (pcn_use < pcn) {
    cnt_all_f <- cbind(
      cnt_all_f,
      matrix(0, nrow = nrow(cnt_all_f), ncol = pcn - pcn_use)
    )
  }

  batch_labels <- as.character(rep(seq_along(each_length), each_length))
  if (length(each_length) <= 1 || length(unique(batch_labels)) < 2) {
    message(
      "Skipping Harmony batch removal because this integration group has only one sample/batch."
    )
    cnt_all_f2 <- cnt_all_f
  } else {
    cnt_all_f2 <- harmony::HarmonyMatrix(
      data_mat = cnt_all_f,
      meta_data = batch_labels,
      do_pca = FALSE,
      verbose = FALSE
    )
  }

  cnt_all_f2 <- as.matrix(cnt_all_f2)
  if (nrow(cnt_all_f2) != sum(each_length)) {
    stop(
      "batch_remove returned ", nrow(cnt_all_f2),
      " rows, expected ", sum(each_length), " spots."
    )
  }

  cnt_all_f2
}

build_V_official <- function(list_count) {
  layer_info <- build_layers(list_count)
  layers <- layer_info$layers
  original_pixel_ids <- colnames(layers)

  counts <- cleanCounts(layers, min.lib.size = 1)
  corpus <- restrictCorpus(counts, removeAbove = 1.0, removeBelow = 0.05)

  corpus_mat <- as.matrix(corpus)
  lda_counts <- t(corpus_mat)
  nonzero_pixels <- rowSums(lda_counts) > 0
  if (!all(nonzero_pixels)) {
    message(
      "Dropping ", sum(!nonzero_pixels),
      " pixels with zero counts after restrictCorpus before fitLDA; ",
      "their BayeSMART V rows will be filled with zeros."
    )
  }
  if (!any(nonzero_pixels)) {
    stop("No pixels with non-zero counts remain after restrictCorpus; cannot run STdeconvolve fitLDA.")
  }

  ldas <- fitLDA(lda_counts[nonzero_pixels, , drop = FALSE], Ks = c(5, 6, 7, 8, 9, 10))
  optLDA <- optimalModel(models = ldas, opt = "min")
  results <- getBetaTheta(optLDA, perc.filt = 0.05, betaScale = 1000)
  deconProp <- results$theta
  V_nonzero <- round(as.matrix(deconProp) * 100)
  kept_pixel_ids <- rownames(lda_counts)
  kept_full_idx <- match(kept_pixel_ids, original_pixel_ids)
  if (any(is.na(kept_full_idx))) {
    stop("Could not map cleanCounts/restrictCorpus pixels back to the original spot order.")
  }

  V <- matrix(
    0,
    nrow = length(original_pixel_ids),
    ncol = ncol(V_nonzero),
    dimnames = list(original_pixel_ids, colnames(V_nonzero))
  )
  V[kept_full_idx[nonzero_pixels], ] <- V_nonzero
  V
}

build_G_official <- function(xys, subgroups, n_neighbor, n_workers = 1) {
  G <- matrix(0, nrow = 0, ncol = n_neighbor)
  G_origin <- vector("list", length(xys))
  each_length <- integer(length(xys))

  per_sample <- parallel_lapply(seq_along(xys), function(s) {
    spot_xy <- as.matrix(xys[[s]])
    G1 <- get.neighbor(spot_xy, n_neighbor)
    list(G = G1, n = nrow(spot_xy))
  }, workers = n_workers)

  for (s in seq_along(per_sample)) {
    G1 <- per_sample[[s]]$G
    each_length[s] <- per_sample[[s]]$n
    G_origin[[s]] <- G1

    if (s > 1) {
      non_zero_indices <- G1 != 0
      G1[non_zero_indices] <- G1[non_zero_indices] + sum(each_length[1:(s - 1)])
    }

    G <- rbind(G, G1)
  }

  # Official DLPFC tutorial manually adds cross-slide adjacency only within
  # adjacent slides from the same individual. Here we replicate that logic using
  # subgroup as the donor/individual identifier when available.
  G_extra <- matrix(0, nrow = sum(each_length), ncol = 2)
  row_offset <- cumsum(c(0, head(each_length, -1)))

  if (length(xys) > 1) {
    for (s in seq_len(length(xys) - 1)) {
      if (is.na(subgroups[s]) || subgroups[s] == "" ||
          is.na(subgroups[s + 1]) || subgroups[s + 1] == "") {
        next
      }
      subgroup_s <- subgroups[s]
      subgroup_t <- subgroups[s + 1]
      if (subgroup_s != subgroup_t) {
        next
      }

      xy_s <- as.matrix(xys[[s]])
      xy_t <- as.matrix(xys[[s + 1]])

      key_s <- paste(xy_s[, 1], xy_s[, 2], sep = "__")
      key_t <- paste(xy_t[, 1], xy_t[, 2], sep = "__")
      match_t <- match(key_s, key_t)
      match_s <- match(key_t, key_s)

      rows_s <- row_offset[s] + seq_len(nrow(xy_s))
      rows_t <- row_offset[s + 1] + seq_len(nrow(xy_t))

      valid_s <- which(!is.na(match_t))
      if (length(valid_s) > 0) {
        G_extra[rows_s[valid_s], 1] <- rows_t[match_t[valid_s]]
      }

      valid_t <- which(!is.na(match_s))
      if (length(valid_t) > 0) {
        G_extra[rows_t[valid_t], 2] <- rows_s[match_s[valid_t]]
      }
    }
  }

  G_full <- cbind(G, G_extra)
  list(G = G_full, G_origin = G_origin)
}

sanitize_G <- function(G, n_spot) {
  G <- as.matrix(G)
  G[!is.finite(G)] <- 0
  G <- round(G)
  G[G < 0 | G > n_spot] <- 0
  storage.mode(G) <- "integer"
  G
}

validate_bayesmart_inputs <- function(V, Y, G) {
  n_spot <- nrow(Y)
  if (nrow(V) != n_spot || nrow(G) != n_spot) {
    stop(
      "BayeSMART input row mismatch: ",
      "nrow(V)=", nrow(V), ", ",
      "nrow(Y)=", nrow(Y), ", ",
      "nrow(G)=", nrow(G), "."
    )
  }
  if (any(!is.finite(V)) || any(!is.finite(Y))) {
    stop("BayeSMART inputs contain non-finite values in V or Y.")
  }
  invisible(TRUE)
}

sample_data <- lapply(seq_len(nrow(manifest)), read_one_sample)

list_count <- align_list_count_genes(lapply(sample_data, function(x) x$counts))
xys <- lapply(sample_data, function(x) x$xy)
subgroups <- vapply(sample_data, function(x) x$subgroup, character(1))

# Official DLPFC-style preprocessing for all cohorts in this benchmark:
#   V from STdeconvolve on stacked counts,
#   G from within-slide neighbors + adjacent-slide same-(x,y) links within subgroup,
#   Y from batch_remove.
V <- build_V_official(list_count)

spatial_info <- build_G_official(
  xys = xys,
  subgroups = subgroups,
  n_neighbor = n_neighbor,
  n_workers = n_workers
)
G <- spatial_info$G
G_origin <- spatial_info$G_origin

Y <- safe_batch_remove(
  list_count = list_count,
  xys = xys,
  gene_select = gene_select,
  n_gene = n_gene,
  pcn = pcn,
  n_workers = n_workers
)
G <- sanitize_G(G, nrow(Y))
validate_bayesmart_inputs(V, Y, G)

cat("BayeSMART settings:\n")
cat("  cohort:", cohort, "\n")
cat("  k:", k, "\n")
cat("  w:", w, "\n")
cat("  n_neighbor:", n_neighbor, "\n")
cat("  pcn:", pcn, "\n")
cat("  mcmc_iter:", mcmc_iter, "\n")
cat("  store_thin:", store_thin, "\n")
cat("  preprocessing_workers:", n_workers, "\n")
flush.console()

t0 <- proc.time()

result <- run.BayeSMART(
  V = V,
  Y = Y,
  G = G,
  n_cluster = k,
  f_val = f_val,
  w = w,
  mcmc_iter = mcmc_iter,
  store_thin = store_thin
)

elapsed <- (proc.time() - t0)[["elapsed"]]

spatial_domain <- get.spatial.domain(result)
z_all <- domain_split(spatial_domain, G_origin)

if (save_posterior_rds) {
  saveRDS(result, file.path(outdir, "posterior_result.rds"))
}

if (save_posterior_z) {
  write.csv(
    as.data.frame(result$Z),
    file.path(outdir, "posterior_Z.csv"),
    row.names = FALSE
  )
}

get_mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

if (save_posterior_summary) {
  posterior_mode <- apply(result$Z, 2, get_mode)
  posterior_summary <- data.frame(
    posterior_mode = posterior_mode
  )

  write.csv(
    posterior_summary,
    file.path(outdir, "posterior_summary.csv"),
    row.names = FALSE
  )
}

all_predictions <- list()

for (i in seq_along(sample_data)) {
  s <- sample_data[[i]]
  sample_id <- s$sample_id

  pred_df <- data.frame(
    cohort = cohort,
    sampleID = sample_id,
    spotID = s$spot_ids,
    x = s$xy[, 1],
    y = s$xy[, 2],
    method = "BayeSMART",
    z = as.character(s$gt),
    z_pred = as.character(z_all[[i]]),
    stringsAsFactors = FALSE
  )

  if (save_sample_predictions) {
    sample_outdir <- file.path(outdir, sample_id)
    dir.create(sample_outdir, recursive = TRUE, showWarnings = FALSE)
    write.csv(
      pred_df,
      file.path(sample_outdir, "predictions.csv"),
      row.names = FALSE
    )
  }

  all_predictions[[i]] <- pred_df
}

predictions <- do.call(rbind, all_predictions)
write.csv(
  predictions,
  file.path(outdir, "predictions.csv"),
  row.names = FALSE
)

if (save_trace_plot) {
  png(file.path(outdir, "trace_plot.png"), width = 1400, height = 900)
  trace_n <- min(6, ncol(result$Z))
  matplot(
    t(result$Z[, seq_len(trace_n), drop = FALSE]),
    type = "l",
    lty = 1,
    xlab = "Iteration",
    ylab = "Cluster label",
    main = paste("BayeSMART trace plot:", cohort)
  )
  dev.off()
}

if (save_omega_median && !is.null(result$omega)) {
  omega_dims <- dim(result$omega)
  if (length(omega_dims) == 3) {
    omega_median <- apply(result$omega, c(2, 3), median)
    write.csv(
      omega_median,
      file.path(outdir, "omega_median.csv"),
      row.names = TRUE
    )
  }
}

if (save_mu_median && !is.null(result$mu)) {
  mu_dims <- dim(result$mu)
  if (length(mu_dims) == 3) {
    mu_median <- apply(result$mu, c(2, 3), median)
    write.csv(
      mu_median,
      file.path(outdir, "mu_median.csv"),
      row.names = TRUE
    )
  }
}

cat("BayeSMART cohort-level run finished successfully.\n")
cat("Cohort:", cohort, "\n")
cat("Output:", outdir, "\n")
cat("Elapsed seconds:", elapsed, "\n")
