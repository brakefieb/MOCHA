
options(repos=c(CRAN="https://cloud.r-project.org"))

if (!requireNamespace("BiocManager", quietly=TRUE)) {
  install.packages("BiocManager")
}

install_if_missing <- function(pkgs) {
  ip <- rownames(installed.packages())
  missing <- pkgs[!(pkgs %in% ip)]
  if (length(missing) > 0) {
    install.packages(missing)
  } else {
    cat("CRAN packages already installed.\n")
  }
}

cran_pkgs <- c(
  "Rcpp", "RcppArmadillo", "RcppDist",
  "jsonlite", "harmony", "remotes"
)

install_if_missing(cran_pkgs)

# Bioconductor packages
bioc_pkgs <- c(
  "scater", "scran"
)

ip <- rownames(installed.packages())
missing_bioc <- bioc_pkgs[!(bioc_pkgs %in% ip)]

if (length(missing_bioc) > 0) {
  BiocManager::install(missing_bioc, ask=FALSE, update=FALSE)
} else {
  cat("Bioconductor packages already installed.\n")
}

# Install STdeconvolve from GitHub if missing
if (!("STdeconvolve" %in% rownames(installed.packages()))) {
  remotes::install_github("JEFworks-Lab/STdeconvolve", ref="package", upgrade="never", dependencies=TRUE)
} else {
  cat("STdeconvolve already installed.\n")
}

# Install SPARK from GitHub if missing.
# BayeSMART calls SPARK::sparkx(...) when gene_select = "sparkx".
if (!("SPARK" %in% rownames(installed.packages()))) {
  remotes::install_github("xzhoulab/SPARK", upgrade="never", dependencies=TRUE)
} else {
  cat("SPARK already installed.\n")
}
