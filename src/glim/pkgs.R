# Load packages and setup environment
# Changes made to this script should not invalidate build cache!
PKGS <- c(
  "ROCR",
  "randomForest",
  "truncnorm",
  "raster",
  "lubridate",
  "furrr",
  "recipes",
  "boot",
  "rstan",
  "cmdstanr",
  "tidyverse",
  "optimr",
  "cubature",
  "mvtnorm",
  "randtoolbox",
  "matrixcalc",
  "Cairo",
  "xtable"
)

suppressMessages(
  PKGS <- sapply(PKGS, library, character.only = TRUE, logical.return = TRUE)
)

if (any(PKGS == FALSE)) {
  print(PKGS)
  stop("Failed loading packages.")
}


message("Loaded packages:", paste(names(PKGS), collapse = ", "))

N_CORES <- min(48, availableCores())
message(sprintf("Futures planned for %s cores", N_CORES))
if (!interactive()) {
  plan(multiprocess(workers = eval(N_CORES)))  # Plan for futures
}
options(mc.cores = N_CORES)

# # for rstan "Error in readRDS(file.rds) : error reading from connection"
# # https://discourse.mc-stan.org/t/error-in-readrds-file-rds-error-reading-from-connection-calls-stan-model-is-readrds-execution-halted/3202/8
# rstan_options(auto_write = TRUE)

theme_set(theme_bw())

select <- dplyr::select
