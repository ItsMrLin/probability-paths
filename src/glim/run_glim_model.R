source("src/glim/pkgs.R")
source("src/glim/consts.R")
source("src/glim/glim_modeling.R")

options(max.print = 100)

args <- commandArgs(trailingOnly = TRUE)
# nba or weather
dataset_name = args[1]
# "map" or "mcmc"
method = args[2]

data_path <- file.path(INPUT_PATH, dataset_name, "train.rds")
stan_mod_path <- file.path(OUTPUT_PATH, dataset_name, paste0("stan_", method, ".rds"))
if (dataset_name == "nba") {
  stan_file_name <- "src/glim/glim_nba.stan"
} else if (dataset_name == "weather") {
  stan_file_name <- "src/glim/glim_weather.stan"
} else {
  print("Unknown dataset!")
  exit()
}

# ==== load data and stan models ====
train_data <- readRDS(data_path)
if (method == "map") {
  # ==== MAP Estimation ====
  glim_map <- fit_glim(stan_file_name = stan_file_name,
                       train_data = train_data, 
                       method = "map")
  saveRDS(glim_map, stan_mod_path)
} else if (method == "mcmc") {
  # ==== MCMC sampling ====
  glim_mcmc <- fit_glim(stan_file_name = stan_file_name,
                        train_data = train_data,
                        method="mcmc")
  saveRDS(glim_mcmc, stan_mod_path)
} else {
  warning("Unsupported fitting method!")
}



