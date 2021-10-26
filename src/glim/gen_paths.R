source("src/glim/pkgs.R")
source("src/glim/glim_modeling.R")
source("src/glim/consts.R")

options(max.print = 200)
args <- commandArgs(trailingOnly = TRUE)
# "nba" or "weather"
dataset_name = args[1]
# "map" or "mcmc"
method = args[2]

test_data_path <- file.path(INPUT_PATH, dataset_name, "test.rds")
stan_mod_path <- file.path(OUTPUT_PATH, dataset_name, paste0("stan_", method, ".rds"))
sample_paths_path <- file.path(OUTPUT_PATH, dataset_name, "glim_sample_paths.rds")

if (dataset_name == "nba") {
  source("src/glim/nba_const.R")
  covar_func <- calc_covar_inv_logit_beta_nba
} else if (dataset_name == "weather") {
  source("src/glim/weather_const.R")
  covar_func <- calc_covar_inv_logit_beta
} else {
  print("Unknown dataset!")
  exit()
}

# ===== read test data =====
test_info_data <- readRDS(test_data_path)

# ===== load fitted GLIM model =====
stan_mod <- readRDS(stan_mod_path)
if (method == "map") {
  coefs <- stan_mod$summary() %>% 
    # filter(!(variable %in% c("lp__", "atanh_rho")))
    filter(!(variable %in% c("lp__", "atanh_rho")), !(str_detect(variable, "pre_tsfm_sigma2")))
  coef_samples <- matrix(coefs$estimate, ncol=nrow(coefs))
  colnames(coef_samples) <- coefs$variable
  print(coef_samples)
  # print(tail(coef_samples))
} else if (method == "mcmc") {
  model_fit_summary <- stan_mod$summary()
  fitted_coefs <- model_fit_summary %>% 
    # filter(!(variable %in% c("lp__", "atanh_rho"))) %>% 
    filter(!(variable %in% c("lp__", "atanh_rho")), !(str_detect(variable, "pre_tsfm_sigma2")))
    select(variable, mean)
  
  # stan fit MCMC draws
  coef_samples <- stan_mod$draws()[,,fitted_coefs$variable]
  coef_samples <- matrix(coef_samples, nrow=prod(dim(coef_samples)[1:2]))
  colnames(coef_samples) <- fitted_coefs$variable
} else {
  print("Unsupported method!")
  exit()
}


# ===== gen sample paths =====
n_obs <- test_info_data$n_obs

t_now <- proc.time()
mcmc_gen_paths <- future_map_dfr(1:N_SAMPLES,
                          function(sample_id) {
                            chosen_coef_samples <- coef_samples[sample(nrow(coef_samples), size=n_obs, replace=TRUE),]
                            covars <- covar_func(test_info_data$X, chosen_coef_samples, T_ = time_left)
                            single_gen_paths <- map_dfr(1:n_obs, function(i) {
                              y0 <- test_info_data$y0[i]
                              covar <- covars[i,,]
                              obs_id <- test_info_data$obs_id[i]
                              end_time <- init_time + test_info_data$T_
                              tibble(
                                obs_id =  obs_id,
                                time = init_time:end_time,
                                pred_outcome = c(y0, ry(y0, covar))
                              )
                            })
                            single_gen_paths$sample_id <- sample_id
                            single_gen_paths
                          })
                          # .options = future_options(scheduling=4))
proc.time() - t_now

message(sample_paths_path)
saveRDS(mcmc_gen_paths, sample_paths_path)