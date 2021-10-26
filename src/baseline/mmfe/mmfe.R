source("src/glim/pkgs.R")
source("src/glim/consts.R")
source("src/glim/util.R")

# nba or weather
# dataset_name = args[1]
for (dataset_name in c("nba", "weather")) {
  
  if (dataset_name == "nba") {
    source("src/glim/nba_const.R")
  } else if (dataset_name == "weather") {
    source("src/glim/weather_const.R")
  } else {
    print("Unknown dataset!")
    exit()
  }
  
  train_data_path <- file.path(INPUT_PATH, dataset_name, "train.rds")
  test_data_path <- file.path(INPUT_PATH, dataset_name, "test.rds")
  sample_paths_path <- file.path(OUTPUT_PATH, dataset_name, "mmfe_sample_paths.rds")
  
  train_data <- readRDS(train_data_path)
  test_data <- readRDS(test_data_path)
  
  lag_y <- cbind(train_data$y0, train_data$y)
  curr_y <- cbind(train_data$y, train_data$y_T)
  y_update <- curr_y - lag_y
  vcv <- cov(y_update)
 
  mmfe_paths <- map_dfr(1:test_data$n_obs, function(i) {
    y0 <- test_data$y0[i]
    covar <- vcv
    obs_id <- test_data$obs_id[i]
    end_time <- init_time + test_data$T_
    tibble(
      sample_id = rep(1:N_SAMPLES, each = (test_data$T_ + 1)),
      obs_id = rep(obs_id, N_SAMPLES * (test_data$T_ + 1)),
      time = rep(init_time:target_time, times = N_SAMPLES),
      updates = c(t(cbind(rep(y0, times = N_SAMPLES), rmvnorm(N_SAMPLES, sigma=vcv))))
    )
  })
  
  mmfe_paths <- mmfe_paths %>% 
    group_by(obs_id, sample_id) %>% 
    arrange(time) %>% 
    mutate(
      pred_outcome = clip01(cumsum(updates), 1e-6)
    ) %>% 
    ungroup() %>% 
    select(time, sample_id, obs_id, pred_outcome) %>% 
    arrange(obs_id, sample_id)
    
  message(sample_paths_path)
  saveRDS(mmfe_paths, sample_paths_path)
}
