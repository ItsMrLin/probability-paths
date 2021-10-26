source("src/glim/pkgs.R")
source("src/glim/consts.R")

options(max.print = 100)

# nba or weather
args <- commandArgs(trailingOnly = TRUE)
dataset_name = args[1]

dataset_dir_path <- file.path(INPUT_PATH, dataset_name)

for (train_or_test in c("train", "test")) {
  feature_path <- file.path(dataset_dir_path, paste0(train_or_test, "_feature.csv"))
  path_path <- file.path(dataset_dir_path, paste0(train_or_test, "_path.csv"))
  
  feature_df <- read_csv(feature_path)
  path_df <- read_csv(path_path)
  
  if (dataset_name == "nba") {
    # rename nba columns
    feature_df <- feature_df %>% 
      rename(obs = GAME_ID) %>%
      transmute(
        obs = obs,
        abs_win_rate_diff = abs(home_win_rate - visitor_win_rate),
        abs_margin = abs(score_margin_last_1),
        margin_std = score_margin_std,
        pred_outcome_std = pred_outcome_std,
        pred_outcome = pred_outcome_last_1,
        intercept = intercept
      )
    
    path_df <- path_df %>% 
      rename(obs = GAME_ID) %>% 
      select(obs, time, target_outcome, pred_outcome)
    
    source("src/glim/nba_const.R")
  } else if (dataset_name == "weather") {
    source("src/glim/weather_const.R")
  } else {
    print("Unknown dataset!")
    exit()
  }
  
  col2drop <- c("X1", "time", "home_score_min")
  for (c in col2drop) {
    if (c %in% colnames(feature_df)) {
      feature_df <- feature_df %>% 
        select(-!!sym(c))
    }
  }

  obs_order_df <- path_df %>% 
    select(obs) %>% 
    unique() %>% 
    arrange(obs)
  
  n_obs <- nrow(obs_order_df)
  X_df <- obs_order_df %>%
    left_join(feature_df, on=obs) %>% 
    select(-obs)
  K <- ncol(X_df)
  X_mat <- as.matrix(X_df, nrow=n_obs, ncol=K)
  
  if (dataset_name == "weather") {
    # weather    
    binary_X <- X_df %>% select(matches("Location_|month_|RainToday_|rain_|intercept"))
    numeric_X <- X_df %>% select(-matches("Location_|month_|RainToday_|rain_|intercept"))
  } else {
    # otherwise assume everything is numeric except column "intercept"
    binary_X <- X_df %>% select(matches("intercept"))
    numeric_X <- X_df %>% select(-matches("intercept"))
  }
  
  binary_X <- as.matrix(binary_X, nrow=n_obs)
  numeric_X <- as.matrix(numeric_X, nrow=n_obs)
  
  if (train_or_test == "train") {
    std_numeric_X_mat <- scale(numeric_X)
    X_center <- attr(std_numeric_X_mat, "scaled:center")
    X_scale <- attr(std_numeric_X_mat, "scaled:scale")
  } else {
    std_numeric_X_mat <- scale(numeric_X, center = X_center, scale = X_scale)
    X_center <- NULL
    X_scale <- NULL
  }
  
  std_X_mat <- cbind(binary_X, std_numeric_X_mat)
  K <- ncol(std_X_mat)
  
  y0 <- obs_order_df %>% 
    left_join(path_df %>% filter(time == init_time), by = "obs") %>% 
    pull(pred_outcome)
  
  y_T <- obs_order_df %>% 
    left_join(path_df %>% filter(time == target_time), by = "obs") %>% 
    pull(pred_outcome)
  
  y <- obs_order_df %>% 
    left_join(path_df %>% filter(time > init_time, time < target_time), by = "obs") %>% 
    arrange(obs, time) %>% 
    pull(pred_outcome)
  y = matrix(y, nrow=n_obs, ncol=(target_time - init_time - 1), byrow = TRUE)
  
  
  info_flow_mod_data <- list(
    obs_id = obs_order_df$obs,
    X_names = colnames(std_X_mat),
    X = std_X_mat,
    K = K,
    n_obs = n_obs,
    T_ = time_left,
    y0 = y0,
    y = y,
    y_T =  y_T
  )
  
  stan_data_path <- file.path(dataset_dir_path, paste0(train_or_test, ".rds"))
  message(stan_data_path)
  saveRDS(info_flow_mod_data, stan_data_path)
}

