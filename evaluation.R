source("src/glim/pkgs.R")
source("src/glim/util.R")
# source("src/glim/glim_modeling.R")
source("src/glim/consts.R")
source("eval_functions.R")

options(max.print = 200)
args <- commandArgs(trailingOnly = TRUE)

# nba or weather
dataset_name = args[1]
# # For debugging
# dataset_name <- "nba"
# dataset_name <- "weather"

# ============= Load and preprocess data ===============
dataset_dir_path <- file.path(INPUT_PATH, dataset_name)
train_path <- file.path(dataset_dir_path, "train_path.csv")
test_path <- file.path(dataset_dir_path, "test_path.csv")
glim_sample_path <- file.path(OUTPUT_PATH, dataset_name, "glim_sample_paths.rds")
mmfe_sample_path <- file.path(OUTPUT_PATH, dataset_name, "mmfe_sample_paths.rds")
mqlstm_sample_path <- file.path(OUTPUT_PATH, dataset_name, "mqlstm.rds")
lr_sample_path <- file.path(OUTPUT_PATH, dataset_name, "lr.rds")

if (dataset_name == "nba") {
  source("src/glim/nba_const.R")
  var_filter_window_size <- 6
} else if (dataset_name == "weather") {
  source("src/glim/weather_const.R")
  var_filter_window_size <- 4
}

glim_paths <- readRDS(glim_sample_path)
glim_paths_init <- glim_paths %>% filter(time == init_time)

mmfe_paths <- readRDS(mmfe_sample_path)
mmfe_paths <- read_and_clean_sample_paths(mmfe_paths, glim_paths_init)

# If csv, save it as rds
if (!file.exists(lr_sample_path)) {
  lr_csv_path <- file.path(OUTPUT_PATH, dataset_name, "lr.csv")
  lr_paths <- read_csv(lr_csv_path)
  message("Saving LR samples into RDS for smaller file size")
  saveRDS(lr_paths, lr_sample_path)
} else {
  lr_paths <- readRDS(lr_sample_path)
}
lr_paths <- read_and_clean_sample_paths(lr_paths, glim_paths_init)

if (!file.exists(mqlstm_sample_path)) {
  mqlstm_csv_path <- file.path(OUTPUT_PATH, dataset_name, "mqlstm.csv")
  mqlstm_paths <- read_csv(mqlstm_csv_path)
  message("Saving MQLSTM samples into RDS for smaller file size")
  saveRDS(mqlstm_paths, mqlstm_sample_path)
} else {
  mqlstm_paths <- readRDS(mqlstm_sample_path)
}
mqlstm_paths <- read_and_clean_sample_paths(mqlstm_paths, glim_paths_init)

ground_truth <- read_csv(test_path) %>% 
  rename(
    obs_id = obs,
    true_pred_outcome = pred_outcome
  ) %>% 
  select(
    obs_id,
    time,
    true_pred_outcome
  )

init_prob <- ground_truth %>% 
  filter(time == init_time) %>% 
  select(obs_id, init_prob = true_pred_outcome)

outcomes <- ground_truth %>% 
  filter(time == target_time) %>% 
  select(obs_id, outcome = true_pred_outcome)

ground_truth <- ground_truth %>% 
  left_join(init_prob, by=c("obs_id")) %>% 
  left_join(outcomes, by=c("obs_id")) %>% 
  group_by(obs_id, time) %>% 
  ungroup()

sample_paths <- list(
  paths = list(mmfe_paths, lr_paths, mqlstm_paths, glim_paths),
  names = c("mmfe", "lr", "mqlstm", "glim")
)


paths <- bind_rows(
    mmfe_paths %>% mutate(method = "mmfe"), 
    lr_paths %>% mutate(method = "lr"), 
    mqlstm_paths %>% mutate(method = "mqlstm"), 
    glim_paths %>% mutate(method = "glim") 
  ) %>% 
  left_join(ground_truth, by = c("obs_id", "time")) %>% 
  mutate(
    method = recode_factor(method,
                           mmfe="MMFE", 
                           lr="LR", 
                           mqlstm="MQLSTM", 
                           glim="GLIM")
  ) 

# ================ Evaluation 1: Evaluate mean predicted probability =======================
init_prob_mse <- tibble()
for (i in 1:length(sample_paths$paths)) {
  single_smp_paths <- sample_paths$paths[[i]]
  single_name <- sample_paths$names[i]
  single_init_prob_mse <- calc_init_prob_mse(single_smp_paths, ground_truth, 
                                             init_time = init_time,
                                             target_time = target_time,
                                             method = single_name)
  init_prob_mse <- bind_rows(init_prob_mse, single_init_prob_mse)
}

init_prob_mse %>% 
  group_by(method) %>% 
  summarize(
    avg_mse = mean(mse),
    .groups = "keep"
  ) %>% 
  arrange(avg_mse)

calib_plot_data_filepath = file.path(EXP_RESULTS_PATH, paste0(dataset_name, "_calib_mse.rds"))
saveRDS(init_prob_mse, calib_plot_data_filepath)


# ================ Evaluation 2: Check excessive movement =======================
# to ensure Q_T = y0(1 - y0)

excessive_move <- paths %>%
  group_by(method, obs_id, sample_id) %>%
  mutate(
    change_sq = (pred_outcome - lag(pred_outcome))^2
  ) %>% 
  drop_na(change_sq) %>% 
  summarize(
    init_prob = first(init_prob),
    Q_T = sum(change_sq),
    .groups = "keep"
  ) %>% 
  ungroup()

excessive_move <- excessive_move %>% 
  group_by(method, obs_id) %>% 
  summarize(
    estimated_Q_T = mean(Q_T),
    expected_change = first(init_prob)*(1 - first(init_prob)),
    Q_T_sq_err = (estimated_Q_T - expected_change)^2,
    .groups = "keep"
  ) %>% 
  ungroup()

excessive_move <- excessive_move %>% 
  group_by(method) %>%
  summarize(
    Q_T_mse = mean(Q_T_sq_err),
    Q_T_se = sd(Q_T_sq_err)/sqrt(n())
  )

excessive_move_data_filepath = file.path(EXP_RESULTS_PATH, paste0(dataset_name, "_excessive_move.rds"))
saveRDS(excessive_move, excessive_move_data_filepath)


# ================ Evaluation 3: Coverage check ==================
coverage_summary <- tibble()
for (z in c(0.5, 0.8, 0.9, 0.95)) {
  # if (dataset_name == "nba") {
  #   mod_n <- 100
  # } else {
  #   mod_n <- 100
  # }
  coverage <- paths %>%
    # filter(obs_id %% mod_n == 0) %>%
    group_by(method, obs_id, time) %>% 
    summarize(
      ub = quantile(pred_outcome, 1 - (1-z)/2),
      lb = quantile(pred_outcome, (1-z)/2),
      true_pred_outcome = first(true_pred_outcome),
      covered = (lb <= true_pred_outcome) & (ub >= true_pred_outcome),
      .groups = "keep"
    )
  
  z_coverage_summary <- coverage %>% 
    filter(time != init_time, time != target_time) %>% 
    group_by(method, time) %>% 
    summarize(
      coverage = mean(covered),
      se = sd(covered) / sqrt(n()),
      coverage_err = coverage - z,
      err_ub = coverage_err + se * 1.96,
      err_lb = coverage_err - se * 1.96,
      z = z,
      .groups = "keep"
    ) %>% 
    arrange(time)
  
  coverage_summary <- bind_rows(coverage_summary, z_coverage_summary)
}

coverage_summary_data_path = file.path(EXP_RESULTS_PATH, paste0(dataset_name, "_coverage_summary.rds"))
saveRDS(coverage_summary, coverage_summary_data_path)

# ============== Check outcome model calibration (Figure 8) ==========
outcome_model_calib <- ground_truth %>%
  mutate(
    pred_bucket = round(true_pred_outcome * 10) / 10
  ) %>%
  group_by(pred_bucket) %>%
  summarize(
    prec_outcome = mean(outcome),
    .groups = "keep"
  )
outcome_model_calib_filepath = file.path(EXP_RESULTS_PATH, paste0(dataset_name, "_outcome_model_calib.rds"))
saveRDS(outcome_model_calib, outcome_model_calib_filepath)

# === Visualization: select path with highest var and lowest var (Figure 2) ====
model_paths <- tibble()
if (dataset_name == "nba") {
  select_id <- 21800521
} else {
  select_id <- 16238
}

for (i in 1:length(sample_paths$paths)) {
  single_smp_paths <- sample_paths$paths[[i]]
  single_name <- sample_paths$names[i]
  print(single_name)
  selected_paths <- single_smp_paths %>% 
    filter(obs_id == select_id) %>% 
    mutate(
      method = single_name
    )
  model_paths <- bind_rows(model_paths, selected_paths)
}
true_select_path <- ground_truth %>% filter(obs_id == select_id)

model_paths_plot_data_path = file.path(EXP_RESULTS_PATH, paste0(dataset_name, "_model_paths.rds"))
true_select_path_plot_data_path = file.path(EXP_RESULTS_PATH, paste0(dataset_name, "_true_select_path.rds"))
saveRDS(model_paths, model_paths_plot_data_path)
saveRDS(true_select_path, true_select_path_plot_data_path)

# ====== Visualization: display clusters of paths with different characteristics (Figure 1) ======
# weather
if (dataset_name == "weather") {
  train_df <- read_csv(train_path) %>% rename(obs_id = obs)
  test_df <- read_csv(test_path) %>% 
    mutate(obs_id = obs + max(train_df$obs_id)) %>%
    select(-obs) %>%
    bind_rows(train_df) %>%
    filter(time >= init_time)
  
  test_df
  
  init_test_df <- test_df %>% 
    filter(time == init_time) %>%
    mutate(
      season = ifelse(month %in% c(9, 10, 11), "Spring",
                      ifelse(month %in% c(12, 1, 2), "Summer",
                             ifelse(month %in% c(3, 4, 5), "Fall", "Winter")))
    ) %>% 
    select(obs_id, init_pred = pred_outcome, season)
  path_cluseter_df <- test_df %>% 
    inner_join(init_test_df, by = "obs_id") %>% 
    filter(abs(init_pred - 0.35) < 0.05, season %in% c("Summer", "Winter"))
  winter_paths <- path_cluseter_df %>% 
    filter(season == "Winter") %>% 
    group_by(obs_id) %>% 
    sample_n_groups(size=100) %>% 
    ungroup()
  summer_paths <- path_cluseter_df %>% 
    filter(season == "Summer") %>% 
    group_by(obs_id) %>% 
    sample_n_groups(size=100) %>% 
    ungroup()
  path_cluseter_df <- bind_rows(winter_paths, summer_paths)
  
  path_cluseter_df_plot_data_path = file.path(EXP_RESULTS_PATH, paste0(dataset_name, "_path_cluseter_df.rds"))
  saveRDS(path_cluseter_df, path_cluseter_df_plot_data_path)
}
