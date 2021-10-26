source("src/glim/pkgs.R")
source("src/glim/util.R")

# ============================== Evaluation ==============================
calc_init_prob_mse <- function(paths, ground_truth, 
                                  init_time, target_time, method=NULL) {
  
  mean_preds <- paths %>% 
    group_by(obs_id, time) %>% 
    summarize(
      mean_pred = mean(pred_outcome),
      .groups = "keep"
    )
  
  mean_preds <- mean_preds %>% 
    inner_join(ground_truth, by=c("obs_id", "time")) %>% 
    mutate(
      sq_err = (mean_pred - init_prob)^2
    ) %>% 
    group_by(time) %>% 
    summarize(
      mse = mean(sq_err),
      se = sd(sq_err)/sqrt(n()),
      .groups = "keep"
    )

  if (!is.null(method)) {
    mean_preds <- mean_preds %>% 
      mutate(method=method)
  }
  
  mean_preds
}


calc_realized_vol <- function(paths, init_time, ext_window_size, method=NULL) {
  smp_ret <- paths %>% 
    group_by(obs_id, sample_id) %>% 
    arrange(obs_id, sample_id, time) %>% 
    mutate(
      ret2 = (pred_outcome - lag(pred_outcome))^2
    ) %>% 
    ungroup()
  
  # ext_window_sizes <- c(6, 12, 18, 24)
  realized_vol <- smp_ret %>% 
         filter(time > init_time, time <= init_time + ext_window_size) %>% 
         group_by(obs_id, sample_id) %>% 
         summarize(
           single_vol_hat = sum(ret2),
           .groups = "keep"
         ) %>%
         group_by(obs_id) %>% 
         summarize(
           vol_hat = mean(single_vol_hat),
           ext_window_size = ext_window_size,
           .groups = "keep"
         )
  
  realized_vol
}

calc_realized_vol_mse <- function(paths, ground_truth, 
                         init_time, target_time, vol_window_size, method=NULL) {
  smp_ret <- paths %>% 
    group_by(obs_id, sample_id) %>% 
    arrange(obs_id, sample_id, time) %>% 
    mutate(
      ret2 = (pred_outcome - lag(pred_outcome))^2
    ) %>% 
    ungroup()
  
  true_ret <- ground_truth %>% 
    group_by(obs_id) %>% 
    arrange(obs_id, time) %>% 
    mutate(
      ret2 = (true_pred_outcome - lag(true_pred_outcome))^2,
    ) %>% 
    ungroup()
  
  vol_window_starts <- init_time:(target_time - vol_window_size)
  smp_vol <- map_dfr(vol_window_starts, ~smp_ret %>% 
                       filter(time > .x, time <= .x + vol_window_size) %>% 
                       group_by(obs_id, sample_id) %>% 
                       summarize(
                         single_vol_hat = sum(ret2),
                         .groups = "keep"
                       ) %>% 
                       group_by(obs_id) %>% 
                       summarize(
                         vol_hat = mean(single_vol_hat),
                         vol_window_start = .x,
                         vol_window_size = vol_window_size,
                         .groups = "keep"
                       ))
  
  true_vol <- map_dfr(vol_window_starts, ~true_ret %>% 
                        filter(time > .x, time <= .x + vol_window_size) %>% 
                        group_by(obs_id) %>% 
                        summarize(
                          real_vol = sum(ret2),
                          vol_window_start = .x,
                          vol_window_size = vol_window_size,
                          .groups = "keep"
                        ))
  
  vol_mse <- smp_vol %>%
    inner_join(true_vol, by=c("obs_id", "vol_window_start")) %>% 
    mutate(
      sq_err = (vol_hat - real_vol)^2
    ) %>% 
    group_by(vol_window_start) %>% 
    summarize(
      mse = mean(sq_err),
      se = sd(sq_err)/sqrt(n()),
      .groups = "keep"
    ) 
  
  
  if (!is.null(method)) {
    vol_mse <- vol_mse %>% 
      mutate(method=method)
  }
  
  vol_mse
}

calc_vol_mse <- function(paths, ground_truth, 
                         init_time, target_time, method=NULL) {
  pred_outcome <- paths %>%
    filter(time == target_time) %>% 
    select(obs_id, pred_outcome, sample_id)
  
  final_outcome <- ground_truth %>% 
    filter(time == target_time) %>% 
    select(obs_id, pred_outcome = true_pred_outcome)
  
  conditional_paths <- pred_outcome %>%
    inner_join(final_outcome, by=c("obs_id", "pred_outcome")) %>%
    select(obs_id, sample_id) %>% 
    inner_join(paths, by=c("obs_id", "sample_id"))
  
  vol_mse <- map_dfr((init_time + 1):(target_time - 1), function(t) {
    conditional_vol <- conditional_paths %>% 
      filter(time <= t) %>% 
      group_by(obs_id, sample_id) %>% 
      summarize(
        single_vol_hat = sd(pred_outcome),
        .groups = "keep"
      ) %>% 
      group_by(obs_id) %>% 
      summarize(
        vol_hat = mean(single_vol_hat),
        .groups = "keep"
      )
    
    realized_vol <- ground_truth %>% 
      filter(time <= t) %>% 
      group_by(obs_id) %>% 
      summarize(
        realized_vol = sd(true_pred_outcome),
        .groups = "keep"
      )
    
    conditional_vol %>% 
      inner_join(realized_vol, by="obs_id") %>% 
      mutate(
        sq_err = (vol_hat - realized_vol)^2
      ) %>% 
      summarize(
        mean_vol_hat = mean(vol_hat),
        mean_realized_vol = mean(realized_vol),
        mse = mean(sq_err),
        se = sd(sq_err)/sqrt(n())
      ) %>% 
      mutate(
        time = t
      )
  })
  
  if (!is.null(method)) {
    vol_mse <- vol_mse %>% 
      mutate(method=method)
  }
  

  vol_mse
}

calc_vol_mse_with_return <- function(paths, ground_truth, 
                         init_time, target_time, method=NULL) {
  # might not be correct
  vol_hats <- paths %>% 
    left_join(init_prob, by=c("obs_id")) %>% 
    group_by(obs_id, time) %>% 
    mutate(
      change = pred_outcome - init_prob
    ) %>% 
    summarize(
      vol_hat = sd(change),
      .groups = "keep"
    )
  
  
  vol_mse <- ground_truth %>% 
    inner_join(vol_hats, by=c("obs_id", "time")) %>%
    filter(time > init_time, time < target_time) %>%  
    group_by(time) %>% 
    mutate(
      sq_err = (vol_hat - realized_vol)^2
    ) %>% 
    summarize(
      mse = mean(sq_err),
      se = sd(sq_err)/sqrt(n())
    )
  
  if (!is.null(method)) {
    vol_mse <- vol_mse %>% 
      mutate(method=method)
  }
  vol_mse
}

calc_right_sink_mse <- function(paths, ground_truth, 
                                init_time, target_time, method=NULL) {
  pred_outcome <- paths %>%
    filter(time == target_time) %>% 
    select(obs_id, pred_outcome, sample_id)
  
  final_outcome <- ground_truth %>% 
    filter(time == target_time) %>% 
    select(obs_id, pred_outcome = true_pred_outcome)
  
  right_sink_info_paths <- pred_outcome %>%
    inner_join(final_outcome, by=c("obs_id", "pred_outcome")) %>%
    select(obs_id, sample_id) %>% 
    inner_join(paths, by=c("obs_id", "sample_id"))
  
  right_sink_info_paths <- right_sink_info_paths %>% 
    group_by(obs_id, time) %>% 
    summarize(
      mean_pred_outcome = mean(pred_outcome),
      .groups = "keep"
    )
  
  right_sink_info_paths <- right_sink_info_paths %>% 
    inner_join(ground_truth, by=c("obs_id", "time")) %>% 
    filter(time > init_time, time < target_time) %>% 
    mutate(
      sq_err = (mean_pred_outcome - true_pred_outcome)^2
    ) %>% 
    group_by(time) %>% 
    summarize(
      mse = mean(sq_err),
      se = sd(sq_err)/sqrt(n()),
      .groups = "keep"
    )
  
  if (!is.null(method)) {
    right_sink_info_paths <- right_sink_info_paths %>% 
      mutate(method=method)
  }
  
  right_sink_info_paths
}

calc_ci_coverage <- function(paths, ground_truth, 
                             init_time, target_time,
                             method=NULL, ci_level=0.95) {
  ci_lvl_lb <- (1 - ci_level) / 2
  ci_lvl_ub <- 1 - ci_lvl_lb
  
  paths_ci <- paths %>% 
    group_by(obs_id, time) %>% 
    summarize(
      ci_lb = quantile(pred_outcome, probs=c(ci_lvl_lb)), 
      ci_ub = quantile(pred_outcome, probs=c(ci_lvl_ub)),
      .groups = "keep" 
    )
  
  ci_coverage <- ground_truth %>% 
    left_join(paths_ci, by=c("obs_id", "time")) %>% 
    filter(time > init_time, time < target_time) %>% 
    mutate(
      is_ci_covered = (true_pred_outcome >= ci_lb) & (true_pred_outcome <= ci_ub)
    ) %>% 
    group_by(time) %>% 
    summarize(
      ci_coverage = mean(is_ci_covered),
      ci_level = ci_level,
      .groups = "keep"
    )
  
  if (!is.null(method)) {
    ci_coverage <- ci_coverage %>% 
      mutate(method=method)
  }
  
  ci_coverage
}

calc_conf_pred_mse <- function(paths, ground_truth, method=NULL) {
  conf_pred <- paths %>% 
    inner_join(ground_truth, by=c("obs_id", "time")) %>% 
    mutate(
      conf_pred = pmax(pred_outcome, 1 - pred_outcome),
      conf_true_pred = pmax(true_pred_outcome, 1 - true_pred_outcome),
      conf_sq_err = (conf_pred - conf_true_pred) ^ 2
    ) %>% 
    group_by(time) %>% 
    summarize(
      mse = mean(conf_sq_err),
      se = sd(conf_sq_err)/sqrt(n())
    )
  
  if (!is.null(method)) {
    conf_pred <- conf_pred %>% 
      mutate(method=method)
  }
  conf_pred
}



