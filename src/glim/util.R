# prediction and performance functions are from package ROCR
compute_auc <- function(p, labels) {
  pred <- prediction(p, labels)
  auc <- performance(pred, 'auc')
  auc <- unlist(slot(auc, 'y.values'))
  auc
}

fill_na <- function(df) {
  # from https://stackoverflow.com/questions/7731192/replace-mean-or-mode-for-missing-values-in-r
  Mode <- function (x, na.rm) {
    if (!any(is.na(x))) { return(NULL) }
    xtab <- table(x)
    xmode <- names(which(xtab == max(xtab)))
    if (length(xmode) > 1) xmode <- ">1 mode"
    return(xmode)
  }
  
  df %>% 
    mutate_if(~ any(is.na(.)), ~ if (is.numeric(.)) {replace_na(., mean(., na.rm = TRUE))} else {replace_na(., Mode(., na.rm = TRUE))})
  
}

clip01 <- function(x, tol=1e-2) {
  pmax(pmin(x, 1 - tol), tol)
}

sample_n_groups <- function(grouped_df, size, replace = FALSE) {
  grp_var <- grouped_df %>%
    groups %>%
    unlist %>%
    as.character
  random_grp <- grouped_df %>%
    summarise() %>%
    sample_n(size, replace) %>%
    mutate(unique_id = 1:NROW(.))
  grouped_df %>%
    right_join(random_grp, by=grp_var) %>%
    group_by(!!sym(grp_var))
}


read_and_clean_sample_paths <- function(paths, info_paths_init) {
  if ("m" %in% colnames(paths)) {
    paths <- paths %>% 
      rename(
        sample_id = m
      )
  }
  
  if (min(paths$sample_id) == 0) {
    paths <- paths %>% 
      mutate(
        sample_id = sample_id + 1
      )
  }

  if ("X1" %in% colnames(paths)) {
    paths <- paths %>% 
      select(-X1)
  }
  
  if ("GAME_ID" %in% colnames(paths)) {
    paths <- paths %>%
      rename(
        obs_id = GAME_ID
      )
  } else if ("obs" %in% colnames(paths)) {
    paths <- paths %>%
      rename(
        obs_id = obs
      )
  }

  paths <- paths %>% 
    filter(obs_id %in% unique(info_paths_init$obs_id)) %>%
    filter(time != init_time, time != target_time) 
  
  correct_final_pred <- paths %>% 
    filter(time == target_time - 1) %>% 
    mutate(
      pred_outcome = rbinom(n(), 1, pred_outcome),
      time=target_time
    )

  paths <- paths %>% 
    bind_rows(info_paths_init) %>% 
    bind_rows(correct_final_pred) %>% 
    arrange(obs_id, sample_id, time) 
  
  paths
}
