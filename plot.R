source("src/glim/pkgs.R")
source("src/glim/glim_modeling.R")
source("src/glim/consts.R")

options(max.print = 200)
args <- commandArgs(trailingOnly = TRUE)
plot_type <- args[1]

if (is.na(plot_type)) {
  message("Plot paper figures")
  # paper submission w/h
  
  calib_w <- 2.8
  calib_h <- 3.75
} else if (plot_type == "full") {
  message("Plot full size figures")
  
  calib_w <- 5
  calib_h <- 4
} else {
  stop("Unknown plot type!")
}

# ======= Figrue 7: Plot parameter recovery exp ===========
opt_coefs_filepath <- file.path(EXP_RESULTS_PATH, "sim_param_recovery.rds")
opt_coefs <- readRDS(opt_coefs_filepath)
summarized_coef <- opt_coefs %>% 
  group_by(true_beta, true_rho, variable) %>% 
  summarize(
    n = n(),
    mean = mean(estimate),
    lb = mean - sd(estimate),
    ub = mean + sd(estimate),
    .groups = "keep"
  )
summarized_coef

opt_coefs %>% 
  group_by(true_beta, true_rho, variable) %>% 
  summarize(
    mean(estimate),
    sd(estimate),
    .groups = "keep"
  )

for (param_name in c("beta", "rho")) {
  # param_name = "rho"
  param_coef <- summarized_coef %>% filter(variable == param_name)
  if (param_name == "beta") {
    true_p_name <- "true_beta"
    true_other_name <- "true_rho"
    other_name_display <- "\u03C1"  # rho
    param_name_display <- "\u03B2"  # beta
  } else {
    true_p_name <- "true_rho"
    true_other_name <- "true_beta"
    param_name_display <- "\u03C1"  # rho
    other_name_display <- "\u03B2"  # beta
  }
  p <- ggplot(param_coef, aes(x=!!sym(true_other_name), y=mean)) +
    geom_point(size=1.75) +
    geom_errorbar(aes(ymin = lb, ymax = ub), width = 0.02) +
    geom_hline(yintercept = param_coef %>% pull(!!sym(true_p_name)) %>% unique(), alpha=0.3) +
    labs(y = param_name_display, x = other_name_display) +
    scale_y_continuous(breaks=seq(-0.4, 0.4, 0.2), limits=c(-0.45, 0.45)) +
    scale_x_continuous(breaks=seq(-0.4, 0.4, 0.2)) +
    theme(panel.grid.minor.y=element_blank(),
          panel.grid.major.y=element_blank(),
          axis.title.y = element_text(angle = 0, vjust = 0.5))
  p
  ggsave(file.path(PLOT_PATH, paste0("sim_recovery_", param_name,".pdf")), p,  
         device=cairo_pdf, width=2.2, height=2.2)
}

# ======= Figrue 6: plot simulated example paths ========
example_T <- 10
example_y0 <- 0.75
exp_paths <- tibble()

set.seed(42)
for (beta in c(-1, 0, 1)) {
  for (rho in c(-0.5, 0, 0.5)) {
    for (sample_id in 1:100) {
      example_covar <- exp_decay_covar_beta(n = example_T, beta_t = beta, rho = rho, keep_rho_sign = F)
      example_z <- rmvnorm(1, sigma=example_covar)
      example_y <- z2y(z=example_z, y0=example_y0, covar=example_covar)
      single_exp_path <- tibble(
        time = 0:example_T,
        y = c(example_y0, example_y),
        beta = beta,
        rho = rho,
        group = paste(paste0("\u03B2: ", beta, ", \u03C1: ", rho)),
        sample_id = sample_id
      )
      exp_paths <- rbind(exp_paths, single_exp_path)
    }
  }
}

exp_paths <- exp_paths %>% 
  arrange(beta, rho) %>% 
  mutate(group = factor(group, levels = unique(group)))

exp_paths_means <- exp_paths %>% 
  group_by(time, group) %>% 
  summarize(
    mean_pred = mean(y),
    .groups = "keep"
  )

example_param_path_plot <- ggplot(exp_paths , aes(x=time, y=y, group=sample_id)) +
  geom_line(alpha = 0.06) +
  geom_line(data=exp_paths_means, aes(y=mean_pred, group=NULL),
            color="red", linetype = "dotdash", size=0.9)  + 
  facet_wrap(~ group) +
  scale_x_continuous(name="Time", breaks=seq(0, example_T, 2)) +
  scale_y_continuous(name="Probability", labels = scales::percent_format(accuracy = 1))
example_param_path_plot

ggsave(file.path(PLOT_PATH, paste0("example_param_path_plot.pdf")),
       device=cairo_pdf,
       plot=example_param_path_plot, width=5, height=4.5)

ggsave(file.path(PLOT_PATH, paste0("example_param_path_plot.png")),
       plot=example_param_path_plot, width=5, height=4.5)


# ======= plot performance plot =====
for (dataset_name in c("nba", "weather")) {
  if (dataset_name == "nba") {
    source("src/glim/nba_const.R")
    plot_breaks <- seq(0, 48, 4)
    full_plot_breaks <- seq(0, 48, 8)
    X_unit_name <- "Minutes"
  } else if (dataset_name == "weather") {
    source("src/glim/weather_const.R")
    plot_breaks <- seq(0, 7, 1)
    full_plot_breaks <- plot_breaks
    X_unit_name <- "Days"
  }
  
  # Table 1: ==== CI coverage ====
  coverage_summary_data_path = file.path(EXP_RESULTS_PATH, paste0(dataset_name, "_coverage_summary.rds"))
  coverage_summary <- readRDS(coverage_summary_data_path)
  
  if (dataset_name == "nba") {
    coverage_table_checkpoints <- c(25, 36, 47)
  } else {
    coverage_table_checkpoints <- c(1, 3, 6)
  }
  
  printable_coverage_summary <-
    coverage_summary %>% 
    mutate(
      formatted_alpha = paste0(z*100, "%"),
      coverage_err = coverage - z
    ) %>%
    filter(time %in% coverage_table_checkpoints) %>% 
    group_by(time, formatted_alpha) %>% 
    mutate(
      is_good = abs(abs(coverage_err) - min(abs(coverage_err))) < 0.02
    ) %>% 
    ungroup() %>% 
    mutate(
      coverage_ci = ifelse(is_good,
                           sprintf("\textbf{%.2f}", coverage_err),
                           sprintf("%.2f", coverage_err)),
      time = sprintf("%d", time)
    ) %>%
    group_by(method) %>% 
    pivot_wider(id_cols=c("time", "formatted_alpha"), names_from=method, values_from=coverage_ci) %>% 
    arrange(time, formatted_alpha)
  
  table_output <- print(xtable(printable_coverage_summary), include.rownames=FALSE)
  table_output <- gsub("\textbf", "\\textbf", table_output, fixed = TRUE)
  table_output <- gsub("\\{", "{", table_output, fixed = TRUE)
  table_output <- gsub("\\}", "}", table_output, fixed = TRUE)
  cat(table_output)
  
  # ==== outcome model calibration ====
  outcome_model_calib_filepath = file.path(EXP_RESULTS_PATH, paste0(dataset_name, "_outcome_model_calib.rds"))
  outcome_model_calib <- readRDS(outcome_model_calib_filepath)
  ggplot(outcome_model_calib, aes(x=pred_bucket, y=prec_outcome)) + 
    geom_point() +
    geom_abline(intercept = 0, slope = 1) + 
    scale_y_continuous(name="Observed probability", 
                       labels = scales::percent_format(accuracy = 1),
                       lim=c(0, 1)) + 
    scale_x_continuous(name="Predicted probability",
                       labels = scales::percent_format(accuracy = 1),
                       lim=c(0, 1)) 
  ggsave(file.path(PLOT_PATH, paste0(dataset_name, "_outcome_model_calib.pdf")),
         device=cairo_pdf,
         width=3, height=3)
  
  # ===== Q_t deviation mse =====
  excessive_move_data_filepath = file.path(EXP_RESULTS_PATH, paste0(dataset_name, "_excessive_move.rds"))
  excessive_move <- readRDS(excessive_move_data_filepath) %>% 
    mutate(
      method = recode_factor(method, mmfe="MMFE", lr="LR", blstm="BLSTM", mqlstm="MQLSTM", glim="GLIM")
    ) %>%
    arrange(method)
  
  excessive_move_plot <- ggplot(excessive_move, aes(x=method, y=Q_T_mse, color=method)) +
    geom_point(size=2) +
    geom_errorbar(aes(ymin=Q_T_mse - Q_T_se * 1.96, 
                      ymax=Q_T_mse + Q_T_se * 1.96,
                      color=method, width=0.3)) +
    scale_x_discrete(name="Method") + 
    scale_y_continuous(name="Expected Vol. MSE", trans="log10") + 
    theme(
      legend.position = "none",
      axis.title.x=element_blank()
    )
  excessive_move_plot
  ggsave(file.path(PLOT_PATH, paste0(dataset_name, "_excessive_move.pdf")),
         plot=excessive_move_plot,
         device=cairo_pdf,
         width=3, height=2)
  
  # ========== calibration =========
  calib_plot_data_filepath <- file.path(EXP_RESULTS_PATH, paste0(dataset_name, "_calib_mse.rds"))
  init_prob_mse <- readRDS(calib_plot_data_filepath) %>% 
    mutate(
      method = recode_factor(method, mmfe="MMFE", lr="LR", blstm="BLSTM", mqlstm="MQLSTM", glim="GLIM")
    ) %>% 
    filter(time > init_time) %>%
    arrange(method)
  
  calib_plot <- ggplot(init_prob_mse, aes(x=time, y=mse, color=method)) +
    geom_line(size = 0.8, aes(linetype = method)) +
    geom_ribbon(aes(ymin=mse - se * 2, ymax=mse + se * 2, color=NULL, fill=method), alpha=0.2) +
    scale_x_continuous(breaks=plot_breaks, name=X_unit_name) +
    scale_y_continuous(name="Calibration MSE", trans="log10") + 
    scale_linetype_manual(values=c("dashed", "dotdash", "dotted", "solid"))
    # theme(legend.position = "none") +
    # scale_fill_discrete(name = "Method") + 
    # scale_color_discrete(name = "Method") +
    # scale_color_brewer(palette="Set1") +
    # scale_fill_brewer(palette="Set1")
  
  if (dataset_name == "nba") {
    calib_plot <- calib_plot +
      theme(
        legend.position = c(0.32, 0.68),
        legend.title = element_blank(),
        legend.text = element_text(size = 8.5),
        legend.background = element_blank()
      )
  } else {
    calib_plot <- calib_plot + 
      theme(legend.position = "none")
  }
  
  calib_plot
  ggsave(file.path(PLOT_PATH, paste0(dataset_name, "_calib_mse.pdf")),
         plot=calib_plot, device=cairo_pdf, width=calib_w, height=calib_h)
  
  # ====== visualize an example path vs. methods =========
  model_paths_plot_data_path <- file.path(EXP_RESULTS_PATH, paste0(dataset_name, "_model_paths.rds"))
  true_select_path_plot_data_path <- file.path(EXP_RESULTS_PATH, paste0(dataset_name, "_true_select_path.rds"))
  model_paths <- readRDS(model_paths_plot_data_path) %>% 
    mutate(
      method = recode_factor(method, mmfe="MMFE", lr="LR", blstm="BLSTM", mqlstm="MQLSTM", glim="GLIM")
    )
  model_paths_means <- model_paths %>% 
    group_by(time, method) %>% 
    summarize(
      mean_pred = mean(pred_outcome),
      .groups = "keep"
    )
  true_select_path <- readRDS(true_select_path_plot_data_path)
  method_paths_plot <- ggplot(model_paths, aes(x=time, y=pred_outcome, group=sample_id)) +
    geom_line(alpha = 0.06) +
    geom_line(data=true_select_path, aes(y=true_pred_outcome, group=NULL),
              color="#619CFF", size=0.9) +
    geom_line(data=model_paths_means, aes(y=mean_pred, group=NULL),
              color="red", linetype = "dotdash", size=0.9) +
    facet_wrap(. ~ method, nrow=1) +
    scale_x_continuous(name=X_unit_name, breaks=full_plot_breaks) +
    scale_y_continuous(name="Probability", labels = scales::percent_format(accuracy = 1))
  method_paths_plot
  ggsave(file.path(PLOT_PATH, paste0(dataset_name, "_method_paths.pdf")), 
         device=cairo_pdf, width=10, height=1.85)
  
  if (dataset_name == "weather") {
    path_cluseter_df_plot_data_path <- file.path(EXP_RESULTS_PATH, paste0(dataset_name, "_path_cluseter_df.rds"))
    path_cluseter_df <- readRDS(path_cluseter_df_plot_data_path)
    
    weather_season_diff <- ggplot(path_cluseter_df, aes(x=time, y=pred_outcome, group=obs_id)) +
      geom_line(alpha=0.2) +
      facet_wrap(. ~ season) +
      scale_x_continuous(name=X_unit_name, breaks=plot_breaks) +
      scale_y_continuous(name="Probability", labels = scales::percent_format(accuracy = 1))
    weather_season_diff
    ggsave(file.path(PLOT_PATH, paste0("weather_seasonal.pdf")), 
           device=cairo_pdf, width=5, height=2.2)
    ggsave(file.path(PLOT_PATH, paste0("weather_seasonal.png")), 
           width=5, height=2.2)
  }
}

