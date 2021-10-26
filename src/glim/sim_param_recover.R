source("src/glim/pkgs.R")
source("src/glim/glim_modeling.R")
source("src/glim/consts.R")

options(max.print = 200)

n_obs <- 20
T_ <- 5
n_round_sim <- 2

opt_coefs <- tibble()
true_rhos <- c(-0.4, -0.2, 0, 0.2, 0.4)
true_betas <- c(-0.4, -0.2, 0, 0.2, 0.4)

for (true_rho in true_rhos) {
 for (true_beta in true_betas) {
    for (j in 1:n_round_sim) {
      y0 <- c()
      y <- NULL
      y_T <- c()
      X <- NULL
      for (i in 1:n_obs) {
        this_y0 <- runif(1, min=0.4, max=0.6)
        x <- i %% 2
        
        if (x == 0) {
          covar <- exp_decay_covar_beta(n = T_, beta_t = 0, rho = true_rho, keep_rho_sign = F)
        } else {
          covar <- exp_decay_covar_beta(n = T_, beta_t = true_beta, rho = true_rho, keep_rho_sign = F)
        }
        
        z <- rmvnorm(1, sigma=covar)
        this_y <- z2y(z=z, y0=this_y0, covar=covar)
        
        this_X <- matrix(c(x), nrow = 1, ncol = 1)
        
        y0 <- c(y0, this_y0)
        y <- rbind(y, this_y[1:(T_-1)])
        y_T <- c(y_T, this_y[T_])
        X <- rbind(X, this_X)
      }
      
      X_names <- c("intercept", "beta")
      sim_info_data <- list(
        X_names = X_names,
        X = X,
        K = ncol(this_X),
        n_obs = n_obs,
        T_ = T_,
        y0 = y0,
        y = y,
        y_T = y_T
      )
      
      
      # ======== fit model =========
      glim_mcmc <- fit_glim(
        stan_file_name = "src/glim/glim_sim.stan",
        train_data = sim_info_data,
        method = "mcmc"
      )
      single_opt_coefs <- glim_mcmc$summary() %>%
        filter(!(variable %in% c("lp__", "atanh_rho"))) %>%
        select(variable, estimate=mean)
      
      single_opt_coefs <- single_opt_coefs %>% 
        mutate(
          variable = c("beta", "rho"),
          sim_id = j,
          true_beta = true_beta,
          true_rho = true_rho,
          n_obs = n_obs
        )
      
      opt_coefs <- rbind(opt_coefs, single_opt_coefs)
    }
 }
}

print(opt_coefs %>% 
  group_by(true_beta, true_rho, variable) %>% 
  summarize(
    mean = mean(estimate),
    lb = mean - 2 * sd(estimate)/sqrt(n()),
    ub = mean + 2 * sd(estimate)/sqrt(n())
  ))

opt_coefs_filepath = file.path(EXP_RESULTS_PATH, "sim_param_recovery.rds")
saveRDS(opt_coefs, opt_coefs_filepath)
