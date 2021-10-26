# ============ Sim data helper ==========
z2y <- function(z, y0, covar, is_diag=F) {
  # tol <- 1e-8
  T_ <- nrow(covar)
  if (length(dim(z)) != 2) {
    z <- matrix(z, byrow=T, ncol=T_)
  }
  N <- nrow(z)
  
  gamma <- qnorm(y0) * sqrt(sum(covar))
  y <- matrix(NA, nrow=N, ncol=T_)
  calc_y_until <- min((T_-1), length(z))
  
  if (is_diag) {
    # independent
    # TODO: improve perf here
    sigma2 <- diag(covar)
    rev_cumsum_sigma2 <- rev(cumsum(rev(sigma2)))
    for (i in 1:N) {
      cumsum_z <- cumsum(z)
      for (t in 1:calc_y_until) {
        sigma_bar_t <- sqrt(rev_cumsum_sigma2[t+1])
        y[i, t] <- pnorm((gamma + cumsum_z[t]) / sigma_bar_t)
      }
    }
  } else {
    # correlated
    a <- matrix(NA, nrow=T_, ncol=T_)
    sigma_bar <- rep(NA, nrow=T_)
    for (t in 1:calc_y_until) {
      res <- calc_a_t(t, covar)
      a[t, 1:t] <- res$a_t
      sigma_bar[t] <- res$sigma_bar_t
    }
    for (i in 1:N) {
      if (i %% 1e6 == 0) {
        message(sprintf("Processed %d paths", i))
      }
      for (t in 1:calc_y_until) {
        given_z <- z[i, 1:t]
        mu_bar_t <- sum(a[t, 1:t] * given_z)
        y[i, t] <- pnorm((gamma + sum(given_z) + mu_bar_t) / sigma_bar[t])
        # if (y[i, t] < tol) {
        #   y[i, (t+1):calc_y_until] <- 0
        #   break
        # } else if (y[i, t] > (1 - tol)) {
        #   y[i, (t+1):calc_y_until] <- 1
        #   break
        # }
      }
    }
  }
  
  y[,T_] <- as.numeric((gamma + rowSums(z)) >= 0)
  if (N == 1) {
    # convert to vector if N == 1
    y <- c(y)
  } 
  y
}

y2z <- function(y, y0, covar) {
  T_ <- nrow(covar)
  gamma <- qnorm(y0) * sqrt(sum(covar))
  z <- rep(NA, T_ - 1)
  
  for (t in 1:(T_-1)) {
    res <- calc_a_t(t, covar)
    a_t <- res$a_t
    sigma_bar_t <- res$sigma_bar_t
    a_t_plus_1 <- a_t + 1
    
    if (t > 1) {
      z[t] <- (sigma_bar_t * qnorm(y[t]) - gamma - sum(a_t_plus_1[1:(t-1)] * z[1:(t-1)])) / a_t_plus_1[t]
    } else {
      z[t] <- (sigma_bar_t * qnorm(y[t]) - gamma) / a_t_plus_1[t]
    }  
  }   
  
  z
}

ry <- function(y0, covar, n=1) {
  z <- rmvnorm(n, sigma=covar)
  T_ <- length(z)
  y <- z2y(z, y0, covar)
  # z_check <- y2z(y, y0, covar)
  # stopifnot(max(abs(z[1:(T_ - 1)] - z_check)) < 1e-8)
  y
}

calc_a_t <- function(t, covar) {
  T_ <- nrow(covar)
  K22 <- covar[1:t, 1:t,drop=F]
  K11 <- covar[(t+1):T_, (t+1):T_,drop=F]
  K12 <- covar[(t+1):T_, 1:t,drop=F]
  K21 <- covar[1:t, (t+1):T_,drop=F]
  
  K22_inv <- chol2inv(chol(K22))

  K12_K22_inv_prod <- K12 %*% K22_inv
  a_t <- colSums(K12_K22_inv_prod)
  covar_t <- K11 - K12_K22_inv_prod %*% K21

  sigma_bar_t <- sqrt(sum(covar_t))

  list(
    a_t = a_t,
    sigma_bar_t = sigma_bar_t
  )
}

# ============= Covariance constructor =============
calc_covar_const <- function(n, diag_val, rho, keep_rho_sign=F) {
  sigma2 <- diag_val
  if (length(diag_val) != n) {
    sigma2 <- rep(diag_val, n)
  }
  # sigma2[1] <- 1
  sigma <- sqrt(sigma2)
  covar <- diag(n)
  covar <- (rho ^ abs(row(covar) - col(covar)))
  covar <- sigma[col(covar)] * sigma[row(covar)] * covar
  
  if (keep_rho_sign) {
    covar[abs(row(covar) - col(covar)) != 0] <- sign(rho) * abs(covar[abs(row(covar) - col(covar)) != 0])
  }
  covar
}

calc_covar_inv_logit_beta <- function(X, params, T_) {
  K <- ncol(X)
  N <- nrow(X)
  
  if (is.null(dim(params))) {
    # if params is a vector and replicate it N times
    # this is used when MLE/MAP estimator from stan is used
    params <- t(replicate(N, params))
    message("Duplicating parameters N times")
  }
  
  covars <- array(NA, c(N, T_, T_))
  
  beta2s <- params[, 1:K]
  b_over_a_ratio <- params[, K+1]
  rhos <- params[, K+2]
  p <- params[, K+3][1]
  sigma2s_intercepts <- params[, (K+3+1):(K+3+T_)]
  
  a <- log(1 + exp(rowSums(as.matrix(beta2s) * X)))
  b <- -a * b_over_a_ratio
  
  # a <- pmax(a, 0)
  # b <- pmin(b, 0)
  # slopes <- rowSums(as.matrix(betas) * X)
  
  vals <- a %*% t((0:(T_-1))^2) + b %*% t((0:(T_-1)))
  # double last value
  mid_offset <- 1.5
  
  vals <- vals - mid_offset
  vals[,1] <- vals[,1] + mid_offset - 0.7
  vals[,T_-1] <- vals[,T_-1] + mid_offset - 0.3
  vals[,T_] <- vals[,T_] + mid_offset + 2
  
  vals <- pmin(pmax(vals, -6), 6)
  # vals <- pmin(pmax(vals, -10), 10)
  
  # vals <- slopes %*% t(0:(T_-1))
  # sigma2s <- log(1 + exp(vals)) + sigma2s_intercepts
  sigma2s <- exp(vals)
  
  for (i in 1:N) {
    sigma2 <- sigma2s[i,]
    rho <- rhos[i]
    covars[i,,] <- calc_covar_const(T_, sigma2, rho=rho)
  }
  
  covars
}

calc_covar_inv_logit_beta_weather <- function(X, params, T_) {
  K <- ncol(X)
  N <- nrow(X)
  
  if (is.null(dim(params))) {
    # if params is a vector and replicate it N times
    # this is used when MLE/MAP estimator from stan is used
    params <- t(replicate(N, params))
    message("Duplicating parameters N times")
  }
  
  covars <- array(NA, c(N, T_, T_))
  
  betas <- params[, 1:K]
  rhos <- params[, K+1]
  p <- params[,K+2][1]
  sigma2s_intercepts <- params[, (K+2+1):(K+2+T_)]
  
  slopes <- 5/(T_-1) * inv.logit(rowSums(as.matrix(betas) * X))
  # slopes <- rowSums(as.matrix(betas) * X)
  vals <- slopes %*% t((0:(T_-1))^p)
  # vals <- pmin(pmax(vals, -30), 30)
  
  # vals <- slopes %*% t(0:(T_-1))
  # sigma2s <- log(1 + exp(vals)) + sigma2s_intercepts
  sigma2s <- exp(vals + sigma2s_intercepts)
  
  for (i in 1:N) {
    sigma2 <- sigma2s[i,]
    rho <- rhos[i]
    covars[i,,] <- calc_covar_const(T_, sigma2, rho=rho)
  }
  
  covars
}

calc_covar_inv_logit_beta_nba <- function(X, params, T_) {
  K <- ncol(X)
  N <- nrow(X)
  
  if (is.null(dim(params))) {
    # if params is a vector and replicate it N times
    # this is used when MLE/MAP estimator from stan is used
    params <- t(replicate(N, params))
    message("Duplicating parameters N times")
  }
  
  covars <- array(NA, c(N, T_, T_))
  
  betas <- params[, 1:K]
  rhos <- params[, K+1]
  sigma2s <- params[, (K+1+1):(K+1+T_)]

  slopes <- 5/(T_-1) * inv.logit(rowSums(as.matrix(betas) * X))
  vals <- slopes %*% t(0:(T_-1))
  sigma2s <- exp(vals)
  
  for (i in 1:N) {
    sigma2 <- sigma2s[i,]
    rho <- rhos[i]
    covars[i,,] <- calc_covar_const(T_, sigma2, rho=rho)
  }
  
  covars
}

exp_decay_covar_beta <- function(n, beta_t, rho, keep_rho_sign=F) {
  sigma2 <- exp((0:(n-1)) * beta_t)
  calc_covar_const(n, sigma2, rho, keep_rho_sign)
}

calc_covars_linear <- function(X, params, T_, rho=NULL, tanh_rho=F, model_rho=F) {
  K <- ncol(X)
  N <- nrow(X)
  
  if (is.null(dim(params))) {
    # if params is a vector and replicate it N times
    # this is used when MLE/MAP estimator from stan is used
    params <- t(replicate(N, params))
    message("Duplicating parameters N times")
  }
  
  covars <- array(NA, c(N, T_, T_))
  for (i in 1:N) {
    beta <- params[i, 1:K]
    if (is.null(rho)) {
      if (model_rho) {
        atanh_rho_beta <- params[i, (K + 1):(K + K)] 
        rho <- tanh(sum(atanh_rho_beta * X[i,]))
      } else {
        rho <- params[i, K + 1]
        if (tanh_rho) {
          rho <- tanh(rho)
        }
      }
    }
    
    sigma2 <- exp(sum(beta * X[i,]) * (0:(T_-1)))
    covars[i,,] <- calc_covar_const(T_, sigma2, rho=rho)
  }
  
  covars
}

# ========= fit glim stan model ==========
fit_glim <- function(stan_file_name, train_data, method) {
  #' fit GLIM model
  #' @param stan_file_name stan file name
  #' @param train_data training data as a list
  #' @param method "map" or "mcmc"
  #'
  #' @return CmdStanMLE or CmdStanMCMC object depending on the method parameter
  
  glim_model <- cmdstan_model(stan_file_name,
                              cpp_options = list(stan_threads = TRUE))
  train_data[["X_names"]] <- NULL
  train_data[["grainsize"]] <- 1
  
  if (method == "map") {
    mod <- glim_model$optimize(
      train_data, 
      # init = 0, 
      threads = 48
      # algorithm = "newton"
    )
    print(mod$summary(), n=nrow(mod$summary()))
  } else if (method == "mcmc") {
    mod <- glim_model$sample(
      train_data,
      chains = 4,
      parallel_chains = 4,
      threads_per_chain = 12,
      refresh = 20,
      iter_warmup = 1000,
      iter_sampling = 500,
      # init = 0
    )
    print(mod$summary())
  } else {
    warning("Unsupported fitting method!")
    mod <- NULL
  }
  
  mod
}


