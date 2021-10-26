functions {
  matrix[] calc_covars(matrix X, vector beta, real rho, int T_) {
    int n_obs = rows(X);
    matrix[T_, T_] covars[n_obs];
    matrix[T_, T_] rho_mat;
    vector[T_] diag_seq;
    vector[T_] sigma;
    vector[n_obs] slopes;
    
    for (i in 1:T_) {
      diag_seq[i] = (i - 1);
    }
    
    for (j in 1:T_) {
      for (i in j:T_) {
        rho_mat[i, j] = pow(rho, abs(i - j));
        rho_mat[j, i] = rho_mat[i, j];
      }
    }
    slopes = X * beta;
    
    for (i in 1:n_obs){
      sigma = sqrt(exp(diag_seq * slopes[i]));
      covars[i] = (sigma * sigma') .* rho_mat;
    }
    return covars;
  }
  
  real partial_sum(matrix[] covars,
                   int start, int end,
                   int T_,
                   vector y0,
                   real[,] y,
                   real[,] phi_inv_y) {
    real sum_log_ll = 0;      
    real tol = 1e-3;
    int i = start;
    for (covar in covars) {
      vector[T_ - 1] z;
      vector[T_ - 1] mu_tilde;
      vector[T_ - 1] sigma_tilde;
  
      real gamma;
      real last_mu_t_1;
      real last_K_t_11;
      int stop_t = 0;
      
      if ((y0[i] >= (1 - tol)) || (y0[i] <= tol)) {
        continue;
      }
      
      // init values
      // covar = covars[i];
      gamma = inv_Phi(y0[i]) * sqrt(sum(covar));
      
      last_mu_t_1 = 0;
      last_K_t_11 = covar[1, 1];
      
      for (t in 1:(T_ - 1)) {
        matrix[t, t] K22 = covar[1:t, 1:t];
        matrix[T_ - t, T_ - t] K11 = covar[(t+1):T_, (t+1):T_];
        matrix[T_ - t, t] K12 = covar[(t+1):T_, 1:t];
        matrix[t, T_ - t] K21 = covar[1:t, (t+1):T_];
        matrix[T_ - t, t] K12_K22_inv_prod;
        matrix[T_ - t, T_ - t] cond_t_covar;
        row_vector[t] a_t;
        row_vector[T_ - t] ones;
        real sigma_bar_t;
        real prev_a1z_sum;
        
        K12_K22_inv_prod = mdivide_right_spd(K12, K22);
        
        cond_t_covar = K11 - K12_K22_inv_prod * K21;
        
        // init values
        for (j in 1:(T_ - t)) ones[j] = 1;
        a_t = ones * K12_K22_inv_prod;
        sigma_bar_t = sqrt(sum(cond_t_covar));
        
        // If extreme value occurs, assume the path stays there
        if ((y[i, t] >= (1 - tol)) || (y[i, t] <= tol)) {
          break;
        }
        stop_t = t;
        
        prev_a1z_sum = (t > 1) ? dot_product(a_t[1:(t-1)] + 1, z[1:(t - 1)]) : 0;
        
        z[t] = (sigma_bar_t * phi_inv_y[i, t] - gamma - prev_a1z_sum) / (a_t[t] + 1);
        
        mu_tilde[t] = (gamma + prev_a1z_sum + (a_t[t] + 1) * last_mu_t_1) / sigma_bar_t;
        sigma_tilde[t] = sqrt(last_K_t_11) * (a_t[t] + 1) / sigma_bar_t;
  
        // likelihood in terms of y
        // sum_log_ll += normal_lpdf(phi_inv_y[i, t] | mu_tilde[t], sigma_tilde[t]);
       
        // update auxiliary statistics for next iteration
        if (t != (T_ - 1)) {
          // last_mu_t_1 = (K12_K22_inv_prod * z[1:t])[1];
          // same the line above, but less multiplication
          last_mu_t_1 = dot_product(K12_K22_inv_prod[1,:], z[1:t]);
          last_K_t_11 = cond_t_covar[1, 1];
        }
      } 
    
      sum_log_ll += normal_lpdf(phi_inv_y[i, 1:stop_t] | mu_tilde[1:stop_t], sigma_tilde[1:stop_t]);
      
      i += 1;
    }
    
    return sum_log_ll;
  }
}

data {
  // number of games
  int n_obs;
  // number of time steps to look into the future
  int T_;
  vector[n_obs] y0;
  // this is for t = 1..T-1
  // since time can range from 0 to T
  // does not need y_T here in data since its likelihood is a constant
  // relative to the parameters.
  // make y array instead of matrix so it's faster when iterating along observations
  real<lower=0, upper=1> y[n_obs, T_ - 1];
  // number of features
  int K;
  // design matrix X for t = 1 .. T
  // should include future_time but not time
  // make X array instead of matrix so it's faster when iterating along observations
  matrix[n_obs, K] X;
  int<lower=1> grainsize;
}


parameters {
  // regression coefficient vector
  vector[K] beta;
  real atanh_rho;
}

transformed parameters {
  real<lower=-1, upper=1> rho;
  rho = tanh(atanh_rho);
}

model {
  matrix[T_, T_] covars[n_obs];
  real phi_inv_y[n_obs, T_ - 1] = inv_Phi(y);
  // int grainsize = 1;
   
  // priors
  beta ~ std_normal();
  atanh_rho ~ std_normal();
  // beta ~ normal(0, 0.05);
  // atanh_rho ~ normal(0, 0.05);
  
  covars = calc_covars(X, beta, rho, T_);
  
  // target += partial_sum(covars, 1, n_obs, T_, y0, y, phi_inv_y);
  target += reduce_sum(partial_sum, covars, grainsize, T_, y0, y, phi_inv_y);
}

generated quantities {
}