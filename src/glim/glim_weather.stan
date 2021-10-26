functions {
  matrix[] calc_covars(matrix X, vector sigma2, vector beta2, real b_over_a_ratio, 
                       real rho, real p, int T_) {
    int n_obs = rows(X);
    matrix[T_, T_] covars[n_obs];
    matrix[T_, T_] rho_mat;
    vector[T_] sigma;
    vector[T_] diag_seq;
    vector[T_] diag_seq2;
    vector[n_obs] a;
    vector[n_obs] b;
    
    
    for (i in 1:T_) {
      // diag_seq[i] = (i - 1);
      diag_seq[i] = (i - 1);
      diag_seq2[i] = (i - 1)^2;
    }
    
    for (j in 1:T_) {
      for (i in j:T_) {
        rho_mat[i, j] = pow(rho, abs(i - j));
        rho_mat[j, i] = rho_mat[i, j];
      }
    }
    
    a = log(1 + exp(X * beta2));
    b = -a * b_over_a_ratio;

    for (i in 1:n_obs){
      real val_lim = 6;
      real mid_offset = 1.5;
      vector[T_] vals;
      
      vals = a[i] * diag_seq2 + b[i] * diag_seq + sigma2;
      // // double the value for the last step
      vals = vals - mid_offset;
      vals[1] = vals[1] + mid_offset - 0.7;
      vals[T_-1] = vals[T_-1] + mid_offset - 0.3;
      vals[T_] = vals[T_] + mid_offset + 2;

      // cap diagonal values from exploding
      for (j in 1:T_) {
        // limits for exp
        if (vals[j] > val_lim) {
          vals[j] = val_lim;
        }
        if (vals[j] < -val_lim) {
          vals[j] = -val_lim;
        }
      }
      
      sigma = sqrt(exp(vals));
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
    real tol = 1e-8;
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
  // untransformed diagonal values starting from the second one
  // vector[T_ - 1] pre_tsfm_sigma2;
  vector<lower=0>[T_] pre_tsfm_sigma2;
  real atanh_rho;
  vector[K] beta2;
  real<lower=4, upper=5> b_over_a_ratio;
}

transformed parameters {
  real<lower=-1, upper=1> rho = 0;
  // real<lower=-1, upper=1> rho = tanh(atanh_rho);
  real<lower=1, upper=2> p=1;
  // vector<lower=0>[T_] sigma2;
  vector[T_] sigma2;
  
  sigma2[1] = 0;
  for (i in 2:T_) {
    sigma2[i] = 0;
  }
}

model {
  matrix[T_, T_] covars[n_obs];
  real phi_inv_y[n_obs, T_ - 1] = inv_Phi(y);
  // int grainsize = 1;
   
  // priors
  pre_tsfm_sigma2 ~ std_normal();
  atanh_rho ~ std_normal();
  beta2 ~ std_normal();
  
  covars = calc_covars(X, sigma2, beta2, b_over_a_ratio, rho, p, T_);
  
  // target += partial_sum(covars, 1, n_obs, T_, y0, y, phi_inv_y);
  target += reduce_sum(partial_sum, covars, grainsize, T_, y0, y, phi_inv_y);
}

generated quantities {
}
