############################################################
# Part 2 (Gibbs sampling) 
# Model:
#   y_i | lambda_i ~ Poisson(lambda_i * t_i)
#   lambda_i | beta ~ Gamma(shape = alpha, rate = beta)
#   beta ~ Gamma(shape = gamma, rate = delta)
#
# Full conditionals:
#   lambda_i | beta, D ~ Gamma(shape = alpha + y_i, rate = beta + t_i)
#   beta | lambda, D   ~ Gamma(shape = n*alpha + gamma, rate = delta + sum(lambda_i))
############################################################

############################################################
# Q9-Q11
############################################################

# Data 
y <- c(4, 1, 5, 14, 3, 19, 7, 6)
t <- c(95, 16, 63, 126, 6, 32, 16, 19)
n <- length(y)

# Hyperparameters
alpha <- 1.8
gamma <- 0.01
delta <- 1

# MCMC settings
M <- 35000
B <- 5000
stopifnot(B < M)

set.seed(2025)  

# Storage: columns lambda1..lambda8, beta
draws <- matrix(NA_real_, nrow = M, ncol = n + 1)
colnames(draws) <- c(paste0("lambda", 1:n), "beta")

# Initialization 
beta_curr <- 1
lambda_curr <- rep(0.1, n)

# Gibbs sampler
for (m in 1:M) {
  # 1) Update lambdas given beta
  # lambda_i | beta, D ~ Gamma(alpha + y_i, beta + t_i)
  lambda_curr <- rgamma(n, shape = alpha + y, rate = beta_curr + t)
  
  # 2) Update beta given lambdas
  # beta | lambda, D ~ Gamma(n*alpha + gamma, delta + sum(lambda_i))
  beta_curr <- rgamma(1, shape = n * alpha + gamma, rate = delta + sum(lambda_curr))
  
  draws[m, ] <- c(lambda_curr, beta_curr)
}

# Gibbs acceptance rate: always 100% (no accept/reject step)
acceptance_rate <- 1.0
cat("Gibbs acceptance rate:", acceptance_rate, "(= 100%)\n")

# Discard burn-in
post <- draws[(B + 1):M, , drop = FALSE]

############################################################
# Q12
############################################################
library(coda)

mcmc_chain <- mcmc(post)

# Geweke diagnostic
gw <- geweke.diag(mcmc_chain)
print(gw)

# Heidelberger-Welch stationarity and halfwidth test
hw <- heidel.diag(mcmc_chain)
print(hw)

plot(mcmc_chain[, "lambda1"])

############################################################
# Q13
############################################################

lambda6 <- post[, "lambda6"]
Ey6 <- t[6] * lambda6  # t6 = 32

# Point estimate (posterior mean) and 95% CrI (quantile-based)
Ey6_mean <- mean(Ey6)
Ey6_ci <- quantile(Ey6, probs = c(0.025, 0.975), names = FALSE)

# Round to nearest integer 
Ey6_mean_r <- round(Ey6_mean)
Ey6_ci_r <- round(Ey6_ci)

cat("\nQ13: E(y6) = lambda6 * 32\n")
cat("Posterior mean (rounded):", Ey6_mean_r, "\n")
cat("95% CrI (rounded): [", Ey6_ci_r[1], ", ", Ey6_ci_r[2], "]\n", sep = "")

############################################################
# Q14
############################################################
prob_lambda6_gt <- mean(lambda6 > 0.53)

cat("\nQ14: P(lambda6 > 0.53) estimate:", prob_lambda6_gt, "\n")
