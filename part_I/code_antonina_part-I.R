# =========================================================
# Q1 - Logistic mixed model with time + random intercept
# JAGS implementation
# =========================================================

library(rjags)
library(coda)
library(here)

# -----------------------------
# 1) Read & prepare data
# -----------------------------
path <- here::here()
dat <- readRDS("./part_I/data/GSPS.RData")

# Ensure binary
dat$working <- as.integer(dat$working)

# Create numeric id index 
dat$id_int <- as.integer(factor(dat$id))
N_id <- length(unique(dat$id_int))

# Time covariate (centered year)
# Center at first year (1984) 
dat$t <- dat$year - min(dat$year)

# Data lists for JAGS
jags_data <- list(
  y = dat$working,
  t = dat$t,
  id = dat$id_int,
  N = nrow(dat),
  N_id = N_id
)

# -----------------------------
# 2) JAGS model
# -----------------------------
model_string <- "
model {
  for (n in 1:N) {
    y[n] ~ dbern(p[n])
    logit(p[n]) <- alpha + beta_t * t[n] + b[id[n]]
  }

  # Random intercepts
  for (i in 1:N_id) {
    b[i] ~ dnorm(0, tau_b)
  }

   # Priors 
  alpha  ~ dnorm(0, 0.01)
  beta_t ~ dnorm(0, 0.01)

  sigma_b ~ dunif(0, 10)
  tau_b <- 1 / (sigma_b * sigma_b)

  
  # Probability of employment for an average person (b=0) at t=0
  p0 <- ilogit(alpha)

  # Odds ratio per 1 year increase (since t is in years)
  OR_year <- exp(beta_t)

  # Approximate ICC for logistic random intercept model
  ICC <- pow(sigma_b, 2) / (pow(sigma_b, 2) + 3.289868)
}
"

# Write model to temp file for JAGS
model_file <- tempfile(fileext = ".txt")
writeLines(model_string, con = model_file)

# -----------------------------
# 3) Initial values
# -----------------------------
inits_fun <- function() {
  list(
    alpha = rnorm(1, 0, 1),
    beta_t = rnorm(1, 0, 0.5),
    sigma_b = abs(rnorm(1, 0.5, 0.2)),
    b = rnorm(N_id, 0, 0.5)
  )
}

# -----------------------------
# 4) Fit with MCMC
# -----------------------------
params <- c("alpha", "beta_t", "sigma_b", "p0", "OR_year", "ICC")

set.seed(241225)

m <- jags.model(
  file = model_file,
  data = jags_data,
  inits = inits_fun,
  n.chains = 4,
  n.adapt = 1500
)

# Burn-in
update(m, n.iter = 3000)

# Sample
samps <- coda.samples(
  model = m,
  variable.names = params,
  n.iter = 10000,
  thin = 5
)


# -----------------------------
# 5) Convergence diagnostics
# -----------------------------
# Trace plots
plot(samps)

# R-hat (Gelman-Rubin)
gelman.diag(samps, multivariate = FALSE)

# Effective sample size
effectiveSize(samps)

# Autocorrelation
acf(as.numeric(samps[[1]][, "beta_t"]), main = "ACF: beta_t (chain 1)")
acf(as.numeric(samps[[1]][, "alpha"]), main = "ACF: alpha (chain 1)")

# -----------------------------
# 6) Posterior summaries
# -----------------------------
sum_stats <- summary(samps)
print(sum_stats)

# 95% credible intervals
HPDinterval(samps)

# =========================================================
# Q2 - Lasso and ridge models
# JAGS implementation
# =========================================================

# -----------------------------
# 1) Data preparation
# -----------------------------

dat$age      <- scale(dat$age)
dat$educ     <- scale(dat$educ)
dat$hhninc   <- scale(dat$hhninc)
dat$hsat     <- scale(dat$hsat)
dat$handper  <- scale(dat$handper)
dat$docvis   <- scale(dat$docvis)
dat$hospvis  <- scale(dat$hospvis)

covars <- c(
  "t",       # time effect included
  "female",
  "age",
  "hsat",
  "handdum",
  "handper",
  "hhninc",
  "hhkids",
  "educ",
  "married",
  "public",
  "docvis",
  "hospvis"
)


# -----------------------------
# 2) Build design matrix X 
# -----------------------------

form <- as.formula(paste("~", paste(covars, collapse = " + "), "- 1"))
X <- model.matrix(form, data = dat)

# Dimensions
N <- nrow(X)
p <- ncol(X)
colnames(X)  

# Response
y <- dat$working
id <- dat$id_int

# Data list for JAGS
jags_data <- list(
  y = y,
  X = X,
  id = id,
  N = N,
  p = p,
  N_id = N_id
)

# -----------------------------
# 3) JAGS model strings
# -----------------------------

# ---- Ridge: beta_k ~ N(0, tau^2) with common tau
model_ridge <- "
model {
  for (n in 1:N) {
    y[n] ~ dbern(p_i[n])
    logit(p_i[n]) <- beta0 + inprod(X[n,], beta[]) + b[id[n]]
  }

  # Random intercepts
  for (i in 1:N_id) {
    b[i] ~ dnorm(0, tau_b)
  }

  # Ridge prior (Gaussian shrinkage)
  for (k in 1:p) {
    beta[k] ~ dnorm(0, tau_beta)
  }

  # Priors
  beta0 ~ dnorm(0, 0.01)

  tau ~ dunif(0, 10)
  tau_beta <- 1 / (tau * tau)

  sigma_b ~ dunif(0, 10)
  tau_b <- 1 / (sigma_b * sigma_b)
}
"

# ---- Lasso: Laplace via Normal-Exponential mixture
# beta_k | psi_k ~ N(0, psi_k),  psi_k ~ Exp(lambda^2/2)
model_lasso <- "
model {
  for (n in 1:N) {
    y[n] ~ dbern(p_i[n])
    logit(p_i[n]) <- beta0 + inprod(X[n,], beta[]) + b[id[n]]
  }

  # Random intercepts
  for (i in 1:N_id) {
    b[i] ~ dnorm(0, tau_b)
  }

  # Lasso prior via Normal-Exponential mixture
  for (k in 1:p) {
    beta[k] ~ dnorm(0, tau_beta[k])
    tau_beta[k] <- 1 / psi[k]
    psi[k] ~ dexp(lambda2 / 2)
  }

  # Priors
  beta0 ~ dnorm(0, 0.01)

  lambda ~ dgamma(1, 1)
  lambda2 <- lambda * lambda

  sigma_b ~ dunif(0, 10)
  tau_b <- 1 / (sigma_b * sigma_b)
}
"

# -----------------------------
# 4) Initial values
# -----------------------------
inits_ridge <- function() {
  list(
    beta0 = rnorm(1, 0, 1),
    beta  = rnorm(p, 0, 0.2),
    tau = runif(1, 0.5, 2),
    sigma_b = runif(1, 0.5, 2),
    b = rnorm(N_id, 0, 0.3)
  )
}

inits_lasso <- function() {
  list(
    beta0 = rnorm(1, 0, 1),
    beta  = rnorm(p, 0, 0.2),
    psi   = rexp(p, rate = 1),      # >0
    lambda = rgamma(1, 1, 1),
    sigma_b = runif(1, 0.5, 2),
    b = rnorm(N_id, 0, 0.3)
  )
}

# -----------------------------
# 5) MCMC settings
# -----------------------------
set.seed(241225)

n_chains <- 3
n_adapt  <- 1500
n_burn   <- 4000
n_iter   <- 12000
thin     <- 5


params_ridge <- c("beta0", "beta", "tau", "sigma_b")
params_lasso <- c("beta0", "beta", "lambda", "sigma_b")

# Helper to write model to a temp file
write_model <- function(txt) {
  f <- tempfile(fileext = ".txt")
  writeLines(txt, con = f)
  f
}

# -----------------------------
# 6) Fit RIDGE model
# -----------------------------
cat("\n--- Fitting RIDGE model ---\n")

ridge_file <- write_model(model_ridge)

m_ridge <- jags.model(
  file = ridge_file,
  data = jags_data,
  inits = inits_ridge,
  n.chains = n_chains,
  n.adapt = n_adapt
)

update(m_ridge, n.iter = n_burn)

samps_ridge <- coda.samples(
  model = m_ridge,
  variable.names = params_ridge,
  n.iter = n_iter,
  thin = thin
)

# Convergence checks (RIDGE)
cat("\nRIDGE: Gelman-Rubin R-hat\n")
print(gelman.diag(samps_ridge, multivariate = FALSE))

cat("\nRIDGE: Effective sample size\n")
print(effectiveSize(samps_ridge))

# Summaries (RIDGE)
sum_ridge <- summary(samps_ridge)
print(sum_ridge)

# -----------------------------
# 7) Fit LASSO model
# -----------------------------
cat("\n--- Fitting LASSO model ---\n")

lasso_file <- write_model(model_lasso)

m_lasso <- jags.model(
  file = lasso_file,
  data = jags_data,
  inits = inits_lasso,
  n.chains = n_chains,
  n.adapt = n_adapt
)

update(m_lasso, n.iter = n_burn)

samps_lasso <- coda.samples(
  model = m_lasso,
  variable.names = params_lasso,
  n.iter = n_iter,
  thin = thin
)

# Convergence checks (LASSO)
cat("\nLASSO: Gelman-Rubin R-hat\n")
print(gelman.diag(samps_lasso, multivariate = FALSE))

cat("\nLASSO: Effective sample size\n")
print(effectiveSize(samps_lasso))

# Summaries (LASSO)
sum_lasso <- summary(samps_lasso)
print(sum_lasso)

# -----------------------------
# 8) Extract coefficient summaries + compare
# -----------------------------
# Helper: extract posterior draws matrix from mcmc.list
as_draws_matrix <- function(mcmc_list) as.matrix(mcmc_list)

draws_ridge <- as_draws_matrix(samps_ridge)
draws_lasso <- as_draws_matrix(samps_lasso)

# Identify beta columns
beta_names <- paste0("beta[", 1:p, "]")

# Ridge beta summaries
ridge_beta_draws <- draws_ridge[, beta_names, drop = FALSE]
ridge_beta_median <- apply(ridge_beta_draws, 2, median)
ridge_beta_ci <- t(apply(ridge_beta_draws, 2, quantile, probs = c(0.025, 0.975)))

# Lasso beta summaries
lasso_beta_draws <- draws_lasso[, beta_names, drop = FALSE]
lasso_beta_median <- apply(lasso_beta_draws, 2, median)
lasso_beta_ci <- t(apply(lasso_beta_draws, 2, quantile, probs = c(0.025, 0.975)))

# Practical "selection" metric: P(|beta| > c)

c_thresh <- 0.10
ridge_prob_nonzero <- colMeans(abs(ridge_beta_draws) > c_thresh)
lasso_prob_nonzero <- colMeans(abs(lasso_beta_draws) > c_thresh)

# Build comparison table
comp <- data.frame(
  term = colnames(X),
  ridge_median = ridge_beta_median,
  ridge_lo = ridge_beta_ci[,1],
  ridge_hi = ridge_beta_ci[,2],
  ridge_Pabs_gt_0.1 = ridge_prob_nonzero,
  lasso_median = lasso_beta_median,
  lasso_lo = lasso_beta_ci[,1],
  lasso_hi = lasso_beta_ci[,2],
  lasso_Pabs_gt_0.1 = lasso_prob_nonzero
)

# Rank by lasso "importance" then ridge
comp <- comp[order(-comp$lasso_Pabs_gt_0.1, -abs(comp$lasso_median)), ]

cat("\n--- Top predictors by Bayesian LASSO P(|beta|>0.1) ---\n")
print(head(comp, 10), row.names = FALSE)


