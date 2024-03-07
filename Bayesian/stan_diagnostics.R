#!/usr/bin/env r

#-------------------------
## Load required libraries
#-------------------------

suppressMessages(suppressWarnings(library(docopt)))
suppressMessages(suppressWarnings(library(rstan)))
suppressMessages(suppressWarnings(library(bayesplot)))
suppressMessages(suppressWarnings(library(ggplot2)))
suppressMessages(suppressWarnings(library(magrittr)))

#------------------------------------
## Define/Read command-line arguments
#------------------------------------
doc <-
'Generate diagnostics plots after a cmdstan run

Usage:
  stan_diagnostics.R -c CHAINS -s SAMPLES_DIR -i ID -p PARAMETERS
  stan_diagnostics.R (-h | --help)

Options:
  -h --help                     Show help message and exit.
  -c --chains CHAINS            Number of chains that have been run.
  -s --samples_dir SAMPLES_DIR  Directorty with the sample files of the MCMC run(s).
  -i --id ID                    ID of the sample files, e.g. for "experiment1_model1_1.csv" the ID is "experiment1_model1".
  -p --parameters PARAMETERS    Parameters to plot diagnostics for. It is a comma-separated string.
                                E.g. "alpha,beta,sigma".
'

opt <- docopt(doc)

samples_dir <- opt$samples_dir
id <- opt$id
chains <- as.integer(opt$chains)
parameters <- unlist(strsplit(opt$parameters, ","))



#----------------------------------------------------
## Read csv files containing draws from the posterior
#----------------------------------------------------
paths <- paste0(samples_dir, "/", id, "_",
                as.character(1:chains),
                ".csv")

fit <- read_stan_csv(paths)

posterior <- as.array(fit)


pdf(paste0(id, "_diagnostics.pdf"))

#----------------------------
## Plot credibility intervals
#----------------------------
# (quantile-based)

print(mcmc_intervals(posterior, pars = parameters))


#-------------
## Diagnostics
#-------------

lp <- log_posterior(fit)
np <- nuts_params(fit)

# scatterplot with divergences
color_scheme_set("darkgray")
print(mcmc_pairs(posterior, np = np, pars = parameters,
                 off_diag_args = list(size = 0.75)))

# trace plot with divergences
color_scheme_set("mix-brightblue-gray")
print(mcmc_trace(posterior, pars = parameters, np = np) +
        xlab("Post-warmup iteration"))

# energy histograms
print(mcmc_nuts_energy(np))

# Convergence
print(mcmc_nuts_divergence(np, lp))

# rhat
color_scheme_set("brightblue")
print(mcmc_rhat(rhat = rhat(fit)) + yaxis_text(hjust = 0))

# neff
print(mcmc_neff(neff_ratio(fit), size = 2) + yaxis_text(hjust = 0))

# autocorrelation
print(mcmc_acf(posterior, pars = parameters, lags = 10))


dev.off()
