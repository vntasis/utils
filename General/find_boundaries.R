#!/usr/bin/env Rscript

#------------------------------------------------------------------------
## Find boundaries in a square matrix
#------------------------------------------------------------------------
## The current code takes as input an (ordered) square matrix.
## It could be, for instance, a correlation matrix. If that matrix is
## ordered in a way, such that separated clustered regions occur, one
## might be interested in identifying the positions where the boundaries
## of those regions are located. This is the objective of the following
## function. It will calculate an average neighbourhood signal along the
## diagonal of the input matrix, and then it will try to identify local
## minima on this signal.
##
## Required R packages:
## - pracma
##
## Positional arguments should be supplied in the command line with
## the following order
##
## @param data                     Input square matrix in a csv file
##
## @param window_size              Window size defining neighbourhood.
##                                 Default value: 10
##
## @param valley_height_threshold  Threshold imposed on the signal of the
##                                 identified valleys. Default value: 0.3
##
## @param na_rm                    Boolean parameter indicating if NA
##                                 values in the input matrix should be
##                                 ignored. Default value: True
#------------------------------------------------------------------------


find_boundaries <- function(data, window_size = 10,
                            valley_height_threshold = 0.3,
                            na_rm = TRUE) {
  window <- window_size
  size <- nrow(data)
  signal <- numeric()

  # Calculate neighbourhood average signal
  for (i in 1:(size - 1)) {
    lowerbound <- max(1, i - window + 1)
    upperbound <- min(i + window, size)
    diamond <- data[lowerbound:i, (i + 1):upperbound]
    signal <- c(signal, mean(diamond, na.rm = na_rm))
  }

  # Identify valleys/boundaries
  signal <- -signal
  thres <- -valley_height_threshold
  valleys <-
    pracma::findpeaks(signal, minpeakheight = thres)

  if (is.null(valleys)) {
    return("No boundaries detected. Consider adjusting parameter values.")
  }
  # Return output
  valleys[, 1] <- -valleys[, 1]
  colnames(valleys) <- c("Signal", "Valley_position", "Valley_start",
                         "Valley_end")

  as.data.frame(valleys)
}


#---------------------------------------------------------
## Read argument from command line and run find_boundaries
#---------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)

# Default parameter values
window_size <- 10
valley_height_threshold <- 0.3
na_rm <- TRUE


# Test if there is at least one argument: if not, return an error
# Otherwise save the right parameters
if (length(args) == 0) {
  stop("At least one argument must be supplied (input file)!", call. = FALSE)
}

if (length(args) >= 1) {
  data_file <- args[1]
}

if (length(args) >= 2) {
  window_size <- as.numeric(args[2])
}

if (length(args) >= 3) {
  valley_height_threshold <- as.numeric(args[3])
}

if (length(args) >= 4) {
  na_rm <- as.logical(args[4])
}

data <- read.csv(data_file, header = FALSE)
data <- as.matrix(data)
print(find_boundaries(data, window_size, valley_height_threshold, na_rm))
