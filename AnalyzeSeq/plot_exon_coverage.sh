#!/bin/bash

#===============================================================================
## The purpose of this script is to evaluate the exon coverage of an RNA
## sequencing experiment, so as to identify potential systematic biases. It uses
## deepTools (https://github.com/deeptools/deepTools), in order to:
## - Compute a bigWig file from mapped reads (bam file)
## - Compute a matrix with the read coverage across all given annotated exons.
##   These will be scaled to have the same length, 1000 bases.
## - Cluster exons according to their coverage profile using kmeans.
## - Produce a pileup heatmap with the raw exon coverage and a profile of the
##   standard deviation of coverage across exon length.
##
## Requirements:
## - deepTools
##
## Positional arguments:
## 1. Bam file containing mapped reads from RNA sequencing.
## 2. File containing annotated exons to be checked (Bed6 format).
## 3. Number of threads to be used.
## 4. Number of clusters to be identified.
#===============================================================================

set -euo pipefail

# Input
bamfile=$1
exonBedfile=$2
threads=$3
k_cluster=$4
#filter=$5
filename=$(basename -s '.bam' $bamfile)

# Compute bigWig file
bamCoverage --binSize 10 -p "$threads" \
    -b "$bamfile" \
    -o "${filename}.bw"

# Compute pileup matrix
computeMatrix scale-regions -p "$threads" \
    -R "$exonBedfile" -S "${filename}.bw" \
    --regionBodyLength 1000 --skipZeros \
    -o "${filename}.gz" --binSize 10 \
    --missingDataAsZero --quiet

# Plot pileup heatmap of exon coverage
# and standard deviation profile
plotHeatmap \
    -m "${filename}.gz" -out "${filename}_coverage_heatmap.pdf" \
    --averageTypeSummaryPlot median --plotType se \
    --plotFileFormat pdf --dpi 100 --kmeans "$k_cluster" \
    --colorList '#fee6ce,#fdae6b,#e6550d'

plotProfile \
    -m "${filename}.gz" -o "${filename}_std_profile.pdf" \
    --dpi 100 --plotFileFormat pdf \
    --kmeans "$k_cluster" --averageType std
