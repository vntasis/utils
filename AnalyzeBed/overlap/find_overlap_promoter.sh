#!/bin/bash

set -eu

#************************************************************
## Find overlap stats for a pair of sets of genomic intervals
## provided as two bed files (sorted and without duplicates)
## Focus overlap in promoter regions.
## Input arguments:
## 1. File containing the coordinates of genes or transcripts
## It has to be in bed6 format. 4th column sould be the
## name of the gene
## 2. File containing the list of intervals to be overlapped
## 3. File containing a list of the chromosome in the
## genome along with their length in bps (e.g. chr20 100)
## 4. The number of bases -/+ of TSS to search for overlap
## with the promoter regions. It has to be in bed3 format.
## requirements:
## - bedtools
#************************************************************


# Input
setA=$1
setB=$2
genome=$3
Npromoter=$4
nameA=$(basename -s '.bed' "$setA")
nameB=$(basename -s '.bed' "$setB")



#------------------------------------------------------------------
#-Use bedtools flank to isolate promoter region upstream of the TSS
#-Use bedtools window to overlap setB with the pomoter extended
# downstream by the same number of bps as the size of the promoter
#-Fro every TSS of setA get the number of intervals in setB that
# overlap.
#------------------------------------------------------------------
echo 'Calculate overlap with promoters'
bedtools flank -i "$setA" -g "$genome" -l "$Npromoter" -r 0 -s | \
    sort -k1,1 -k2,2n -k3,3n -u | \
    bedtools window -a stdin -b "$setB" -r "$Npromoter" -l 0 -sw \
    > "${nameA}_${nameB}_overlap.tsv"



#--------------------------------------------------------------------
# Get the proportion of promoters in setA that overlap with intervals
# in setB and the average number of intervals in setB that overlap
#--------------------------------------------------------------------

echo 'Calculate proportion of promoters that overlap'
bedtools flank -i "$setA" -g "$genome" -l "$Npromoter" -r 0 -s | \
    sort -k1,1 -k2,2n -k3,3n -u | \
    bedtools window -a stdin -b "$setB" -r "$Npromoter" -l 0 -sw -c | \
    awk '
    !seen1[$4]++ { Ngenes++ }
    $7 > 0{ NsetB+=$7; NsetA++ }
    !seen2[$4]++ && $7 > 0 { Ngenes_positive++ }
    END{
        if ( NsetA > 0 ) {
            OFS="\t";
            print "Proportion of genes in setA whith overlapped promoters:", Ngenes_positive/Ngenes;
            print "Proportion of promoters in setA overlapping:", NsetA/NR;
            print "Average Number of intervals in setB that overlap:", NsetB/NsetA;
        }
    }' > "${nameA}_${nameB}_report.tsv"


echo 'Done!'
