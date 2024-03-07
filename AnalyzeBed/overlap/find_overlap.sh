#!/bin/bash

set -eu

#************************************************************
## Find overlap stats for a pair of sets of genomic intervals
## provided as two bed files (sorted and without duplicates)
## requirements:
## - bedtools
#************************************************************

# Input
setA=$1
setB=$2
nameA=$(basename -s '.bed' "$setA")
nameB=$(basename -s '.bed' "$setB")


#----------------------------------------------------------
# Calculate the Initial Number of interval for the two sets
#----------------------------------------------------------
echo 'Count the number of genomic intervals'
NumberA=$(wc -l "$setA" | awk '{ print $1 }')
NumberB=$(wc -l "$setB" | awk '{ print $1 }')


#-------------------------------------------------
# Calculate the Jaccard statistic for the two sets
#-------------------------------------------------
echo 'Calculate the Jaccard statistic for the two sets'
jac_stat=$(bedtools jaccard -a "$setA" -b "$setB" | awk 'NR==2{ print $3 }')



#------------------------------------------------------------
# Caclulate the numbers of intervals that overlap in each set
# and the amount of overlap, in terms of number of bases or a
# Jaccard index for every pair of intervals that intesect.
#------------------------------------------------------------
echo 'Calculate detailed overlap'
bedtools window -a "$setA" -b "$setB" -w 0 | \
    bedtools overlap -i stdin -cols 2,3,5,6 | \
    awk '
    BEGIN { OFS="\t" }
    {
        intersect = $7
        start = ($2<$5) ? $2 : $5;
        end = ($3>$6) ? $3 : $6;
        union = end - start;
        jaccard = intersect/union;
        print $0, jaccard
    }' > \
    "${nameA}_${nameB}_overlap.tsv"

mean_jac_stat=$(awk '{ count+=$8 }END{ print count/NR }' "${nameA}_${nameB}_overlap.tsv")



#-------------------------------------------------------------
# Calculate the number of intervals that intersect in each set
#-------------------------------------------------------------
echo 'Calculate the number of intervals that intersect in each set'
NinterA=$(cut -f1-3 "${nameA}_${nameB}_overlap.tsv" | sort -k1,1 -k2,2n -k3,3n -u | wc -l)
NinterB=$(cut -f4-6 "${nameA}_${nameB}_overlap.tsv" | sort -k1,1 -k2,2n -k3,3n -u | wc -l)




#-----------------------------------------------------------------
# Calculate the distribution of relative distances between the two
# sets of intervals
#-----------------------------------------------------------------
echo 'Calculate the distribution of relative distances between the two sets of intervals'
bedtools reldist -a "$setA" -b "$setB" | \
    awk '
    BEGIN{ OFS="," }
    { print $1,$4 }' > "${nameA}_${nameB}_reldist.csv"




#----------------------------------------------------------------
# Produce a report of the numbers and the proportion of intervals
# that overlap in each set
#----------------------------------------------------------------
echo 'Produce report for the number of intervals that overlap'
awk -v numA="$NumberA" \
    -v numB="$NumberB" \
    -v interA="$NinterA" \
    -v interB="$NinterB" \
    -v nameA="$nameA" \
    -v nameB="$nameB" \
    -v jac="$jac_stat" \
    -v av_jac="$mean_jac_stat" \
    ' BEGIN{
        OFS="\t";
        print nameA,numA;
        print nameB,numB;
        print "Overlapping_"nameA, interA;
        print "Overlapping_"nameB,interB;
        print "Proportion_"nameA,interA/numA;
        print "Proportion_"nameB,interB/numB;
        print "Jaccard_index",jac;
        print "Average_Jaccard_index",av_jac;

        exit;
    }' > "${nameA}_${nameB}_report.tsv"


echo 'Done!'
