#!/bin/bash

set -euo pipefail

function finish {
    rm  "${nameA}_${nameB}.tmp"
}

#======================================================================================================================
# The purpose of this script is to assess the overlap enrichments between two bed files in a record by record fashion.
# The important file for which we want to assess enrichment and significance of overlaps is the "context" file
# this is compared to a "peaks" file carrying the regions of interest for which we want to assess the enrichment
# Enrichment is calculated on an obs/exp ratio. Significance is assessed via a number of permutations chosen by the user.
# Input bed files have to be sorted and without duplicates.
# Requirements: bedtools, awk, sort, bash
#
# Positional arguments:
# 1. Peaks file
# 2. Context file
# 3. Genome (chrom) size file
# 4. Number of permutations
#
# Output tsv content:
# 1. Peaks filename
# 2. Context filename
# 3. Proportion of the genome covered by Context
# 4. Proportion of Peaks bases that overlap with the Ccontext
# 5. Enrichment value (i.e. 4. / 3. )
# 6. P-value calculated with permutations (shuffling) of intervals in Peaks file
#======================================================================================================================

#-----------------
# Input paramaters
#-----------------

peaks=$1 # the peaks file
context=$2 # the "context" file (promoters, genes, syntenic regions etc)
genomefile=$3 # genome file - It contains the sequence length per scaffold
bs=$4 # number of permutations

nameA=$(basename -s '.bed' "$peaks")
nameB=$(basename -s '.bed' "$context")

#---------------------
# Calculate enrichment
#---------------------

size_peaks=$(awk '{val+=($3-$2)}END{print val}' "$peaks") # Raw coverage of Peak file
gsize=$(awk '{sum+=$2}END{print sum}' "$genomefile") # Size of Genome
context_coverage=$(awk -v gs="$gsize" '{val+=($3-$2)}END{print val/gs}' "$context") # Coverage of context file

# Coverage of intersection
overlap_proportion=$(intersectBed -sorted -wao -a "$peaks" -b "$context" | \
    awk -v size="$size_peaks" '$NF!=0 { val+=$NF } END { print (val/size) }')

# Key Calculation: Enrichment of Peak/Context Overlap vs Expected Ratio
enr=$(awk -v overlap="$overlap_proportion" -v context="$context_coverage" 'BEGIN { print overlap/context; exit; }')



echo "Enrichment of overlap=" $enr
echo "Calculating significance"
echo "Grab a coffee, this may take a while..."


#-------------------
# Perform permutations
#-------------------

counter=1

while [ $counter -le "$bs" ]
do
    # creating random permutation of Peaks file
    # Results stored in temp file
    shuffleBed -i "$peaks" -g "$genomefile" | \
    sort -k1,1 -k2,2n -k3,3n | \
    intersectBed -sorted -wao -a - -b "$context" | \
        awk -v size="$size_peaks" -v ratio="$context_coverage" \
        '$NF!=0 { val+=$NF } END { print (val/size)/ratio }' >> "${nameA}_${nameB}.tmp"

    # report progress
    echo "$counter" | awk '{if ($1%100==0) { print "simulations="$1 }}'


        #echo $counter
        ((counter++))
done



#--------------------------------
# Calculating permutation p-value
#--------------------------------

pvalue=$(\
    awk -v en="$enr" '
    {
        if (((en>1) && ($1>=en)) || ((en<1) && ($1<=en))) { val++ }
    }
    END{ print val/NR }'  "${nameA}_${nameB}.tmp" | \
    \
    awk -v lim="$bs" '{ if ($1==0) { print 1/lim } else { print $1 } }'\
)



#-------------
# Write output
#-------------

awk -v peaks="$nameA" \
    -v con="$nameB" \
    -v con_cov="$context_coverage" \
    -v ov_prop="$overlap_proportion" \
    -v enr="$enr" \
    -v pvalue="$pvalue" \
    ' BEGIN{
        OFS="\t";
        print "Peaks:", peaks;
        print "Context:", con;
        print "Context coverage:", con_cov;
        print "Overlap proportion:", ov_prop;
        print "Enrichment:", enr;
        print "P-value:", pvalue;

        exit;
    }' > ${nameA}_${nameB}_enrichment.tsv


# Clean up
trap finish EXIT
