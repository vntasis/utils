#!/usr/bin/env python

#==========================================
## Plot bam statistics output from bamstats
#==========================================
## This script will take as parameter the
## path to a directory that contains json
## files outputed by bamstats
## (https://github.com/guigolab/bamstats)
## and will produce two plots that
## summarize the results
##
## Positional parameters:
## @param1 path to dir with json files
## @param2 width of the pdf containing
## the barplot
## @ param3 height of the pdf containing
## the barplot
##
## Returns:
## Two pdf files
## 1. A barplot of the number of mapped
## reads per sample
## 2. A boxplot with the proportion of
## mapped reads in different genomic
## regions e.g. exon, intron, intergenic
#------------------------------------------


# Import required libraries
import json
import pandas as pd
import sys
import os
import numpy as np
from plotnine import *
from dfply import *


# Function that reads the path to a json
# file and returns the data in a pandas df
def read_stats_json(file_path):
    with open(file_path) as f:
        json_file = json.load(f)
    df = pd.json_normalize(json_file)
    return df

# Main function
def main():

    # Input parameters
    bamstats_dir = sys.argv[1]
    plot_width = float(sys.argv[2])
    plot_height = float(sys.argv[3])

    # Read all json files, turn into a panda dataframe
    bamstats_files = os.listdir(bamstats_dir)
    bamstats_df = read_stats_json(bamstats_dir + bamstats_files[0])
    for i in range(1,len(bamstats_files)):
        df = read_stats_json(bamstats_dir + bamstats_files[i])
        bamstats_df = bamstats_df.append(df, ignore_index=True)

    # Summarize the number of total - mapped - uniqule mapped reads
    mapped_reads = \
        bamstats_df >> \
        rename(Total='general.reads.total', Mapped='coverage.total.total', Uniquely_mapped='coverageUniq.total.total') >> \
        gather('reads', 'number_of_reads',['Total', 'Mapped', 'Uniquely_mapped'], add_id=True) >> \
        select(['_ID', 'reads', 'number_of_reads'])

    # Plot a barplot with the number of reads per sample
    plot = (ggplot(mapped_reads, \
        aes(x='factor(_ID)', y='number_of_reads', fill='reads'))
        + geom_col(stat='identity', position='dodge')
        + theme_bw()
        + scale_fill_brewer(type = 'qual', palette='Dark2')
        + xlab('Sample')
        + ylab('Number of reads'))

    plot.save('mapped_reads.pdf', width=plot_width, height=plot_height)


    # Summarize the proportion of mapped reads in different
    # genomic regions, e.g exon, intergenic, intron
    bamstats_df.columns = bamstats_df.columns.str.replace('.', '_')
    proportion_reads = \
        bamstats_df >> \
        mutate(\
        all__exonic_intronic=(X.coverage_total_exonic_intronic /  X.coverage_total_total), \
        all__intron=(X.coverage_total_intron /  X.coverage_total_total), \
        all__exon=(X.coverage_total_exon /  X.coverage_total_total), \
        all__intergenic=(X.coverage_total_intergenic /  X.coverage_total_total), \
        all__others=(X.coverage_total_others /  X.coverage_total_total), \
        \
        continuous__exonic_intronic=(X.coverage_continuous_exonic_intronic /  X.coverage_total_total), \
        continuous__intron=(X.coverage_continuous_intron /  X.coverage_total_total), \
        continuous__exon=(X.coverage_continuous_exon /  X.coverage_total_total), \
        continuous__intergenic=(X.coverage_continuous_intergenic /  X.coverage_total_total), \
        continuous__others=(X.coverage_continuous_others /  X.coverage_total_total), \
        continuous__total=(X.coverage_continuous_total /  X.coverage_total_total), \
        \
        split__exonic_intronic=(X.coverage_split_exonic_intronic /  X.coverage_total_total), \
        split__intron=(X.coverage_split_intron /  X.coverage_total_total), \
        split__exon=(X.coverage_split_exon /  X.coverage_total_total), \
        split__intergenic=(X.coverage_split_intergenic /  X.coverage_total_total), \
        split__others=(X.coverage_split_others /  X.coverage_total_total), \
        split__total=(X.coverage_split_total /  X.coverage_total_total), \
        ) >> \
        select(contains('__')) >> \
        gather('category', 'proportion') >> \
        separate(X.category, ['mapping_type', 'mapped_region'], '__', True, extra='merge')


        # Plot boxplots with the different proportions of mapped reads
    plot = (ggplot(proportion_reads, \
        aes(x='mapped_region', y='proportion', fill='mapping_type'))
        + geom_boxplot()
        + theme_bw()
        + scale_y_continuous(breaks = np.arange(0, 1.1, 0.1))
        + scale_fill_brewer(type = 'qual', palette='Dark2')
        + xlab('')
        + ylab('Proportion of mapped reads'))

    plot.save('mapped_proportion.pdf', width=10, height=7)


if __name__ == '__main__':
    main()
