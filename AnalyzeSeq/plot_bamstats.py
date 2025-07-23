#!/usr/bin/env python

"""Plot bam statistics output from bamstats

This script will take as parameter the
path to a directory that contains json
files outputed by bamstats
(https://github.com/guigolab/bamstats)
and will produce two plots that
summarize the results

Positional parameters:
@param1 path to dir with json files
@param2 width of the pdf files
@ param3 height of the pdf files

Returns:
Two pdf files
1. A barplot of the number of mapped
reads per sample
2. A boxplot with the proportion of
mapped reads in different genomic
regions e.g. exon, intron, intergenic
"""


# Import required libraries
import json
import pandas as pd
import sys
import os
import numpy as np
from plotnine import *


# Define a recursive function to remove a specific key from a nested
# dictionary or list
def remove_key(data, key):
    if isinstance(data, dict):
        for k, v in list(data.items()):
            if k == key:
                del data[k]
            else:
                remove_key(v, key)
    elif isinstance(data, list):
        for item in data:
            remove_key(item, key)


# Function that reads the path to a json
# file and returns the data in a pandas df
def read_stats_json(file_path):
    # Open json file
    with open(file_path) as f:
        json_file = json.load(f)

    # Remove Insert Sizes entry
    # It is not required, and it has huge size
    remove_key(json_file, 'insert_sizes')

    # Convert json dictionary to a pandas df
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
    bamstats_df = read_stats_json(os.path.join(bamstats_dir,
                                               bamstats_files[0]))
    for i in range(1, len(bamstats_files)):
        df = read_stats_json(os.path.join(bamstats_dir, bamstats_files[i]))
        bamstats_df = bamstats_df.append(df, ignore_index=True)

    # Summarize the number of total - mapped - uniquely mapped reads
    mapped_reads = (
        bamstats_df.rename(
            columns={
                'general.reads.total': 'Total',
                'coverage.total.total': 'Mapped',
                'coverageUniq.total.total': 'Uniquely_mapped'
            })
            .loc[:, ['Total', 'Mapped', 'Uniquely_mapped']]
            .melt(
                value_vars=['Total', 'Mapped', 'Uniquely_mapped'],
                var_name='reads', value_name='number_of_reads', ignore_index=False
            )
            .reset_index()
            .rename(columns={'index': 'ID'})
     )

    # Plot a barplot with the number of reads per sample
    plot = (ggplot(mapped_reads, \
        aes(x='factor(ID)', y='number_of_reads', fill='reads'))
        + geom_col(stat='identity', position='dodge')
        + theme_bw()
        + scale_fill_brewer(type = 'qual', palette='Dark2')
        + xlab('Sample')
        + ylab('Number of reads'))

    plot.save('mapped_reads.pdf', width=plot_width, height=plot_height)

    # Summarize the proportion of mapped reads in different
    # genomic regions, e.g exon, intergenic, intron
    bamstats_df.columns = bamstats_df.columns.str.replace('.', '_')
    proportion_reads = (
        bamstats_df.assign(
            all__exonic_intronic=(bamstats_df.eval("coverage_total_exonic_intronic /  coverage_total_total")),
            all__intron=(bamstats_df.eval("coverage_total_intron /  coverage_total_total")),
            all__exon=(bamstats_df.eval("coverage_total_exon /  coverage_total_total")),
            all__intergenic=(bamstats_df.eval("coverage_total_intergenic /  coverage_total_total")),
            all__others=(bamstats_df.eval("coverage_total_others /  coverage_total_total")),
            \
            continuous__exonic_intronic=(bamstats_df.eval("coverage_continuous_exonic_intronic /  coverage_total_total")),
            continuous__intron=(bamstats_df.eval("coverage_continuous_intron /  coverage_total_total")),
            continuous__exon=(bamstats_df.eval("coverage_continuous_exon /  coverage_total_total")),
            continuous__intergenic=(bamstats_df.eval("coverage_continuous_intergenic /  coverage_total_total")),
            continuous__others=(bamstats_df.eval("coverage_continuous_others /  coverage_total_total")),
            continuous__total=(bamstats_df.eval("coverage_continuous_total /  coverage_total_total")),
            \
            split__exonic_intronic=(bamstats_df.eval("coverage_split_exonic_intronic /  coverage_total_total")),
            split__intron=(bamstats_df.eval("coverage_split_intron /  coverage_total_total")),
            split__exon=(bamstats_df.eval("coverage_split_exon /  coverage_total_total")),
            split__intergenic=(bamstats_df.eval("coverage_split_intergenic /  coverage_total_total")),
            split__others=(bamstats_df.eval("coverage_split_others /  coverage_total_total")),
            split__total=(bamstats_df.eval("coverage_split_total /  coverage_total_total"))
        )
        .filter(like='__', axis='columns')
        .melt(var_name='category', value_name='proportion')
        .assign(
            mapping_type = lambda x: x['category'].str.split('__', expand=True) [0],
            mapped_region = lambda x: x['category'].str.split('__', expand=True) [1]
        )
        .drop(columns='category')
    )

    # Plot boxplots with the different proportions of mapped reads
    plot = (ggplot(proportion_reads, \
        aes(x='mapped_region', y='proportion', fill='mapping_type'))
        + geom_boxplot()
        + theme_bw()
        + scale_y_continuous(breaks = np.arange(0, 1.1, 0.1))
        + scale_fill_brewer(type = 'qual', palette='Dark2')
        + xlab('')
        + ylab('Proportion of mapped reads'))

    plot.save('mapped_proportion.pdf', width=plot_width, height=plot_height)


if __name__ == '__main__':
    main()
