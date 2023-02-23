#!/bin/bash

#=================================================
## Fetch Quantification data from Grape-NF output
## Run with RSEM!
##
## Required utilities: bash, awk, cat, cut, paste,
## mktemp, xargs, echo, ls
#=================================================

set -euo pipefail

# Functions
function finish {
    rm -r "$tmp_dir1" "$tmp_dir2" "${type}_tmp.tsv"
}

# Input
qt=$1
fi=$2
db=$3


# Set fetching variables
## Quantification type to be fetched
[ "$qt" = 'genes' ] && \
    type='GeneQuantifications'
[ "$qt" = 'transcripts' ] && \
    type='TranscriptQuantifications'

## Field to be fetched from the rsem tsv output
[ "$fi" = 'fpkm' ] && \
    field=7
[ "$fi" = 'tpm' ] && \
    field=6
[ "$fi" = 'raw' ] && \
    field=5
[ "$fi" = 'length' ] && \
    field=3
[ "$fi" = 'ef_length' ] && \
    field=4



# Create a temporary directories
tmp_dir1=$(mktemp -d -p .)
tmp_dir2=$(mktemp -d -p .)

# Fetch data
awk -v type="$type" \
    -v field="$field" \
    -v tmp_dir="$tmp_dir1" '
    BEGIN { OFS="," } $5==type { print $1,$3,field,tmp_dir }' "$db" | \
    xargs -I % sh -c 'cut -f $(echo % | cut -d ',' -f3) \
    $(echo % | cut -d ',' -f2) > \
    $(echo % | cut -d ',' -f4)/$(echo % | cut -d ',' -f1)'

# Paste data for all samples
paste "$tmp_dir1/"* | \
    awk 'NR > 1' > "${type}_tmp.tsv"

# Create a header denoting the sample
ls "$tmp_dir1" | paste -s > "$tmp_dir2/header"

# Fetch Gene / Transcript ids
sample=$(awk -v type="$type" '$5==type && !seen[$5]++ { print $3 }' "$db")
awk '{ print $1 }' "$sample" > "$tmp_dir2/ids"

# Create final output
cat "$tmp_dir2/header" "${type}_tmp.tsv" | \
    paste "$tmp_dir2/ids" - > \
    "${type}_${fi}.tsv"

# Clean up
trap finish EXIT
