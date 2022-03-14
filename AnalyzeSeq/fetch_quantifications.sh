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
    rm -r "$tmp_dir" "${type}_tmp.tsv"
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
[ "$fi" = 'ef.length' ] && \
    field=4



# Create a temporary directory
tmp_dir=$(mktemp -d -p .)

# Fetch data
awk -v type="$type" '
    BEGIN { OFS="," } $5==type { print $1,$3 }' "$db" | \
    xargs -I % sh -c "cut -f $field \
    $(echo % | cut -d ',' -f2) > \
    $tmp_dir/$(echo % | cut -d ',' -f1)"

# Paste data for all samples
paste "$tmp_dir/*" \
    awk 'NR > 1' > "${type}_tmp.tsv"

# Create a header denoting the sample
ls "$tmp_dir" | paste -s > "$tmp_dir/header"

# Fetch Gene / Transcript ids
sample=$(awk -v type="$type" '
    $5==type & NR==1 { print $3 }' "$db")
awk '{ print $1 }' "$sample" > "$tmp_dir/ids"

# Create final output
cat "$tmp_dir/header" "${type}_tmp.tsv" | \
    paste "$tmp_dir/ids" - > \
    "$type.tsv"

# Clean up
trap finish EXIT
