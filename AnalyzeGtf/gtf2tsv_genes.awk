#!/usr/bin/awk -f

#=============================================================================
# This script takes as input a GTF annotation file and outputs a tab delimited
# file with the following fields for every Gene record: ChromosomeName, Start,
# End, Strand, GeneID, GeneType, GeneName
#=============================================================================


BEGIN {
    OFS="\t"
    geneid_regex = "ENSG[0-9]{11}[.]?[0-9]*(_PAR_Y)?"
    type_regex = "gene_type[^;]+;?"
    name_regex = "gene_name[^;]+;?"
}

$3=="gene" {
    chr = $1
    start = $4
    end = $5
    strand = $7
    match($0, geneid_regex); geneid = substr($0,RSTART,RLENGTH)
    match($0, type_regex); type = substr($0,RSTART+11,RLENGTH-13)
    match($0, name_regex); name = substr($0,RSTART+11,RLENGTH-13)

    print chr, start, end, strand, geneid, type, name
}
