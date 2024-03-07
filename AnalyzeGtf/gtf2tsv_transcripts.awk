#!/usr/bin/awk -f

#=============================================================================
# This script takes as input a GTF annotation file and outputs a tab delimited
# file with the following fields for every Transcript record: ChromosomeName,
# Start, End, Strand, GeneID, GeneType, TranscriptID, TranscriptType, GeneName
#=============================================================================

BEGIN {
    OFS="\t"
    geneid_regex = "ENSG[0-9]{11}[.]?[0-9]*(_PAR_Y)?"
    transid_regex = "ENST[0-9]{11}[.]?[0-9]*(_PAR_Y)?"
    gtype_regex = "gene_type[^;]+;?"
    ttype_regex = "transcript_type[^;]+;?"
    name_regex = "gene_name[^;]+;?"
}

$3=="transcript" {
    chr = $1
    start = $4
    end = $5
    strand = $7
    match($0, geneid_regex); geneid = substr($0,RSTART,RLENGTH)
    match($0, gtype_regex); gtype = substr($0,RSTART+11,RLENGTH-13)
    match($0, transid_regex); transid = substr($0,RSTART,RLENGTH)
    match($0, ttype_regex); ttype = substr($0,RSTART+17,RLENGTH-19)
    match($0, name_regex); name = substr($0,RSTART+11,RLENGTH-13)

    print chr, start, end, strand, geneid, gtype, transid, ttype, name
}
