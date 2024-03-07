#!/usr/bin/awk -f

#=============================================================================
# Usage:
#   gtf2tsv.awk -- [ -g | -t ] GTF_FILE
#
# This script takes as input a GTF annotation file and outputs a tab delimited
# file with the following fields:
#
# GENES MODE (Default)
# For every Gene record: ChromosomeName, Start,
# End, Strand, GeneID, GeneType, GeneName
#
# TRANSCRIPTS MODE
# For every Transcript record: ChromosomeName,
# Start, End, Strand, GeneID, GeneType, TranscriptID, TranscriptType, GeneName
#=============================================================================

BEGIN {
    mode = "genes"
    for (i = 1; i < ARGC; i++) {
        if (ARGV[i] == "-g")
            mode = "genes"
        else if (ARGV[i] == "-t")
            mode = "transcripts"
        else if (ARGV[i] ~ /^-./) {
            e = sprintf("%s: unrecognized option -- %c",
                ARGV[0], substr(ARGV[i], 2, 1))
            print e > "/dev/stderr"; exit
        } else break
        delete ARGV[i]
    }
    OFS="\t"
    geneid_regex = "ENSG[0-9]{11}[.]?[0-9]*(_PAR_Y)?"
    gtype_regex = "gene_type[^;]+;?"
    transid_regex = "ENST[0-9]{11}[.]?[0-9]*(_PAR_Y)?"
    ttype_regex = "transcript_type[^;]+;?"
    name_regex = "gene_name[^;]+;?"
}


$3=="gene" && mode=="genes" {
    chr = $1
    start = $4
    end = $5
    strand = $7
    match($0, geneid_regex); geneid = substr($0,RSTART,RLENGTH)
    match($0, gtype_regex); type = substr($0,RSTART+11,RLENGTH-13)
    match($0, name_regex); name = substr($0,RSTART+11,RLENGTH-13)

    print chr, start, end, strand, geneid, type, name
}

$3=="transcript" && mode=="transcripts" {
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
