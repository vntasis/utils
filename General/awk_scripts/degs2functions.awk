#!/usr/bin/awk -f

#=============================================================================
# MATCH EACH GENE WITH ITS FUNCTIONAL CATEGORY.
#
# This script converts a genes per functional category list into a functional
# categories per gene list.
#
# Two tabular files are required as input. The first one is a tsv and contains
# in the first column Gene IDs. The second one is a csv file, each field is
# also flanked by double quotes, and each record of this file is a functional
# category that has come up by an functional enrichment analysis of the genes
# of file 1. The first column contains the term name for the corresponding
# functional category, the second column contains the term ID and the 10th
# column contain a list of a subset gene IDs from file 1 that belong to the
# corresponding enriched functional category. This script will return the
# functional categories, that each gene (one per line in file 1) has been
# detected in.
#
# This script will return two column tab delimited file. The first column
# contains a comma-separated list of term names of enriched fuctional
# categories that the corresponding gene has been found in. The second column
# contains a comma-separated list of the corresponding term IDs
#=============================================================================

BEGIN {
    OFS="\t"
    FS="\t"
}

NR==FNR && FNR > 1 {
    id[FNR]=$1
    next
}

NR!=FNR && FNR==1 { FS="\",\"" }

FNR > 1 {
    term_name[FNR]=$2
    term_id[FNR]=$3
    interactions[FNR]=$10
}

END {
    for (i in term_name) {
        for (j in id) {
            if (match(interactions[i],id[j])) {
                matched_term_names[j]=matched_term_names[j]","term_name[i]
                matched_term_ids[j]=matched_term_ids[j]","term_id[i]
            }
        }
    }

    print "Term_Names","Term_IDs"
    for (k=2; k<=(NR-FNR); k++) {
        print matched_term_names[k], matched_term_ids[k]
    }
}

