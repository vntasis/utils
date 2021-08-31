#!/usr/bin/awk -f

# Input file: Every record (row) represent a transcript - enhancer pair
# Need variables
# echr: the number of column with chromosome name
# estart: the number of column with the start cordinate of enhancers
# gid: the number of column with the gene ID

BEGIN { OFS="\t"; eend=estart+1 }

# distinct enhancer gene pairs
!seen[$echr","$estart","$eend","$gid]++ { pairs++ }

# distinct genes
!seen[$gid]++ { genes++ }

# distinct enhancers
!seen[$echr","$estart","$eend]++ { enhancers++ }

END {
  print "Number of enhancer - transcript pairs:", NR
  print "Number of enhancer - gene pairs:", pairs
  print "Number of enhancers:", enhancers
  print "Number of genes:", genes
}
