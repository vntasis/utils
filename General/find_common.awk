#!/usr/bin/awk -f

#------------------------------------------------------
# Given two files that each contain a list of elements,
# print the elements that are common between them.
#------------------------------------------------------

NR==FNR { set1[$1]=1; next; }
{ set2[$1]=1 }
END { for (element in set1){ if (set2[element]==1) print element }}
