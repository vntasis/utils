#!/usr/bin/awk -f

#----------------------------------------
# Print a range of columns from a file
# Required variables
# f: rank of the column to start printing
# l: rank of the column to stop printing
#----------------------------------------

BEGIN { OFS="\t" }

{
	for (i=f; i<=l; i++) printf("%s%s",$i,(i==l) ? "\n" : OFS)
}
