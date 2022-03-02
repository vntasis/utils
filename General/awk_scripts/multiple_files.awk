#!/usr/bin/awk -f

# Example script for processing multiple files

fname != FILENAME { fname = FILENAME; idx++ }

idx == 1 { discount[$1] = $2 }
idx == 2 { price[$1] = $2 * ( 1 - discount[$3] ) }
idx == 3 { printf "%s $%.2f %s\n",$1, price[$1]*$2, $3 }
