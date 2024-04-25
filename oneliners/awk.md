# Awk one-liners

- __Print the number of columns of a table__
```
awk '{ print NR; exit}' file
```

- __Change delimiter of a file with tabular structure__
```
awk '$1=$1' FS="\t" OFS=":" file
```

- __Remove duplicate lines__
```
awk '!seen[$0]++' file
```

- __Concatenate every 5 lines of input with a comma__
```
awk 'ORS=NR%5?",":"\n"' file
```

- __Calculate the arithmetic mean of a column__
```
awk '{ sum+=$1 } END { print sum/NR }' file
```

- __Calculate the median of a set of values__
```
sort -n values.txt | awk '{ values[NR]=$1 } END { if (NR % 2) { print values[(NR + 1) / 2] } else { print (values[NR/2] + values[NR/2 + 1]) / 2.0 }}'
```

- __Calculate the 95th percentile of a set of values__
```
sort -n values.txt | awk '{ values[NR]=$1 } END { print values[int(NR*0.95)] }'
```

- __Filter lines based on an arbitrary threshold__
```
awk '$3 > 10' file
```

- __Subset every line given a regular expression__
```
awk 'BEGIN { regex="ENST[0-9]{11}.[0-9]" } { match($0, regex); print substr($0,RSTART,RLENGTH); }'
```

- __Detect lines that match a pattern__
```
awk '$0~"protein_coding"'
```

- __Split a file in two for every 8 lines__
```
awk 'NR%8==1 || NR%8==2 || NR%8==3 || NR%8==4 { print $0 > "read1.fastq" } NR%8==5 || NR%8==6 || NR%8==7 || NR%8==0 { print $0 > "read2.fastq" }' reads.fastq
```
