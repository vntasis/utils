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

- Filter lines based on an arbitrary threshold
```
awk '$3 > 10' file
```

- Subset every line given a regular expression
```
awk 'BEGIN { regex="ENST[0-9]{11}.[0-9]" } { match($0, regex); print substr($0,RSTART,RLENGTH); }'
```

- Detect lines that match a pattern
```
awk '$0~"protein_coding"'
```
