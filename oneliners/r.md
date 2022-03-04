# R one-liners
---

## Use of [littler](https://github.com/eddelbuettel/littler) for quick generation of stats and plots

- __Summary stats for a variable__
```
awk 'BEGIN{ print "my_var" }1' my_var.txt | r -de 'print(summary(X[,1]))'
```

- Frequencies of values of discrete variable
```
awk 'BEGIN{ print "my_var" }1' my_var.txt | r -de 'print(table(X[,1]))'
```

- __Plot a histogram with the distribution of a variable__
```
awk 'BEGIN{ print "my_var" }1' my_var.txt | r -de 'pdf("my_var.pdf"); hist(X[,1]); dev.off();'
```

- __Generate random numbers from the standard normal distribution__
```
r -e 'write(rnorm(5), stdout())' | awk '$1=$1' OFS="\n"
```

- __Calculate correlation coefficient for two variables of interest__
```
awk 'BEGIN { OFS=","; print "x","y" } $1=$1' my_vars.tsv | r -de 'print(cor(X$x, X$y))'
```

- __Student's t test__
```
awk 'BEGIN { OFS=","; print "x","y" } $1=$1' my_vars.tsv | r -de 'print(t.test(X$x, X$y))'
```
