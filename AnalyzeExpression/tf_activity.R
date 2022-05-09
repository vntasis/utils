#!/usr/bin/env r

#-------------------------
## Load required libraries
#-------------------------

suppressMessages(suppressWarnings(library(docopt)))
suppressMessages(suppressWarnings(library(tidyverse)))
suppressMessages(suppressWarnings(library(magrittr)))
suppressMessages(suppressWarnings(library(vroom)))
library(decoupleR)
suppressMessages(suppressWarnings(library(pheatmap)))
suppressMessages(suppressWarnings(library(paletteer)))



#------------------------------------
## Define/Read command-line arguments
#------------------------------------
'Transcription Factor Activity Analysis

Usage:
  tf_activity.R --counts COUNTS [--diff_expr DIFF_EXPR] [--tf_net TF_NET] [--n_tf N_TF]
  tf_activity.R (-h | --help)
  tf_activity.R --version

Options:
  -h --help              Show help message and exit.
  --version              Show version.
  --counts COUNTS        Matrix with expression counts. Rows represent genes and columns represent samples.
  --diff_expr DIFF_EXPR  Matrix with results from Differential Expression Analysis.
                         Rows represent genes and columns different statistics e.g. Log2FC, t-stat, p-value.
  --tf_net TF_NET        Matrix with transcription factor network. Rows represent an edge on the network.
                         Columns represent different features of those edges, i.e. source, target, weight, confidence.
  --n_tf N_TF            Number of TFs with top activity scores to include in the plots. [default: 25]
' -> doc

opt <- docopt(doc, version = 'TF Activity Analysis 1.0\n')
#print(opt)



#-------------------
## Read input tables
#-------------------


# Counts matrix
counts <- suppressMessages(vroom(opt$counts))
counts %<>%
  dplyr::mutate_if(~ any(is.na(.x)), ~ if_else(is.na(.x),0,.x)) %>%
  column_to_rownames('ID') %>%
  as.matrix()
#head(counts) %>% print


# Differential expression results
if (!(is.null(opt$diff_expr))) {

  diff_expr <- suppressMessages(vroom(opt$diff_expr))
  # Differentially expressed entries
  de <-
    diff_expr %>%
    filter(P.Value <= 0.05) %>%
    pull(ID)
  #head(de) %>% print
  #length(de) %>% print

  # Tibble to matrix
  diff_expr %<>%
    mutate(FC_Pvalue = -log10(P.Value) * logFC) %>%
    column_to_rownames('ID') %>%
    as.matrix()
  #head(diff_expr) %>% print


}


# TF Network.
# If it is not provided fetch Dorothea network from OmniPath
if (!(is.null(opt$tf_net))) {

  tf_net <- suppressMessages(vroom(opt$tf_net))

} else {

  tf_net <-
    get_dorothea(organism='human', levels=c('A', 'B', 'C'))

}
#head(tf_net) %>% print



#--------------------------------------------------
## Inference TF Activity based on expression counts
#--------------------------------------------------

# If differential expression analysis results are provided,
# filter count matrix to get only differentially expressed
# entries

if (!(is.null(opt$diff_expr))) {
  counts <- counts[de,]
}
#print(dim(counts))

activity <-
  decouple(counts, tf_net,
           .source = 'source', .target = 'target',
           statistics = c("mlm", "ulm", "wsum", "fgsea"),
           args = list(
                       fgsea = list(times = 100, nproc = 4),
                       mlm = list(.mor = "mor"),
                       ulm = list(.mor = "mor"),
                       wsum = list(.mor = "mor", times = 100)),
           consensus_stats = list('mlm','ulm','norm_wsum','norm_fgsea')) %>%
  filter(statistic == 'consensus')
#print(activity)



#--------------------------------------------------------
## Inference TF Activity based on differential expression
#--------------------------------------------------------

if (!(is.null(opt$diff_expr))) {

  activity_de <-
    decouple(diff_expr[, 'FC_Pvalue', drop=FALSE], tf_net,
             .source = 'source', .target = 'target',
             statistics = c("mlm", "ulm", "wsum", "fgsea"),
             consensus_score = F,
             args = list(
                         fgsea = list(times = 100, nproc = 4),
                         mlm = list(.mor = "mor"),
                         ulm = list(.mor = "mor"),
                         wsum = list(.mor = "mor", times = 100))) %>%
    mutate(condition = 'FC_Pvalue') %>%
    filter(statistic  %in% c('mlm','ulm','norm_wsum','norm_fgsea')) %>%
    run_consensus %>%
    filter(statistic == 'consensus')
    #print(activity_de)

}



#------------------------------
## Save output activity to disk
#------------------------------

write.table(activity, file = "tf_activity_exp.tsv", sep="\t",
            quote=F, row.names=F)

if (!(is.null(opt$diff_expr))) {
  write.table(activity_de, file = "tf_activity_diff_exp.tsv",
              sep="\t", quote=F, row.names=F)
}



#--------------
## Plot results
#--------------

n_tfs <-
  opt$n_tf %>%
  as.integer

pdf('tf_activity.pdf', width = 10)

# Plot results based expression counts #
# Transform to wide matrix
activity_mat <-
  activity %>%
  pivot_wider(id_cols = 'condition',
              names_from = 'source',
              values_from = 'score') %>%
  column_to_rownames('condition') %>%
  as.matrix()

# Get top tfs with more variable means across clusters
tfs <-
  activity %>%
  group_by(source) %>%
  summarise(std = sd(score)) %>%
  arrange(-abs(std)) %>%
  head(n_tfs) %>%
  pull(source) %>%
  suppressMessages

activity_mat <-
  activity_mat[,tfs] %>%
  scale


# Choose color palette
palette_length = 100
#my_color = colorRampPalette(c("Darkblue", "white","red"))(palette_length)
col <- paletteer_c("viridis::cividis", n = palette_length)

brk <- c(seq(-3, 0, length.out=ceiling(palette_length/2) + 1),
               seq(0.05, 3, length.out=floor(palette_length/2)))


# Plot
print(pheatmap(activity_mat, color = col, breaks=brk))


# Plot results based on differential expression #
# Add rank column
if (!(is.null(opt$diff_expr))) {
  activity_de %<>%
    mutate(rnk = 0)

  # Filter top TFs in both signs
  msk <- activity_de$score > 0
  activity_de[msk, 'rnk'] <- rank(-activity_de[msk, 'score'])
  activity_de[!msk, 'rnk'] <- rank(-abs(activity_de[!msk, 'score']))

  tfs <-
    activity_de %>%
    arrange(rnk) %>%
    head(n_tfs) %>%
    pull(source)

  activity_de %<>%
    dplyr::filter(source %in% tfs)

  # Plot
  print(
        ggplot(activity_de, aes(x = reorder(source, score), y = score)) +
          geom_bar(aes(fill = score), stat = "identity") +
          scale_fill_gradient2(low = "darkblue", high = "indianred", mid = "whitesmoke", midpoint = 0) +
          theme_bw() +
          theme(axis.title = element_text(face = "bold", size = 12),
                axis.text.x = element_text(angle = 45, hjust = 1, size =10, face= "bold"),
                axis.text.y = element_text(size =10, face= "bold"),
                panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
          xlab("Pathways")
  )

}
dev.off()
