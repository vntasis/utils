#!/usr/bin/env python

# import numpy as np
# import umap
# import phate
# import pymde

from pathlib import Path

import pandas as pd
import typer
from plotnine import aes, geom_point, ggplot, labs, theme_bw
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from sklearn.preprocessing import StandardScaler
from typing_extensions import Annotated

dreduc = typer.Typer(no_args_is_help=True)
global_opts = {}


# Function for reading and scaling the input
def preprocess(
    input_file: Path,
    scale: bool,
    do_pca: bool = False,
    var_expl: float = 0.95
):
    """
    Read and standardize the data
    Perform a PCA if necessary
    """

    data = pd.read_csv(input_file)
    data = data.values
    if scale:
        data = StandardScaler().fit_transform(data)

    if do_pca:
        pca = PCA(n_components=var_expl)
        data = pca.fit_transform(data)

    return data


# Use callback for global options and documentation
@dreduc.callback()
def main(
    input_file: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            writable=False,
            readable=True,
            resolve_path=True,
            help="Path to the input data file.",
        ),
    ],
    output_file: Annotated[
        str, typer.Option(help="Path to save the reduced data.")
    ] = "data_reduced.tsv",
    scale: Annotated[
        bool,
        typer.Option(help="Scale data before any dim-reduc method is applied.")
    ] = True,
):
    """
    Apply dimensionality reduction methods in a dataset
    """

    # Save global options in a list
    global_opts["input"] = input_file
    global_opts["scale"] = scale
    global_opts["output"] = output_file


# Command for PCA
@dreduc.command()
def pca(
    n_components: Annotated[
        int,
        typer.Option(
            min=2,
            help="Number of Components to keep for reduction."
        ),
    ] = 2,
    plot: Annotated[
        bool,
        typer.Option(
            "--plot",
            help="Make a scatterplot of the two first Principal Components."
        ),
    ] = False,
):
    """
    Perform PCA for dimensionality reduction.
    """
    # Read data
    data = preprocess(global_opts["input"], global_opts["scale"])

    # Perform PCA
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)

    # Write Principal Components to a file
    df = pd.DataFrame(
        reduced_data, columns=[f"PC{i}" for i in range(1, n_components + 1)]
    )
    df.to_csv(global_opts["output"], index=False)

    # Make PCA scatterplot
    if plot:
        var_explained = pca.explained_variance_ratio_ * 100
        pca_plot = (
            ggplot(df, aes(x="PC1", y="PC2"))
            + geom_point()
            + theme_bw()
            + labs(
                x=f"PC1 ({var_explained[0]:.2f}%)",
                y=f"PC2 ({var_explained[1]:.2f}%)"
            )
        )
        pca_plot.save("PCA_plot.pdf")


# Command for Isomap
@dreduc.command()
def isomap(
    n_components: Annotated[
        int,
        typer.Option(
            min=2,
            help="Number of Components to keep for reduction."
        ),
    ] = 2,
    n_neighbors: Annotated[
        int,
        typer.Option(help="Number of neighbors to consider for Isomap."),
    ] = 5,
    plot: Annotated[
        bool,
        typer.Option(
            "--plot",
            help="Make a scatterplot of the two first Isomap Components."
        ),
    ] = False,
):
    """
    Perform Isomap for dimensionality reduction.
    """
    # Read data
    data = preprocess(global_opts["input"], global_opts["scale"])

    # Perform Isomap
    isomap = Isomap(n_neighbors=n_neighbors, n_components=n_components)
    reduced_data = isomap.fit_transform(data)

    # Write Isomap Components to a file
    df = pd.DataFrame(
        reduced_data,
        columns=[f"Isomap{i}" for i in range(1, n_components + 1)]
    )
    df.to_csv(global_opts["output"], index=False)

    # Make Isomap scatterplot
    if plot:
        isomap_plot = (
            ggplot(df, aes(x="Isomap1", y="Isomap2"))
            + geom_point()
            + theme_bw()
        )
        isomap_plot.save("Isomap_plot.pdf")


# Command for t-SNE
@dreduc.command()
def tsne(
    n_components: Annotated[
        int,
        typer.Option(
            min=2,
            help="Number of Components to keep for reduction."
        ),
    ] = 2,
    perplexity: Annotated[
        float,
        typer.Option(help="Perplexity parameter for t-SNE."),
    ] = 30.0,
    n_iter: Annotated[
        int,
        typer.Option(help="Number of iterations for t-SNE."),
    ] = 1000,
    verbose: Annotated[
        int,
        typer.Option(help="Verbosity level."),
    ] = 0,
    pca_before: Annotated[
        bool,
        typer.Option(
            "--pca_before",
            help="Whether to perform PCA before t-SNE."
        ),
    ] = False,
    var_explained: Annotated[
        float,
        typer.Option(
            min=0,
            max=1,
            help="""
            If --pca_before is set, this specifies the proportion of variance
            for the PCA Components to be kept.
            """,
        ),
    ] = 0.95,
    plot: Annotated[
        bool,
        typer.Option(
            "--plot",
            help="Make a scatterplot of the two first Principal Components."
        ),
    ] = False,
):
    """
    Perform t-SNE for dimensionality reduction.
    """
    # Read data
    data = preprocess(
        global_opts["input"],
        global_opts["scale"],
        do_pca=pca_before,
        var_expl=var_explained,
    )

    # Perform t-SNE
    tsne = TSNE(
        n_components=n_components, perplexity=perplexity,
        n_iter=n_iter, verbose=verbose
    )
    reduced_data = tsne.fit_transform(data)

    # Write t-SNE Components to a file
    df = pd.DataFrame(
        reduced_data, columns=[f"t-SNE{i}" for i in range(1, n_components + 1)]
    )
    df.to_csv(global_opts["output"], index=False)

    # Make t-SNE scatterplot
    if plot:
        tsne_plot = (
            ggplot(df, aes(x="t-SNE1", y="t-SNE2"))
            + geom_point()
            + theme_bw()
        )
        tsne_plot.save("tSNE_plot.pdf")


if __name__ == "__main__":
    dreduc()
