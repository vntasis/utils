#!/usr/bin/env python

# import numpy as np
# import umap
# from sklearn.manifold import TSNE, Isomap
# import phate
# import pymde

from pathlib import Path

import pandas as pd
import typer
from plotnine import aes, geom_point, ggplot
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing_extensions import Annotated

dreduc = typer.Typer(no_args_is_help=True)
global_opts = {}


# Function for reading and scaling the input
def preprocess(input_file: Path, scale: bool):
    """
    Read and standardize the data
    """

    data = pd.read_csv(input_file)
    data = data.values
    if scale:
        data = StandardScaler().fit_transform(data)

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
            help="Number of components to keep for reduction."
        ),
    ] = 2,
    plot: Annotated[
        bool,
        typer.Option(
            "--plot",
            help="Make a scatterplot of the two first Principal Components"
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
        pca_plot = ggplot(df, aes(x="PC1", y="PC2")) + geom_point()
        pca_plot.save("PCA_plot.pdf")


if __name__ == "__main__":
    dreduc()
