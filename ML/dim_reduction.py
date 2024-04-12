#!/usr/bin/env python

import ast
from pathlib import Path

import pandas as pd
import typer
from plotnine import aes, geom_point, ggplot, labs, theme_bw
from sklearn.decomposition import PCA
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


# Function for parsing argument string
def parse_kwargs_string(kwargs_string: str) -> dict:
    """
    Parse a string of comma-separated key-value pairs into a dictionary.
    The values are evaluated using ast.literal_eval().

    Args:
        kwargs_string (str): A string of comma-separated key-value pairs.
            Example: "x=5, y=10, z='hello'"

    Returns:
        dict: A dictionary containing the parsed key-value pairs.
            Example: {'x': 5, 'y': 10, 'z': 'hello'}
    """
    kwargs_list = [param.strip() for param in kwargs_string.split(",")]
    kwargs_dict = dict(param.split("=") for param in kwargs_list)

    for key, value in kwargs_dict.items():
        kwargs_dict[key] = ast.literal_eval(value)

    return kwargs_dict


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
    from sklearn.manifold import Isomap

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
            help="Make a scatterplot of the two first t-SNE Components."
        ),
    ] = False,
):
    """
    Perform t-SNE for dimensionality reduction.
    """
    from sklearn.manifold import TSNE

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


# Command for UMAP
@dreduc.command()
def umap(
    n_components: Annotated[
        int,
        typer.Option(
            min=2,
            help="Number of Components to keep for reduction."
        ),
    ] = 2,
    n_neighbors: Annotated[
        int,
        typer.Option(
            help="Number of neighbors to use for constructing the UMAP graph."
        ),
    ] = 15,
    min_dist: Annotated[
        float,
        typer.Option(
            help="Minimum distance between points in the embedded space."
        ),
    ] = 0.1,
    metric: Annotated[
        str,
        typer.Option(help="Distance metric to use."),
    ] = "euclidean",
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            help="Status updates during the optimization process."
        ),
    ] = False,
    umap_kwargs: Annotated[
        str,
        typer.Option(help="String with arbitrary extra parameters for UMAP."),
    ] = "",
    pca_before: Annotated[
        bool,
        typer.Option(
            "--pca_before",
            help="Whether to perform PCA before UMAP."
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
            help="Make a scatterplot of the two first UMAP Components."
        ),
    ] = False,
):
    """
    Perform UMAP for dimensionality reduction.
    """
    from umap import UMAP

    # Read data
    data = preprocess(
        global_opts["input"],
        global_opts["scale"],
        do_pca=pca_before,
        var_expl=var_explained,
    )

    # Parse UMAP kwargs string to dictionary
    umap_kwargs_dict = parse_kwargs_string(umap_kwargs) if umap_kwargs else {}

    # Perform UMAP
    umap = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        verbose=verbose,
        **umap_kwargs_dict,
    )
    reduced_data = umap.fit_transform(data)

    # Write UMAP Components to a file
    df = pd.DataFrame(
        reduced_data, columns=[f"UMAP{i}" for i in range(1, n_components + 1)]
    )
    df.to_csv(global_opts["output"], index=False)

    # Make UMAP scatterplot
    if plot:
        umap_plot = (
            ggplot(df, aes(x="UMAP1", y="UMAP2"))
            + geom_point()
            + theme_bw()
        )
        umap_plot.save("UMAP_plot.pdf")


# Command for PHATE
@dreduc.command()
def phate(
    n_components: Annotated[
        int,
        typer.Option(
            min=2,
            help="Number of Components to keep for reduction."
        ),
    ] = 2,
    knn: Annotated[
        int,
        typer.Option(
            help="Number of nearest neighbors on which to build kernel."
        ),
    ] = 15,
    decay: Annotated[
        int,
        typer.Option(
            help="Decay parameter for the potential of heat-diffusion."
        ),
    ] = 40,
    knn_dist: Annotated[
        str,
        typer.Option(help="Distance metric to use for building kNN graph."),
    ] = "euclidean",
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            help="Status updates during the optimization process."
        ),
    ] = False,
    phate_kwargs: Annotated[
        str,
        typer.Option(help="String with arbitrary extra parameters for PHATE."),
    ] = "",
    pca_before: Annotated[
        bool,
        typer.Option(
            "--pca_before",
            help="Whether to perform PCA before PHATE."
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
            help="Make a scatterplot of the two first PHATE Components."
        ),
    ] = False,
):
    """
    Perform PHATE for dimensionality reduction.
    """
    from phate import PHATE

    # Read data
    data = preprocess(
        global_opts["input"],
        global_opts["scale"],
        do_pca=pca_before,
        var_expl=var_explained,
    )

    # Parse PHATE kwargs string to dictionary
    phate_kwargs_dict = parse_kwargs_string(
        phate_kwargs
    ) if phate_kwargs else {}

    # Perform PHATE
    phate = PHATE(
        n_components=n_components,
        knn=knn,
        decay=decay,
        knn_dist=knn_dist,
        verbose=verbose,
        **phate_kwargs_dict,
    )
    reduced_data = phate.fit_transform(data)

    # Write PHATE Components to a file
    df = pd.DataFrame(
        reduced_data, columns=[f"PHATE{i}" for i in range(1, n_components + 1)]
    )
    df.to_csv(global_opts["output"], index=False)

    # Make PHATE scatterplot
    if plot:
        phate_plot = (
            ggplot(df, aes(x="PHATE1", y="PHATE2"))
            + geom_point()
            + theme_bw()
        )
        phate_plot.save("PHATE_plot.pdf")


# Command for PyMDE
@dreduc.command()
def pymde(
    embedding_dim: Annotated[
        int,
        typer.Option(
            min=2,
            help="Number of embedding dimensions."
        ),
    ] = 2,
    preserve_neighbors: Annotated[
        bool,
        typer.Option(
            "--preserve_neighbors/--preserve_distances",
            help="""
            --preserve_neighbors creates embeddings that focus on the local
            structure of the data.
            --preserve_distances focuses more on the global structure.
            """
        ),
    ] = True,
    n_neighbors: Annotated[
        int,
        typer.Option(
            help="""
            Number of nearest neighbors to compute for each row. A sensible
            value is chosen by default, depending on the number of items. Used
            when --preserve_neighbors is selected.
            """
        ),
    ] = None,
    standardized_constraint: Annotated[
        bool,
        typer.Option(
            "--standardized_constraint",
            help="""
            This causes the embedding to have uncorrelated columns, and
            prevents it from spreading out too much.
            """
        ),
    ] = False,
    seed: Annotated[
        int,
        typer.Option(
            help="""
            A random seed that PyMDE uses in various preprocessing methods. Use
            this for reproducible output.
            """
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            help="Status updates during the optimization process."
        ),
    ] = False,
    pymde_kwargs: Annotated[
        str,
        typer.Option(help="String with arbitrary extra parameters for PyMDE."),
    ] = "",
    pca_before: Annotated[
        bool,
        typer.Option(
            "--pca_before",
            help="Whether to perform PCA before PyMDE."
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
            help="Make a scatterplot of the two first PyMDE Components."
        ),
    ] = False,
):
    """
    Perform PyMDE for dimensionality reduction. Minimum-Distortion Embedding
    """
    import pymde

    # Read data
    data = preprocess(
        global_opts["input"],
        global_opts["scale"],
        do_pca=pca_before,
        var_expl=var_explained,
    )

    # Parse PyMDE kwargs string to dictionary
    pymde_kwargs_dict = parse_kwargs_string(
        pymde_kwargs
    ) if pymde_kwargs else {}

    # Constraint
    constraint = pymde.Standardized() if standardized_constraint else None

    # Seed
    if seed:
        pymde.seed(seed)

    # Perform PyMDE
    if preserve_neighbors:
        mde = pymde.preserve_neighbors(
            data=data,
            embedding_dim=embedding_dim,
            n_neighbors=n_neighbors,
            constraint=constraint,
            verbose=verbose,
            **pymde_kwargs_dict,
        )
    else:
        mde = pymde.preserve_distances(
            data=data,
            embedding_dim=embedding_dim,
            constraint=constraint,
            verbose=verbose,
            **pymde_kwargs_dict,
        )

    reduced_data = mde.embed(verbose=verbose)

    # Write PyMDE Components to a file
    df = pd.DataFrame(
        reduced_data,
        columns=[f"PyMDE{i}" for i in range(1, embedding_dim + 1)],
    )
    df.to_csv(global_opts["output"], index=False)

    # Make PyMDE scatterplot
    if plot:
        pymde_plot = (
            ggplot(df, aes(x="PyMDE1", y="PyMDE2"))
            + geom_point()
            + theme_bw()
        )
        pymde_plot.save("PyMDE_plot.pdf")


if __name__ == "__main__":
    dreduc()
