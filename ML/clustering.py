#!/usr/bin/env python

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from plotnine import aes, geom_point, ggplot, labs, theme_bw
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing_extensions import Annotated


cluster = typer.Typer(no_args_is_help=True)
global_opts = {}


# Function for reading and scaling the input
def preprocess(
    input_file: Path,
    scale: bool = False,
    do_pca: bool = False,
    var_expl: float = 0.95
):
    """
    Read and standardize the data
    Perform a PCA
    """

    data_df = pd.read_csv(input_file)
    data = data_df.values
    if scale:
        data = StandardScaler().fit_transform(data)

    pca = PCA(n_components=var_expl)
    if do_pca:
        data = pca.fit_transform(data)
        pca_data = data
    else:
        pca_data = pca.fit_transform(data)

    return data_df, data, pca_data, pca


# Function for reading file with labels
def read_labels(label_file: Path) -> pd.DataFrame:
    """
    Read labels from a file and check for the label column.

    Args:
        label_file (str): Path to the file containing labels.

    Returns:
        pd.DataFrame: DataFrame containing labels.
    """
    labels = pd.read_csv(label_file)
    if "label" not in labels.columns:
        raise ValueError("Label file must contain a 'label' column.")
    return labels


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


# Function for making a scatterplot of the data
def plot_data(
        df: pd.DataFrame, vars_to_plot: list, labels_exist: bool
) -> ggplot:
    """
    Make a scatterplot of the data annotated with the cluster membership, and
    return the plot object.

    Args:
        df: DataFrame containing the data, and the cluster information.
        label_file: Are extra user-provided labels inculded in df?
        vars_to_plot: Column names of the variables to be included in the plot

    Returns:
        ggplot: Plot object.
    """

    if labels_exist:
        plot = (
            ggplot(
                df,
                aes(x=vars_to_plot[0], y=vars_to_plot[1],
                    color="cluster", shape="label")
            )
            + geom_point()
            + theme_bw()
        )
    else:
        plot = (
            ggplot(
                df,
                aes(x=vars_to_plot[0], y=vars_to_plot[1], color="cluster")
            )
            + geom_point()
            + theme_bw()
        )

    return plot


# Use callback for global options and documentation
@cluster.callback()
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
            help="""
            Path to the input data file. It has to be a csv file with a header
            (column names).
            """,
        ),
    ],
    output_file: Annotated[
        str, typer.Option(help="Path to save the cluster labels.")
    ] = "data_clusters.txt",
    scale: Annotated[
        bool,
        typer.Option(
            "--scale",
            help="Scale data before any clustering method is applied."
        )
    ] = False,
    pca_before: Annotated[
        bool,
        typer.Option(
            "--pca-before",
            help="Whether to perform PCA before clustering."
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
        str,
        typer.Option(
            help="""
            The column names of two of the variables in the data. The names
            should be separated by a comma. These two variables will be used to
            make a scatterplot, which will also be annotated with the results
            of the clustering.
            """
        ),
    ] = "",
    pca_plot: Annotated[
        bool,
        typer.Option(
            "--pca-plot",
            help="""
            Perform a PCA, and make a scatterplot of the two first principal
            components. Annotate the plot with the results of the clustering.
            """
        ),
    ] = False,
    label_file: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            writable=False,
            readable=True,
            resolve_path=True,
            help="""
            Path to a file containing labels to be considered in the
            scatterplot(s). The shape of the point in the plot(s) will follow
            the labels in this file. It must contain a column named 'label'.
            """,
        ),
    ] = None,
):
    """
    Apply clustering analysis with the selected method.
    For more details on the methods, check:
    https://scikit-learn.org/stable/modules/clustering.html

    Examples:

    clustering.py --input-file input.csv --scale
    --pca-plot kmeans --n-clusters 3

    clustering.py --input-file input.csv --scale --pca-plot dbscan --eps 0.8

    clustering.py --input-file input.csv --scale --pca-plot hdbscan
    --min-cluster-size 10

    clustering.py --input-file input.csv --scale --pca-plot optics --xi 0.2
    """

    # Save global options in a list
    global_opts["input"] = input_file
    global_opts["output"] = output_file
    global_opts["labels"] = label_file
    global_opts["scale"] = scale
    global_opts["pca_before"] = pca_before
    global_opts["var_explained"] = var_explained
    global_opts["plot"] = plot
    global_opts["pca_plot"] = pca_plot


# Command for kmeans
@cluster.command()
def kmeans(
    n_clusters: Annotated[
        int,
        typer.Option(
            min=1,
            help="""
            The number of clusters to form as well as the number of centroids
            to generate.
            """,
        ),
    ],
    init: Annotated[
        str,
        typer.Option(
            help="""
            Method for initialization.
            It can take the following values: k-means++, random.
            """
        ),
    ] = "k-means++",
    algorithm: Annotated[
        str,
        typer.Option(
            help="""
            K-Means algorithm to use.
            It can take the following values: lloyd, elkan.
            """
        ),
    ] = "lloyd",
    seed: Annotated[
        int,
        typer.Option(help="Seed for making the clustering deterministic."),
    ] = 123,
    verbose: Annotated[
        int,
        typer.Option(help="Verbosity level."),
    ] = 0,
    kmeans_kwargs: Annotated[
        str,
        typer.Option(
            help="String with arbitrary extra parameters for KMeans."
        ),
    ] = "",
):
    """
    Perform K-Means clustering.

    Examples:

    clustering.py --input-file input.csv --scale --pca-plot kmeans --n-clusters
    3 --seed 32 --init random --kmeans-kwargs 'n_init=15,max_iter=500'
    --verbose 1
    """
    from sklearn.cluster import KMeans

    # Read data
    data_df, data, pca_data, pca = preprocess(
        global_opts["input"],
        global_opts["scale"],
        do_pca=global_opts["pca_before"],
        var_expl=global_opts["var_explained"],
    )

    # Parse KMeans kwargs string to dictionary
    kmeans_kwargs_dict = parse_kwargs_string(kmeans_kwargs) if kmeans_kwargs else {}

    # Seed
    np.random.seed(seed)

    # Perform K-Means
    kmeans = KMeans(
        n_clusters=n_clusters,
        init=init,
        algorithm=algorithm,
        verbose=verbose,
        **kmeans_kwargs_dict,
    )
    clusters = kmeans.fit(data)

    # K-Means output
    cluster_labels = clusters.labels_
    inertia = clusters.inertia_
    niter = clusters.n_iter_

    # Write K-Means cluster labels to a file
    df = pd.DataFrame(cluster_labels.astype(str), columns=["cluster"])
    df.to_csv(global_opts["output"], index=False)

    # Make scatterplot(s)
    labels_exist = False
    if global_opts["labels"]:
        labels = read_labels(global_opts["labels"])
        df["label"] = labels["label"]
        labels_exist = True

    if global_opts["plot"]:
        vars_to_plot = [var.strip() for var in global_opts["plot"].split(",")]
        df[vars_to_plot[0]] = data_df[vars_to_plot[0]]
        df[vars_to_plot[1]] = data_df[vars_to_plot[1]]

        kmeans_plot = (
                plot_data(df, vars_to_plot, labels_exist)
                + labs(
                    title="""
                    K-Means Clustering
                    Inertia: {:.2f} | Number of Runs: {}
                    """.format(inertia, niter)
                )
        )
        kmeans_plot.save("K-Means_plot.pdf")

    if global_opts["pca_plot"]:
        var_explained = pca.explained_variance_ratio_ * 100
        df["PC1"] = pca_data[:, 0]
        df["PC2"] = pca_data[:, 1]

        kmeans_pca_plot = (
                plot_data(df, ["PC1", "PC2"], labels_exist)
                + labs(
                    title="""
                    K-Means Clustering
                    Inertia: {:.2f} | Number of Runs: {}
                    """.format(inertia, niter),
                    x=f"PC1 ({var_explained[0]:.2f}%)",
                    y=f"PC2 ({var_explained[1]:.2f}%)"
                )
        )
        kmeans_pca_plot.save("K-Means_pca_plot.pdf")


# Command for dbscan
@cluster.command()
def dbscan(
    eps: Annotated[
        float,
        typer.Option(
            min=0,
            help="""
            The maximum distance between two samples for one to be considered
            as in the neighborhood of the other. This is not a maximum bound on
            the distances of points within a cluster. This is the most
            important DBSCAN parameter to choose appropriately for your data
            set and distance function.
            """,
        ),
    ] = 0.5,
    min_samples: Annotated[
        int,
        typer.Option(
            min=2,
            help="""
            The number of samples (or total weight) in a neighborhood for a
            point to be considered as a core point. This includes the point
            itself. If min_samples is set to a higher value, DBSCAN will find
            denser clusters, whereas if it is set to a lower value, the found
            clusters will be more sparse.
            """,
        ),
    ] = 5,
    algorithm: Annotated[
        str,
        typer.Option(
            help="""
            The algorithm to be used by the NearestNeighbors module to compute
            pointwise distances and find nearest neighbors.
            It can take the following values: auto, ball_tree, kd_tree, brute.
            """,
        ),
    ] = "auto",
    dbscan_kwargs: Annotated[
        str,
        typer.Option(
            help="String with arbitrary extra parameters for DBSCAN."
        ),
    ] = "",
):
    """
    Perform DBSCAN clustering.
    Density-Based Spatial Clustering of Applications with Noise

    Examples:

    clustering.py --input-file input.csv --scale --pca-plot dbscan --eps 0.3
    --min-samples 10 --dbscan-kwargs 'metric="cosine",n_jobs=2'
    """
    from sklearn.cluster import DBSCAN

    # Read data
    data_df, data, pca_data, pca = preprocess(
        global_opts["input"],
        global_opts["scale"],
        do_pca=global_opts["pca_before"],
        var_expl=global_opts["var_explained"],
    )

    # Parse DBSCAN kwargs string to dictionary
    dbscan_kwargs_dict = parse_kwargs_string(dbscan_kwargs) if dbscan_kwargs else {}

    # Perform DBSCAN
    dbscan = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        algorithm=algorithm,
        **dbscan_kwargs_dict,
    )
    clusters = dbscan.fit(data)

    # DBSCAN output
    cluster_labels = clusters.labels_
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)

    # Write DBSCAN cluster labels to a file
    df = pd.DataFrame(cluster_labels.astype(str), columns=["cluster"])
    df.to_csv(global_opts["output"], index=False)

    # Make scatterplot(s)
    labels_exist = False
    if global_opts["labels"]:
        labels = read_labels(global_opts["labels"])
        df["label"] = labels["label"]
        labels_exist = True

    if global_opts["plot"]:
        vars_to_plot = [var.strip() for var in global_opts["plot"].split(",")]
        df[vars_to_plot[0]] = data_df[vars_to_plot[0]]
        df[vars_to_plot[1]] = data_df[vars_to_plot[1]]

        dbscan_plot = (
                plot_data(df, vars_to_plot, labels_exist)
                + labs(
                    title="""
                    DBSCAN Clustering
                    Number of clusters: {} | Number of noise samples: {}
                    """.format(n_clusters, n_noise)
                )
        )
        dbscan_plot.save("DBSCAN_plot.pdf")

    if global_opts["pca_plot"]:
        var_explained = pca.explained_variance_ratio_ * 100
        df["PC1"] = pca_data[:, 0]
        df["PC2"] = pca_data[:, 1]

        dbscan_pca_plot = (
                plot_data(df, ["PC1", "PC2"], labels_exist)
                + labs(
                    title="""
                    DBSCAN Clustering
                    Number of clusters: {} | Number of noise samples: {}
                    """.format(n_clusters, n_noise),
                    x=f"PC1 ({var_explained[0]:.2f}%)",
                    y=f"PC2 ({var_explained[1]:.2f}%)"
                )
        )
        dbscan_pca_plot.save("DBSCAN_pca_plot.pdf")


# Command for hdbscan
@cluster.command()
def hdbscan(
    min_cluster_size: Annotated[
        int,
        typer.Option(
            min=2,
            help="""
            The minimum number of samples in a group for that group to be
            considered a cluster; groupings smaller than this size will be left
            as noise.
            """,
        ),
    ] = 5,
    min_samples: Annotated[
        int,
        typer.Option(
            min=2,
            help="""
            The number of samples in a neighborhood for a point to be
            considered as a core point. This includes the point itself. When
            None, defaults to min_cluster_size.
            """,
        ),
    ] = None,
    algorithm: Annotated[
        str,
        typer.Option(
            help="""
            Exactly which algorithm to use for computing core distances.
            It can take the following values: auto, ball_tree, kd_tree, brute.
            """,
        ),
    ] = "auto",
    hdbscan_kwargs: Annotated[
        str,
        typer.Option(
            help="String with arbitrary extra parameters for HDBSCAN."
        ),
    ] = "",
):
    """
    Perform HDBSCAN clustering.
    Hierarchical Density-Based Spatial Clustering of Applications with Noise

    Examples:

    clustering.py --input-file input.csv --scale --pca-plot hdbscan
    --min-cluster-size 10 --hdbscan-kwargs
    'metric="cosine",n_jobs=2,leaf_size=50'
    """
    from sklearn.cluster import HDBSCAN

    # Read data
    data_df, data, pca_data, pca = preprocess(
        global_opts["input"],
        global_opts["scale"],
        do_pca=global_opts["pca_before"],
        var_expl=global_opts["var_explained"],
    )

    # Parse HDBSCAN kwargs string to dictionary
    hdbscan_kwargs_dict = parse_kwargs_string(hdbscan_kwargs) if hdbscan_kwargs else {}

    # Perform HDBSCAN
    hdbscan = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        algorithm=algorithm,
        **hdbscan_kwargs_dict,
    )
    clusters = hdbscan.fit(data)

    # HDBSCAN output
    cluster_labels = clusters.labels_
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)

    # Write HDBSCAN cluster labels to a file
    df = pd.DataFrame(cluster_labels.astype(str), columns=["cluster"])
    df.to_csv(global_opts["output"], index=False)

    # Make scatterplot(s)
    labels_exist = False
    if global_opts["labels"]:
        labels = read_labels(global_opts["labels"])
        df["label"] = labels["label"]
        labels_exist = True

    if global_opts["plot"]:
        vars_to_plot = [var.strip() for var in global_opts["plot"].split(",")]
        df[vars_to_plot[0]] = data_df[vars_to_plot[0]]
        df[vars_to_plot[1]] = data_df[vars_to_plot[1]]

        hdbscan_plot = (
                plot_data(df, vars_to_plot, labels_exist)
                + labs(
                    title="""
                    HDBSCAN Clustering
                    Number of clusters: {} | Number of noise samples: {}
                    """.format(n_clusters, n_noise)
                )
        )
        hdbscan_plot.save("HDBSCAN_plot.pdf")

    if global_opts["pca_plot"]:
        var_explained = pca.explained_variance_ratio_ * 100
        df["PC1"] = pca_data[:, 0]
        df["PC2"] = pca_data[:, 1]

        hdbscan_pca_plot = (
                plot_data(df, ["PC1", "PC2"], labels_exist)
                + labs(
                    title="""
                    HDBSCAN Clustering
                    Number of clusters: {} | Number of noise samples: {}
                    """.format(n_clusters, n_noise),
                    x=f"PC1 ({var_explained[0]:.2f}%)",
                    y=f"PC2 ({var_explained[1]:.2f}%)"
                )
        )
        hdbscan_pca_plot.save("HDBSCAN_pca_plot.pdf")


# Command for optics
@cluster.command()
def optics(
    max_eps: Annotated[
        float,
        typer.Option(
            min=0,
            help="""
            The maximum distance between two samples for one to be considered
            as in the neighborhood of the other. If None, OPTICS will identify
            clusters across all scales; reducing max_eps will result in shorter
            run times.
            """,
        ),
    ] = None,
    min_samples: Annotated[
        int,
        typer.Option(
            min=2,
            help="""
            The number of samples in a neighborhood for a point to be
            considered as a core point.
            """,
        ),
    ] = 5,
    min_cluster_size: Annotated[
        int,
        typer.Option(
            min=2,
            help="""
            Minimum number of samples in an OPTICS cluster.
            If None, the value of min_samples is used instead.
            """,
        ),
    ] = None,
    xi: Annotated[
        float,
        typer.Option(
            min=0,
            max=1,
            help="""
            Determines the minimum steepness on the reachability plot that
            constitutes a cluster boundary. For example, an upwards point in
            the reachability plot is defined by the ratio from one point to its
            successor being at most 1-xi.
            """,
        ),
    ] = 0.05,
    algorithm: Annotated[
        str,
        typer.Option(
            help="""
            Algorithm used to compute the nearest neighbors.
            It can take the following values: auto, ball_tree, kd_tree, brute.
            """,
        ),
    ] = "auto",
    optics_kwargs: Annotated[
        str,
        typer.Option(
            help="String with arbitrary extra parameters for OPTICS."
        ),
    ] = "",
):
    """
    Perform OPTICS clustering.
    Ordering Points To Identify the Clustering Structure

    Examples:

    clustering.py --input-file input.csv --scale --pca-plot optics
    --min-samples 10 --optics-kwargs 'metric="cosine",n_jobs=2'
    """
    from sklearn.cluster import OPTICS

    # Read data
    data_df, data, pca_data, pca = preprocess(
        global_opts["input"],
        global_opts["scale"],
        do_pca=global_opts["pca_before"],
        var_expl=global_opts["var_explained"],
    )

    # max_eps
    max_eps = max_eps if max_eps else np.inf

    # Parse OPTICS kwargs string to dictionary
    optics_kwargs_dict = parse_kwargs_string(optics_kwargs) if optics_kwargs else {}

    # Perform OPTICS
    optics = OPTICS(
        max_eps=max_eps,
        min_samples=min_samples,
        min_cluster_size=min_cluster_size,
        xi=xi,
        algorithm=algorithm,
        **optics_kwargs_dict,
    )
    clusters = optics.fit(data)

    # OPTICS output
    cluster_labels = clusters.labels_
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)

    # Write OPTICS cluster labels to a file
    df = pd.DataFrame(cluster_labels.astype(str), columns=["cluster"])
    df.to_csv(global_opts["output"], index=False)

    # Make scatterplot(s)
    labels_exist = False
    if global_opts["labels"]:
        labels = read_labels(global_opts["labels"])
        df["label"] = labels["label"]
        labels_exist = True

    if global_opts["plot"]:
        vars_to_plot = [var.strip() for var in global_opts["plot"].split(",")]
        df[vars_to_plot[0]] = data_df[vars_to_plot[0]]
        df[vars_to_plot[1]] = data_df[vars_to_plot[1]]

        optics_plot = (
                plot_data(df, vars_to_plot, labels_exist)
                + labs(
                    title="""
                    OPTICS Clustering
                    Number of clusters: {} | Number of noise samples: {}
                    """.format(n_clusters, n_noise)
                )
        )
        optics_plot.save("OPTICS_plot.pdf")

    if global_opts["pca_plot"]:
        var_explained = pca.explained_variance_ratio_ * 100
        df["PC1"] = pca_data[:, 0]
        df["PC2"] = pca_data[:, 1]

        optics_pca_plot = (
                plot_data(df, ["PC1", "PC2"], labels_exist)
                + labs(
                    title="""
                    OPTICS Clustering
                    Number of clusters: {} | Number of noise samples: {}
                    """.format(n_clusters, n_noise),
                    x=f"PC1 ({var_explained[0]:.2f}%)",
                    y=f"PC2 ({var_explained[1]:.2f}%)"
                )
        )
        optics_pca_plot.save("OPTICS_pca_plot.pdf")


if __name__ == "__main__":
    cluster()
