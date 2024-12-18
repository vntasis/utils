#!/usr/bin/env python

import ast
import os
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error, silhouette_score
from typing_extensions import Annotated

dnmf = typer.Typer(no_args_is_help=True)


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


# Function for parsing grid parameters
def parse_param_grid(param_grid_str: dict) -> dict:
    """
    Parse comma-separated strings in
    the param_grid into lists of integers or floats.

    Args:
        param_grid_str (dict): Dictionary with
        hyperparameter names and comma-separated strings.

    Returns:
        dict: Dictionary with hyperparameter names and
              lists of integers or floats.
    """
    parsed_grid = {}
    for key, value_str in param_grid_str.items():
        value_list = ast.literal_eval(f"[{value_str}]")
        parsed_grid[key] = value_list
    return parsed_grid


# Function for calculating reconstruction error
def reconstruction_error(A: np.ndarray, W: np.ndarray, H: np.ndarray) -> float:
    """
    Compute the reconstruction error using mean squared error.

    Args:
        A (np.ndarray): Original data matrix.
        W (np.ndarray): Matrix of data points in latent space.
        H (np.ndarray): Latent components matrix.

    Returns:
        float: Mean squared error between the original
               and reconstructed matrix.
    """
    return mean_squared_error(A, np.dot(W, H))


# Function for multiple NMF runs with random initialization
def nmf_multiple_runs(
    A: np.ndarray, n_components: int, n_runs: int = 10, **kwargs
) -> tuple:
    """
    Perform NMF multiple times with random initializations and compute
    reconstruction error and silhouette score.
    Returns the best model based on reconstruction error.

    Args:
        A (np.ndarray): Original data matrix.
        n_components (int): Number of latent factors.
        n_runs (int): Number of runs to perform.
        **kwargs: Additional keyword arguments passed to NMF.

    Returns:
        tuple: (Best reconstruction error, Average silhouette score, Best model
               (tuple of (W, H, nmf)))
    """
    best_error = float("inf")
    silhouette_scores = []
    best_model = None

    for run in range(n_runs):
        nmf = NMF(
            n_components=n_components,
            init="random",
            random_state=run,
            **kwargs
        )
        W = nmf.fit_transform(A)
        H = nmf.components_

        # Compute reconstruction error
        error = reconstruction_error(A, W, H)

        # Compute Silhouette score for clustering features (transpose H)
        H_T = H.T  # Transpose H so rows are features
        try:
            hdbscan = HDBSCAN(
                min_cluster_size=20, algorithm="auto"
            ).fit(H_T)
            silhouette = silhouette_score(H_T, hdbscan.labels_)
        except Exception as e:
            print(f"Error calculating silhouette score: {e}")
            silhouette = np.nan
        silhouette_scores.append(silhouette)

        # Track the best model based on reconstruction error
        if error < best_error:
            best_error = error
            best_model = (W, H, nmf)

        # Return best reconstruction error, mean silhouette score
        # (excluding NA values), and best model
        valid_silhouettes = [s for s in silhouette_scores if not np.isnan(s)]
        avg_silhouette = np.mean(valid_silhouettes) if valid_silhouettes \
            else np.nan

    return best_error, avg_silhouette, best_model


# Use callback for global options and documentation
@dnmf.callback()
def main():
    """
    Dynamic NMF analysis
    """


@dnmf.command()
def tune(
    input: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            writable=False,
            readable=True,
            resolve_path=True,
            help="Path to the input data file. It has to be a csv file with"
            + " a header (column names).",
        ),
    ],
    n_comp_grid_values: Annotated[
        str,
        typer.Option(
            help="""
            Character string of comma-separated values
            for the number of components to test.
            """
        ),
    ],
    alpha_grid_values: Annotated[
        str,
        typer.Option(
            help="""
            Character string of comma-separated values
            for the alpha constants to test.
            """
        ),
    ],
    l1_ratio_grid_values: Annotated[
        str,
        typer.Option(
            help="""
            Character string of comma-separated values
            for the l1 ratio to test.
            """
        ),
    ],
    n_runs: Annotated[
        int,
        typer.Option(
            min=1,
            help="Times to run NMF with random initialization.",
        ),
    ] = 10,
    max_iter: Annotated[
        int,
        typer.Option(help="Maximum number of iterations for each NMF run."),
    ] = 200,
):
    """
    Perform grid search for NMF hyperparameter tuning.
    """

    from plotnine import aes, geom_line, geom_point, ggplot, labs, theme_bw
    from sklearn.model_selection import ParameterGrid

    # Read input
    data_df = pd.read_csv(input)

    # Checks if any value in the DataFrame is negative
    if (data_df < 0).any().any():
        raise ValueError(
            "Error: The input matrix contains negative values."
            + " NMF requires non-negative data."
        )

    # Convert input df to an array
    data = data_df.values

    # Grid of hyperparameter values
    param_grid_str = {
        "n_components": n_comp_grid_values,
        "alpha": alpha_grid_values,
        "l1_ratio": l1_ratio_grid_values
    }

    param_grid = parse_param_grid(param_grid_str)

    # Iterate over all combinations of hyperparameters
    eval_results = []

    for params in ParameterGrid(param_grid):
        n_components = params["n_components"]
        alpha = params["alpha"]
        l1_ratio = params["l1_ratio"]

        # Print the current combination of hyperparameters
        print(
            f"Testing hyperparameters: n_components={n_components}, "
            f"alpha={alpha}, l1_ratio={l1_ratio}"
        )

        # Run NMF multiple times and evaluate the model
        error, silhouette_avg, _ = nmf_multiple_runs(
            data, n_components, n_runs,
            alpha_W=alpha, alpha_H="same",
            l1_ratio=l1_ratio, max_iter=max_iter
        )

        eval_results.append({
            "n_components": n_components,
            "alpha": alpha,
            "l1_ratio": l1_ratio,
            "reconstruction_error": error,
            "silhouette_score": silhouette_avg
        })

        # Print error and silhouette score
        print(
            f"Reconstruction Error: {error}, "
            f"Silhouette Score: {silhouette_avg}"
        )

    # Convert results to DataFrame
    eval_results_df = pd.DataFrame(eval_results)

    # Save results to a TSV file
    eval_results_df.to_csv("nmf_tuning_results.tsv", sep="\t", index=False)

    # Plot Reconstruction Error
    plot_error = (
        ggplot(
            eval_results_df,
            aes(
                x="n_components",
                y="reconstruction_error",
                color="factor(l1_ratio)",
                shape="factor(alpha)"
            )
        )
        + geom_line()
        + geom_point()
        + labs(
            title="Reconstruction Error",
            y="Reconstruction Error",
            x="n_components"
        )
        + theme_bw()
    )
    plot_error.save("reconstruction_error.pdf")

    # Plot Silhouette Score
    plot_silhouette = (
        ggplot(
            eval_results_df,
            aes(
                x="n_components",
                y="silhouette_score",
                color="factor(l1_ratio)",
                shape="factor(alpha)"
            )
        )
        + geom_line()
        + geom_point()
        + labs(
            title="Silhouette Score",
            y="Silhouette Score",
            x="n_components"
        )
        + theme_bw()
    )
    plot_silhouette.save("silhouette_score.pdf")


@dnmf.command()
def fit(
    input: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            writable=False,
            readable=True,
            resolve_path=True,
            help="Path to the input data file. It has to be a csv file with"
            + " a header (column names).",
        ),
    ],
    n_components: Annotated[
        int,
        typer.Option(
            min=1,
            help="""
            Number of components. If it is not set all features are kept.
            """,
        ),
    ] = None,
    init: Annotated[
        str,
        typer.Option(
            help="""
            Method for initialization.
            It can take the following values: random, nndsvd, nndsvda,
            nndsvdar.
            """
        ),
    ] = None,
    max_iter: Annotated[
        int,
        typer.Option(help="Maximum number of iterations before timing out."),
    ] = 200,
    n_runs: Annotated[
        int,
        typer.Option(
            min=1,
            help="Times to run NMF when random initialization is chosen.",
        ),
    ] = 10,
    verbose: Annotated[
        int,
        typer.Option(help="Whether to be verbose."),
    ] = 0,
    nmf_kwargs: Annotated[
        str,
        typer.Option(help="String with arbitrary extra parameters for NMF."),
    ] = "",
):
    """
    Run a NMF model.
    """

    # Read input
    data_df = pd.read_csv(input)

    # Checks if any value in the DataFrame is negative
    if (data_df < 0).any().any():
        raise ValueError(
            "Error: The input matrix contains negative values."
            + " NMF requires non-negative data."
        )

    # Convert input df to an array
    data = data_df.values

    # Parse NMF kwargs string to dictionary
    nmf_kwargs_dict = parse_kwargs_string(nmf_kwargs) if nmf_kwargs else {}

    # NMF model
    if init == "random":
        error, silhouette_avg, (W, H, model) = nmf_multiple_runs(
            data, n_components, n_runs,
            max_iter=max_iter, **nmf_kwargs_dict
        )
    else:
        model = NMF(
            n_components=n_components,
            init=init,
            max_iter=max_iter,
            verbose=verbose,
            **nmf_kwargs_dict,
        )

        W = model.fit_transform(data)
        H = model.components_
        error = reconstruction_error(data, W, H)

    # Print error and silhouette score
    print(f"Reconstruction Error: {error}")
    if init == "random":
        print(f"Silhouette Score: {silhouette_avg}")

    # Save output
    W_df = pd.DataFrame(
        W, columns=[f"Factor_{i}" for i in range(n_components)]
    )
    W_df.to_csv("W_matrix.csv", index=False)

    H_df = pd.DataFrame(H, columns=data_df.columns)
    H_df.to_csv("H_matrix.csv", index=False)


@dnmf.command()
def merge(
    input: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            dir_okay=True,
            writable=False,
            readable=True,
            resolve_path=True,
            help="Path to the directory with the H matrices (csv files).",
        ),
    ],
    keep: Annotated[
        int,
        typer.Option(
            min=1,
            help="""
            The t top-ranked features to keep
            from every row vector of each H matrix.
            """,
        ),
    ] = 10,
):
    """
    Merge H matrices from different timepoints.
    """
    from kneed import KneeLocator
    from plotnine import (
        aes, annotate, geom_line, geom_vline, ggplot, labs, theme_bw
    )

    # List to store processed DataFrames
    processed_dfs = []

    # List to store ranked data for plotting
    ranked_data = []

    # Iterate over all files in the directory
    for file_name in os.listdir(input):
        if file_name.endswith(".csv"):
            file_path = os.path.join(input, file_name)
            df = pd.read_csv(file_path)

            # Separate the header
            header = df.columns.tolist()

            # Remove the ".csv" extension from the file name
            source_name = file_name.replace(".csv", "")

            # Process each row: rank values and keep top t features
            processed_matrix = []
            for _, row in df.iterrows():
                # Rank values from largest to smallest (for plotting)
                ranked_row = row.sort_values(ascending=False).values
                for rank, value in enumerate(ranked_row, start=1):
                    ranked_data.append({"Rank": rank, "Value": value})

                # Keep only the top t features (for processed matrix)
                top_indices = row.nlargest(keep).index
                new_row = [row[idx] if idx in top_indices else 0 for idx in header]
                processed_matrix.append(new_row)

            # Convert the list to a DataFrame with the same header
            processed_df = pd.DataFrame(processed_matrix, columns=header)

            # Add columns for source file and original row index
            processed_df["Source"] = source_name
            processed_df["Original_Factor"] = df.index

            # Append to the list of processed DataFrames
            processed_dfs.append(processed_df)

    # Concatenate all DataFrames to create the final stacked DataFrame
    final_df = pd.concat(processed_dfs, ignore_index=True)

    # Create a DataFrame for metadata
    metadata_df = final_df[["Source", "Original_Factor"]].copy()

    # Remove metadata columns from the final DataFrame
    final_df = final_df.drop(columns=["Source", "Original_Factor"])

    # Remove columns filled with 0s only
    final_df = final_df.loc[:, (final_df != 0).any(axis=0)]

    # Save the final DataFrames to CSV files
    final_df.to_csv("B_matrix.csv", index=False)
    metadata_df.to_csv("B_metadata.csv", index=False)

    # Convert the ranked data list to a DataFrame
    ranked_df = pd.DataFrame(ranked_data)

    # Calculate medians for each rank and
    medians = ranked_df.groupby("Rank")["Value"].median().reset_index()

    # Find the positions of the vertical lines based on median values
    vline_1 = medians[medians["Value"] >= 1]["Rank"].max()
    vline_2 = medians[medians["Value"] >= 0.5]["Rank"].max()
    vline_3 = medians[medians["Value"] >= 0.1]["Rank"].max()

    # Find the elbow point
    kneedle = KneeLocator(
        medians['Rank'],
        medians['Value'],
        curve='convex',
        direction='decreasing'
    )
    elbow_rank = kneedle.elbow
    elbow_value = medians.loc[medians['Rank'] == elbow_rank, 'Value'].values[0]

    # Plot of ranked values
    x_range = medians["Rank"].max() - medians["Rank"].min()
    offset = 0.01 * x_range

    plot = (
        ggplot(medians, aes(x="Rank", y="Value"))
        + geom_line()
        + labs(
            title="Ranked Values in H matrices",
            x="Ranked Position",
            y="Median Weight"
        )
        + theme_bw()
        + geom_vline(xintercept=vline_1, linetype="dashed", color="red")
        + geom_vline(xintercept=vline_2, linetype="dashed", color="blue")
        + geom_vline(xintercept=vline_3, linetype="dashed", color="green")
        + geom_vline(xintercept=elbow_rank, linetype="dashed", color="purple")
        + annotate(
            "text", x=vline_1 + offset, y=2,
            label=f"({vline_1}, 1.0)", color="red",
            ha="left"
        )
        + annotate(
            "text", x=vline_2 + offset, y=2,
            label=f"({vline_2}, 0.5)", color="blue",
            ha="left"
        )
        + annotate(
            "text", x=vline_3 + offset, y=2,
            label=f"({vline_3}, 0.1)", color="green",
            ha="left"
        )
        + annotate(
            "text", x=elbow_rank + offset, y=1.5,
            label=f"Elbow point at: ({elbow_rank}, {elbow_value:.1f})",
            color="purple",
            ha="left"
        )
    )
    plot.save("Weights_H_matrices.pdf")


@dnmf.command()
def summarize(
    b_metadata: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            writable=False,
            readable=True,
            resolve_path=True,
            help="""
            File containing the metadata of matrix B. It should be a csv file.
            """,
        ),
    ],
    w_matrix: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            writable=False,
            readable=True,
            resolve_path=True,
            help="Final W matrix produced by Dynamic NMF.",
        ),
    ],
    h_directory: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            dir_okay=True,
            writable=False,
            readable=True,
            resolve_path=True,
            help="Path to the directory with the time specific H matrices (csv files).",
        ),
    ],
    keep: Annotated[
        int,
        typer.Option(
            min=1,
            help="""
            The t top-ranked features to keep
            from every row vector of each H matrix.
            """,
        ),
    ] = 10,
):
    """
    Summarize the output of Dynamic NMF in one file.
    """

    # Read the first CSV file which has two columns
    b_meta = pd.read_csv(b_metadata)

    # Read the second CSV file with multiple columns
    w_mat = pd.read_csv(w_matrix)

    # Check if the number of rows match between b_meta and w_mat
    if len(b_meta) != len(w_mat):
        raise ValueError(
            "The two CSV files must have the same number of rows."
        )

    output_data = []

    # Process each row
    for index, row in b_meta.iterrows():
        first_column_value = row.iloc[0]
        second_column_value = row.iloc[1]

        # Get the corresponding row from the second CSV file
        w_mat_row = w_mat.iloc[index]

        # Find the column name with the maximum value in that row
        max_value_column = w_mat_row.idxmax()

        # Read the corresponding CSV file from the directory
        h_mat_path = os.path.join(h_directory, first_column_value + ".csv")

        if not os.path.exists(h_mat_path):
            raise FileNotFoundError(f"The file {h_mat_path} does not exist.")

        h_mat = pd.read_csv(h_mat_path)

        # Ensure the second column is within the row limits of the file
        if second_column_value >= len(h_mat):
            raise ValueError(
                f"Row number {second_column_value} exceeds the row count of the file {h_mat_path}"
            )

        # Get the specific row from the file
        row_to_read = h_mat.iloc[second_column_value]

        # Find the top k values and their corresponding column names
        t_top_columns = row_to_read.nlargest(keep).index.tolist()
        t_top_columns_str = ",".join(t_top_columns)

        # Append the result for this row
        output_data.append([
            first_column_value,
            second_column_value,
            max_value_column,
            t_top_columns_str
        ])

    # Create a DataFrame for the output and save it as a TSV file
    summary_df = pd.DataFrame(
        output_data,
        columns=[
            "Source",
            "Time_Specific_Factor",
            "Dynamic_Factor",
            "Top_Features"
        ]
    )
    summary_df.to_csv("dnmf_summary.tsv", sep="\t", index=False)


if __name__ == "__main__":
    dnmf()
