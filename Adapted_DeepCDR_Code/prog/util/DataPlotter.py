""" Collection of methods to plot different statistics"""
import os.path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
# import statsmodels


def sort_by_iqr(df, group_attribute, sort_attribute):
    """
    Sorts a dataframe by their ascending interquartile ranges.
    Args:
        df: dataframe to be sorted
        group_attribute: attribute by which dataframe should be grouped
        sort_attribute: attribute on which interquartile range of (grouped) dataframe should be computed

    Returns:
        grouped dataframe sorted by intequartile range of defined sort attribute.

    """
    unique = df[group_attribute].unique()
    gt_data = []
    pred_data = []
    diff_data = []
    for item in unique:
        gt_values = df[df[group_attribute] == item]["gt"].tolist()
        pred_values = df[df[group_attribute] == item]["pred"].tolist()
        diff_values = df[df[group_attribute] == item]["diff"].tolist()
        gt_data.append(gt_values)
        pred_data.append(pred_values)
        diff_data.append(diff_values)

    grouped_df = pd.DataFrame(list(zip(unique, gt_data, pred_data, diff_data)),
                              columns=[group_attribute, "gt", "pred", "diff"])
    iqr_values = [np.percentile(d, 75) - np.percentile(d, 25) for d in grouped_df[sort_attribute]]
    grouped_df["iqr"] = iqr_values
    grouped_df.sort_values("iqr", inplace=True)
    return grouped_df

def plotStatistic(history, foldIdx, leaveOut, task, debug_mode):
    """
    Plots validation and trainings loss and saves generated plot.
    Args:
        history:
        foldIdx: index of current fold
        leaveOut: trainings mode, e.g. cell out or drug out
        task: which task model performs (classification or regression)
        debug_mode: True if model gets debugged on smaller subset, otherwise false
    """
    plt.plot(history.history["loss"], label="train loss")
    plt.plot(history.history["val_loss"], label="validation loss")
    plt.legend
    date_time = datetime.now().strftime("%d.%m.%Y-%H:%M:%S")
    if debug_mode:
        plt.savefig(f"../plots/{task}/debug/{leaveOut}/fold_{foldIdx}_{date_time}.png")
    else:
        plt.savefig(f"../plots/{task}/no_debug/{leaveOut}/fold_{foldIdx}_{date_time}.png")
    plt.close()


def sort_by_var(df, group_attribute="drug", sort_attribute="gt"):
    """
    Sorts grouped dataframe by ascending variance of sort attribute.
    Args:
        df: dataframe to be sorted
        group_attribute: attribute by which dataframe should be sorted
        sort_attribute: attribute to calculate variance for. (gt: ground truth, pred: prediction)
    Returns:
        dataframe sorted by variance of the ground truth grouped by drug.
    """
    var = df.groupby(group_attribute)[sort_attribute].var().reset_index()
    var.columns = [group_attribute, "var"]
    sorted_groups = var.sort_values(by="var", ascending=False)
    sorted_groups.reset_index(drop=True, inplace=True)
    return sorted_groups

def plot_var(path):
    """
    Sorts, plots and saves the plot of a dataframe grouped by drug according to ascending variance in ground truth, prediction and the difference between ground truth and prediction.
    Args:
        path: path to the dataframe
    """
    df = pd.read_csv(path)
    var_df = df.groupby('drug').agg({"gt": "var", "pred": "var"}).reset_index()
    # Group by "Drug" and calculate variance for each group

    var_df.columns = ["drug", "var_gt", "var_pred"]
    var_df["diff"] = abs(var_df["var_gt"] - var_df["var_pred"])

    var_df["var_gt"] = var_df["var_gt"].round(4)
    var_df["var_pred"] = var_df["var_pred"].round(4)
    var_df["diff"] = var_df["diff"].round(4)

    sort_by_gt = var_df.sort_values(by="var_gt")
    sort_by_pred = var_df.sort_values(by="var_pred")
    sort_by_diff = var_df.sort_values(by="diff")

    dfs = {"var_gt": sort_by_gt, "diff": sort_by_diff, "var_pred": sort_by_pred}

    for name, df in dfs.items():
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(6, 50))
        colormap = plt.cm.get_cmap("Spectral")  # Choose a colormap

        prelim_cols = colormap(df.iloc[:, 1:])

        colors = np.zeros((223, 4, 4))

        # Copy the values from the original array to the sort_by_pred array
        colors[:, 1:, :] = prelim_cols
        table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center", cellColours=colors)

        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.2)

        # Remove axis
        ax.axis("off")

        plt.savefig(f'../plots/var_by_drug/{name}.png')

        plt.show()
        plt.close()

    sort_by_gt.to_csv(f"../data/var_per_drug_sort_by_gt.csv", index=False)
    sort_by_pred.to_csv(f"../data/var_per_drug_sort_by_pred.csv", index=False)
    sort_by_diff.to_csv(f"../data/var_per_drug_sort_by_diff.csv", index=False)


def plotIC50DotPlot(df, group_attribute):
    """
    Plots predicted vs actual IC 50 values grouped by an attribute of a dataframe and saves the generated plot.
    One plot contains the results of five different group_attributes.

    Args:
        df: dataframe with results
        group_attribute: attribute to group the results by
    """
    step = 5

    sorted_df = sort_by_var(df)
    df["rank"] = [sorted_df.index[sorted_df[group_attribute] == row[group_attribute]].tolist()[0] for index, row in
                  df.iterrows()]
    df.sort_values("rank", inplace=True)
    df.reset_index(drop=True, inplace=True)
    n_drugs = df[group_attribute].nunique()
    for i in range(0, n_drugs, step):
        stop_index = min(i + step - 1, n_drugs)
        current_frame = df[df["rank"].between(i, stop_index)]
        draw_single_dot_plot(current_frame, group_attribute, i)


def draw_single_dot_plot(current_frame, group_attribute, i):
    """
    Draws and saves a single plot of predicted vs actual IC 50 values grouped by an attribute.

    Args:
        current_frame: dataframe of all datapoints to be plotted in the single plot
        group_attribute: attribute to goup dataframe
        i: integer to identify a particular plot if multiple plots are drawn
    """
    sns.lmplot(data=current_frame, x="gt", y="pred", hue='drug', palette=sns.color_palette(), ci=None, robust=True,
               truncate=True, sharex=True, sharey=True)
    plt.xlabel('Actual IC$_{50}$')
    plt.ylabel('Predicted IC$_{50}$')
    plt.title('Predicted vs Actual IC$_{50}$ per %s' % group_attribute)
    # plt.xlim(0, 8)
    # plt.ylim(0, 8)
    x_values = np.linspace(min(min(current_frame['gt']), min(current_frame['pred'])),
                           max(max(current_frame['gt']), max(current_frame['pred'])), 100)  # Adjust the range and number of points as needed
    plt.plot(x_values, x_values, color='black', linestyle='-')
    plt.tight_layout()
    if group_attribute == "drug":
        directory = "performance_per_Drug"
    elif group_attribute == "tissue":
        directory = "performance_per_tissue"
    else:
        directory = "performance_per_cellline"
    if not os.path.isdir(f"../plots/{directory}/dot/sort_by_gt/"):
        os.makedirs(f"../plots/{directory}/dot/sort_by_gt/")
    plt.savefig(f'../plots/{directory}/dot/sort_by_gt/fold_{i}.png')
    plt.show()
    plt.close()


def plotIC50BoxPlot(df, group_attribute, sort_attribute):
    """
    Plots predicted IC50 values as boxplots grouped and sorted by indicated attributes.
    One plot contains the data points of five group attributes

    Args:
        df: dataframe containing predicted IC50 values
        group_attribute: attribute to group dataframe
        sort_attribute: attribute how the group attributes should be sorted for plotting
    """
    sorted_df = sort_by_iqr(df, group_attribute=group_attribute, sort_attribute=sort_attribute)

    df["rank"] = [sorted_df.index[sorted_df[group_attribute] == row[group_attribute]].tolist()[0] for index, row in
                  df.iterrows()]
    df.sort_values("rank", inplace=True)
    df.reset_index(drop=True, inplace=True)

    next_index = 0
    for i in range(5, len(sorted_df), 5):
        last_index = next_index
        next_index = df.index[df["rank"] == i].tolist()[0]
        current_frame = df[last_index:next_index]
        dd = pd.melt(current_frame, id_vars=[group_attribute], value_vars=['gt', 'pred'], var_name="values")

        sns.boxplot(x=group_attribute, y='value', data=dd, hue='values')

        plt.xlabel(group_attribute)
        plt.ylabel('IC50')
        plt.title(f'Predicted IC50 per {group_attribute}')

        if group_attribute == "drug":
            directory = "performance_per_Drug"
        elif group_attribute == "tissue":
            directory = "performance_per_tissue"
        else:
            directory = "performance_per_cellline"
        plt.savefig(f'../plots/{directory}/comparison/fold_{i - 5}.png')
        plt.show()
        plt.close()
