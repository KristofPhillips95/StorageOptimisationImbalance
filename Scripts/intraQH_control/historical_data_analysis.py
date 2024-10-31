

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

min_in_qh = 15

data = pd.read_csv('SI_data_minute_2023.csv',sep=';')

print(data.dtypes)

data = data[['Datetime','System imbalance']]

data['Datetime'] = pd.to_datetime(data['Datetime'], utc=True)
data.set_index('Datetime', inplace=True)

dict_list_arrays = {}
for i in range(min_in_qh):
    dict_list_arrays[i] = []

x=1

qh_count = 0
qh_limit = 50000

for quarter_hour, group in data.groupby(pd.Grouper(freq='15min')):

    if qh_count >= qh_limit:
        break

    if group.empty:
        continue

    minute_data = group.resample('1min').sum().fillna(0)
    check = 0

    # Resample within the quarter-hour by minute and calculate the cumulative sum of SI
    minute_data = group.resample('1min').sum().fillna(0)
    minute_data['Cumulative_SI'] = minute_data['System imbalance'].cumsum()
    minute_data['Cumulative_Count'] = np.arange(1, len(minute_data) + 1)

    # Cumulative average of SI
    minute_data['Cumulative_Avg_SI'] = minute_data['Cumulative_SI'] / minute_data['Cumulative_Count']

    # Total SI for the quarter-hour
    total_si = minute_data['System imbalance'].sum()/min_in_qh

    # Create the numpy array for the current quarter-hour
    # Column 1: Cumulative SI up to that minute
    # Column 2: Total SI for the quarter-hour
    quarter_hour_array = np.column_stack((
        minute_data['Cumulative_Avg_SI'].values,
        np.full(minute_data.shape[0], total_si)
    ))

    for i in range(min_in_qh):
        dict_list_arrays[i].append(quarter_hour_array[i,:])

    x=1

    qh_count += 1


dict_arrays = {}

for i in range(min_in_qh):
    dict_arrays[i] = np.vstack(dict_list_arrays[i])


def copy_nonzero(source, target):
    """
    Copy elements from `source` array to `target` array unless the corresponding
    element in `target` is zero.

    Parameters:
    source (numpy.ndarray): Source array to copy values from.
    target (numpy.ndarray): Target array to update, unless the value is zero.

    Returns:
    numpy.ndarray: Updated target array.
    """

    # Ensure both arrays have the same shape
    assert source.shape == target.shape, "Source and target arrays must have the same shape."

    # Use np.nditer() to iterate through the arrays efficiently
    with np.nditer([source, target], flags=['multi_index'], op_flags=[['readonly'], ['readwrite']]) as it:
        for src_val, tgt_val in it:
            # Only copy the source element if the target value is zero
            if src_val != 0:
                tgt_val[...] = src_val  # Use ellipsis to modify the value at the current index

    return target

def bin_and_count(array, min_val, max_val, num_bins, threshold):
    """
    Create a distribution by binning the first column using specified bin ranges and counting how many
    times the second column exceeds a threshold in each bin.

    Parameters:
    array (numpy.ndarray): 2D array where rows correspond to observations.
    min_val (float): Minimum value for the bins (for the first column).
    max_val (float): Maximum value for the bins (for the first column).
    num_bins (int): Number of bins to divide the range [min_val, max_val] into.
    threshold (float): The threshold for counting values in the second column.

    Returns:
    numpy.ndarray: Array of counts where the second column exceeds the threshold, per bin.
    """
    # Create the bins using linspace based on min_val, max_val, and num_bins
    bins = np.linspace(min_val, max_val, num_bins + 1)

    # Extract the first and second columns from the array
    first_col = array[:, 0]
    second_col = array[:, 1]

    # Digitize the first column values into bins
    bin_indices = np.digitize(first_col, bins)

    # Initialize a result array to store the counts for each bin
    counts = np.zeros(num_bins, dtype=int)
    counts_th = np.zeros(num_bins, dtype=int)


    # Loop through the bins and count where the second column exceeds the threshold
    for i in range(1, num_bins + 1):
        # Select the rows where the first column falls into the i-th bin
        in_bin = (bin_indices == i)

        # Count the number of times the second column exceeds the threshold in this bin
        counts[i-1] = np.sum(in_bin)
        counts_th[i - 1] = np.sum(second_col[in_bin] > threshold)

    # Replace zeros with ones using boolean indexing
    counts_adjusted = np.ones_like(counts)
    counts_adjusted = copy_nonzero(counts,counts_adjusted)

    prob = counts_th/counts_adjusted

    return counts,prob,bins


def plot_probability(prob, bins):
    """
    Plot the probability (exceed_counts / bin_counts) as a bar chart.

    Parameters:
    exceed_counts (numpy.ndarray): Array of counts where the second column exceeds the threshold, per bin.
    bin_counts (numpy.ndarray): Array of counts where the first column falls within each bin.
    bins (numpy.ndarray): The bin edges.
    """
    # Calculate bin centers for plotting
    bin_centers = (bins[:-1] + bins[1:]) / 2


    # Plot the probabilities as a bar chart
    plt.bar(bin_centers, prob, width=(bins[1] - bins[0]), color='blue', alpha=0.7)

    # Add labels and title
    plt.xlabel('Bins (First Column Values)')
    plt.ylabel('Probability (Exceedance / Total)')
    plt.title('Probability of Exceedance by Bin')

    # Show the plot
    plt.show()


def plot_probabilities_subplots(prob_dict, integer_array, bins):
    """
    Plot subplots with bar charts of probabilities for different integer arrays.

    Parameters:
    prob_dict (dict): Dictionary where keys are integers, and values are probabilities.
    integer_arrays (list of lists): List of arrays, each representing a set of integers to show on the plot.
    bins (numpy.ndarray): The bin edges for proper alignment of the x-axis.
    """
    # Number of subplots is based on the length of integer_arrays
    num_subplots = len(integer_array)

    # Create subplots
    fig, axes = plt.subplots(num_subplots, 1, figsize=(8, 4 * num_subplots))  # Adjust figure size as needed

    if num_subplots == 1:
        axes = [axes]  # If there is only one subplot, wrap it in a list for consistency

    # Calculate bin centers for plotting
    bin_centers = (bins[:-1] + bins[1:]) / 2

    for i, val in enumerate(integer_array):
        # Extract the probabilities from the dictionary for the specified integers
        probabilities = prob_dict[val]

        # Plot the probabilities as a bar chart in the i-th subplot
        axes[i].bar(bin_centers, probabilities, width=(bins[1] - bins[0]), color='blue', alpha=0.7)
        axes[i].set_xlabel('Bins (Integer Values)')
        axes[i].set_ylabel('Probability')
        axes[i].set_title(f'Probability Distribution for SI measurement after minute {val + 1}')

    # Automatically adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()

dict_probabilities = {}
bin_ll = -500
bin_ul = 500
n_bins = 50

for i in range(min_in_qh):
    _,prob,bins = bin_and_count(dict_arrays[i],bin_ll,bin_ul, n_bins, 0)
    dict_probabilities[i] = prob

plot_probabilities_subplots(dict_probabilities,[0,6,14],bins)
plot_probability(dict_probabilities[14],bins)

x=1

