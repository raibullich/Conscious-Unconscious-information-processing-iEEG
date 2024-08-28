# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import mne
import scipy
from nilearn import plotting


mne.set_log_level('WARNING') # to not have many 'debug' or warning 'messages'



# Functions to AVERAGE REFERENCE THE DATA

def create_probe_channels_list_of_lists(list_channels):

    ''' Creates a lists of lists. Each sub-list belongs to one probe, so all channels in that list will share the same letter (and ' sign).
        Input a list of all channel names (e.g., A2, A4, A6, B2, B5...)
        Returns a list of lists.'''
    
    # Initialize dictionaries to store the classified elements
    classified_elements = {}

    # Iterate through the list and classify the elements
    for item in list_channels:
        letter = item[0]  # Get the first character (letter)
        if "'" in item:  # Check if the item has a prime
            classified_elements.setdefault((letter, True), []).append(item) # True means has a prime
        else:
            classified_elements.setdefault((letter, False), []).append(item)
    # Convert dictionaries to lists
    result_lists = list(classified_elements.values())
    return result_lists

def average_reference_data(raw_data, channels_list):
    ''' Average referencing the data probe-wise. Goes over the channels_list (created with the function above), and uses data from each sub-list to reference the channels.
        The function also checks that there is more than one channel per probe. If not, it does not avg. reference the data for that channel.
        Returns RawArray object with the referenced data for all inputed probes'''
    
    averaged_raw_data = []
    # Iterate over the electrode contact lists
    for contacts in channels_list:
        
        # Pick the channels of the current electrode
        electrode_channels = raw_data.copy().pick_channels(contacts)
        # Check that there is more than one channel per probe.
        if len(contacts) < 2:
            print('Only one contact for the average referencing:',contacts,'. It has been added without referencing')
            averaged_raw_data.append(electrode_channels.get_data()) # skip referencing if there is only one channel per probe
            continue
        # Reference the channels of the current electrode
        electrode_channels.set_eeg_reference()
        # Append the referenced epoch of the current electrode to the list
        averaged_raw_data.append(electrode_channels.get_data())
    # Concatenate the referenced eochs into a single object
    referenced_data = np.concatenate(averaged_raw_data) # concatenates over channels
    # Create a new raw_data object with the referenced data
    averaged_raw_data = mne.io.RawArray(referenced_data, raw_data.info, first_samp=raw_data.first_samp) 

    return averaged_raw_data

#############

# Functions to plot epochs - channel variances 
def plot_variances_epochs_channels(epochs,threshold, median=True):

    ''' epochs should be an mne.Epochs object
        Threshold in std.: how many sdt. away from the mean to remove epoch/channel
        If median=True: median abs deviation * threshold will also be plotted
        Returns: variances heatmap. Each pixel is the variance in one epoch from one channel. 
                 It also returns idx of epochs and/or channels that have variances higher than mean + threshold * std'''
    # Calculate the variance of each epoch
    data = epochs.get_data()
    epoch_variances = np.var(data, axis=2)
    # print(epoch_variances.shape)

    # Create a figure with subplots
    fig, ax_heatmap = plt.subplots(figsize=(8, 6))
    divider = make_axes_locatable(ax_heatmap)
    ax_channels = divider.append_axes("right", size="30%", pad=0.1)
    ax_epochs = divider.append_axes("top", size="30%", pad=0.1)

    # Plot the heatmap
    im = ax_heatmap.imshow(epoch_variances.T, cmap='hot', aspect='auto', interpolation='nearest')
    fig.colorbar(im, ax=ax_heatmap, pad=0.1)
    ax_heatmap.set_xlabel('Epochs')
    ax_heatmap.set_ylabel('Channels')
    ax_heatmap.set_yticks(np.arange(len(epochs.info.ch_names)))  # Set y-ticks to match number of channels
    ax_heatmap.set_yticklabels(epochs.info.ch_names)
    # ax_heatmap.set_title('Epoch Variances Heatmap')

    # Plot the scatter plot for channel variances
    channel_indices = np.arange(epoch_variances.shape[1])
    channel_means = np.mean(epoch_variances, axis=0)
    ax_channels.scatter(channel_means, channel_indices, c='red', s=5) # channel_indices[::-1]
    ax_channels.axvline(channel_means.mean() + threshold *channel_means.std(), ls='--', c='k', lw='0.5')
    if median:
        ax_channels.axvline(np.median(channel_means) + threshold * scipy.stats.median_abs_deviation(channel_means), ls='--', lw='0.5')
    ax_channels.axvline(channel_means.mean(), ls='--', lw='0.5')
    ax_channels.yaxis.tick_left()
    ax_channels.yaxis.set_label_position("left")
    ax_channels.spines['right'].set_visible(False)
    ax_channels.spines['top'].set_visible(False)
    ax_channels.tick_params(axis='y', left=False, labelleft=False)
    ax_channels.set_ylim(channel_indices[-1]+0.5, channel_indices[0]+0.5) # Set the y-axis limits to span from last to first channel
    # find extreme channel variances (5std, find reference)
    extreme_ch_var_idx = np.where(channel_means > (channel_means.mean() +  threshold * channel_means.std()))[0]
    print('Noisy channel idxs:', np.where(channel_means > (channel_means.mean() +  threshold * channel_means.std()) )[0]) # | channel_means < (channel_means.mean() - channel_means.std())

    # Plot the scatter plot for epoch variances
    epoch_indices = np.arange(epoch_variances.shape[0])
    epoch_means = np.mean(epoch_variances, axis=1)
    ax_epochs.scatter(epoch_indices, epoch_means, c='blue', s=5)
    ax_epochs.axhline(epoch_means.mean() + threshold * epoch_means.std(), ls='--', c='k', lw='0.5', label=f'{threshold} * std')
    if median:
        ax_epochs.axhline(np.median(epoch_means)+ threshold * scipy.stats.median_abs_deviation(epoch_means), ls='--', lw='0.5', label=f'{threshold} * med. abs. dev.')
    # ax_epochs.set_ylim(0,5e-8)
    ax_epochs.set_ylabel('Variance')
    ax_epochs.spines['top'].set_visible(False)
    ax_epochs.spines['right'].set_visible(False)
    ax_epochs.tick_params(axis='x', bottom=False, labelbottom=False)
    ax_epochs.set_xlim(epoch_indices[0], epoch_indices[-1])  # Set the x-axis limits to span from first to last epoch
    ax_epochs.legend(loc='upper right', bbox_to_anchor=(1.48, 1.0))
    # find extreme channel variances (5std, find reference)
    extreme_epoch_var_idx = np.where(epoch_means > (epoch_means.mean() +  threshold * epoch_means.std()))[0]
    print('Noisy epochs idxs:', np.where(epoch_means > (epoch_means.mean() +  threshold * epoch_means.std()) )[0]) # | epoch_means < (epoch_means.mean() - epoch_means.std())
    
    fig.suptitle('Epoch-Channel Variances Heatmap')
    fig.tight_layout()
    plt.show(block=False) # this is for when using matplotlib.use('Qt5Agg')

    return np.array(extreme_ch_var_idx), np.array(extreme_epoch_var_idx)

# Plot epochs for a specific channel (choose how many, starting from where)
def plot_epochs_of_one_channel(epochs, ch_idx, num_epochs, start_epoch=None, test_ch_idx=None):

    ''' Plots epochs for specific channel.
        Inputs: epochs object, idx of noisy channel, number of epochs that should be plotted to compare channels, start_epoch to set idx of first plotted epoch, test_ch_idx = what ch reference to be plotted
        Outputs: figure of n plots corresponding to num_epochs'''

    data = epochs.get_data()

    if start_epoch == None:
        start_epoch = 0
    else: start_epoch = start_epoch
    num_epochs = num_epochs

    if start_epoch > data.shape[0] - num_epochs:
        print('vas', data.shape[0])
        raise ValueError(f'''Cannot index over total num of epochs ({data.shape[0]}). Given the start epoch idx ({start_epoch}), 
                pick a num_epochs value equal or smaller than {data.shape[0]-start_epoch}, or start epoch =< than {data.shape[0]-num_epochs}''')

    n_epochs = np.arange(start_epoch, start_epoch + num_epochs, 1)

    # Calculate the number of rows and columns for an optimal layout
    cols = int(np.ceil(np.sqrt(num_epochs)))
    rows = int(np.ceil(num_epochs / cols))

    # plot
    fig, axs = plt.subplots(rows, cols, figsize=(12, 2 * rows))

    # Loop through subplots and plot data
    for i, n in enumerate(n_epochs):
        row = i // cols
        col = i % cols
        y = np.squeeze(data[n][ch_idx][:])
        axs[row, col].plot(epochs.times, y)
        if type(test_ch_idx) == int: # check if test channel is True
            z = np.squeeze(data[n][test_ch_idx][:])
            axs[row, col].plot(epochs.times, z)
        axs[row, col].set_title(f'Epoch {n}')
        axs[row, col].set(ylim=(-5e-4, 5e-4))

    # Remove empty subplots
    for i in range(num_epochs, rows * cols):
        fig.delaxes(axs.flatten()[i])

    fig.legend(['Noisy ch', 'Control ch'], loc='upper right')

    plt.suptitle(f'Testing channel {ch_idx}', y=1., fontsize=15)
    plt.tight_layout()
    sns.despine()
    plt.show(block=False) # to interact well with pyqt5

# Plot channels for one epoch
def plot_channels_of_one_epoch(epochs, epoch_idx, num_channels, start_channel=None, test_epoch_idx=None):

    ''' Plots channels for specific epoch.
        Inputs: epochs object, idx of noisy epoch, number of channels that should be plotted to compare channels, start_channel to set idx of first plotted channel, test_epoch_idx = what epoch reference to be plotted
        Outputs: figure of n plots corresponding to num_channels'''
    
    data = epochs.get_data()
    
    if start_channel == None:
        start_channel = 0
    else: start_channel = start_channel
    num_channels = num_channels

    if start_channel > data.shape[1] - num_channels:
        print('vas', data.shape[1])
        raise ValueError(f'''Cannot index over total num of channels ({data.shape[1]}). Given the start channel idx ({start_channel}), 
                pick a num_channels value equal or smaller than {data.shape[1]-start_channel}, or start channel =< than {data.shape[1]-num_channels}''')

    n_channels = np.arange(start_channel, start_channel + num_channels, 1)

    # Calculate the number of rows and columns for an optimal layout
    cols = int(np.ceil(np.sqrt(num_channels)))
    rows = int(np.ceil(num_channels / cols))

    # plot
    fig, axs = plt.subplots(rows, cols, figsize=(12, 2 * rows))

    # Loop through subplots and plot data
    for i, n in enumerate(n_channels):
        row = i // cols
        col = i % cols
        y = np.squeeze(data[epoch_idx,n,:])
        axs[row, col].plot(epochs.times, y)
        if type(test_epoch_idx) == int: # check if test channel is True
            z = np.squeeze(data[test_epoch_idx][n][:])
            axs[row, col].plot(epochs.times, z)
        axs[row, col].set_title(f'Channel {n}')
        axs[row, col].set(ylim=(-5e-4, 5e-4))


    # Remove empty subplots
    for i in range(num_channels, rows * cols):
        fig.delaxes(axs.flatten()[i])

    fig.legend(['Noisy epoch', 'Control epoch'], loc='upper right')
    
    plt.suptitle(f'Testing epoch {epoch_idx}', y=1., fontsize=15)
    plt.tight_layout()
    sns.despine()
    # Show the plot
    plt.show(block=False)


#############

# Functions to visualize and select probes

# Get channel names grouped by areas (returns a dict: area_name: list of channels), inputs: the file with all channels, list with relevant channels
def group_channels_brain_regions(df, relevant_channels):
    ''' This function groups the channels based on which brain area they belong from the AAL3 atlas. 
        The input is the dataframe (df) with all channel information, and a list of channels of interest (relevant_channels) (potentially the ones that fall into the gray matter)
        The output is a dictionary in which each key is a brain area and each value is a list of the contacts that belong to that area'''
    
    # Define possible column names for brain region information (this is because .tsv files migth name the same column differently...)
    column_names = ['AAL3 (MNI-segment)', 'AAL3 (MNI-linear)']
    
    # Iterate over possible column names to find the correct one in the DataFrame
    for col_name in column_names:
        if col_name in df.columns:
            region_column = col_name
            break

    # Filter df to include only relevant channels
    relevant_df = df[df['Channel'].isin(relevant_channels)]

    # Group by brain area and aggreagate the channels into lists
    grouped_channels = relevant_df.groupby(region_column)['Channel'].apply(list).to_dict()

    return grouped_channels

# Plot channel locations in a 3d brain. Input channels of interest, file with all channels
def plot_channel_locations(df, list_channels, title):
    ''' This function plots the mni location of the inputed channels
        Input: df with all probes, mni locations,...,
               list_channels: if it is a list, all locations will be ploted with the same color. If it is a list of lists, each sub list will have a specific color (nice if e.g., plotting diff brain areas)
               title = title of the plot'''
                

    if isinstance(list_channels[0], list): # check if it is a list of lists

        # List of lists: assign colors to each sublist dynamically
        colors = plt.cm.get_cmap(lut=len(list_channels)).colors # potentially i could do name='vidris'
        color_dict = {}
        coordinates = []
        for idx, sublist in enumerate(list_channels):
            color = colors[idx]
            for channel in sublist:
                mni_loc = df.loc[df['Channel'] == channel, 'MNI'].values[0]
                coord = tuple(map(float, eval(mni_loc)))
                coordinates.append(coord)
                color_dict[channel] = color

        # Create an empty connectivity matrix (for plotting purposes)
        n_regions = len(coordinates)
        connectivity = np.zeros((n_regions, n_regions))

        # Create a plain brain template and plot
        display = plotting.plot_connectome(connectivity, coordinates, node_size=10, display_mode='ortho', node_color=[color_dict[channel] for sublist in list_channels for channel in sublist], title=title)
        plt.show(block=False)

    else:
        # In case of a single list, plot using only one color
        mni_locs = df.loc[df['Channel'].isin(list_channels), 'MNI'].values
        coordinates = [tuple(map(float, eval(coord))) for coord in mni_locs]

        # Create an empty connectivity matrix (for plotting purposes)
        n_regions = len(coordinates)
        connectivity = np.zeros((n_regions, n_regions))

         # Create a plain brain template and plot
        display = plotting.plot_connectome(connectivity, coordinates, node_size=10, display_mode='ortho', node_color='b', title=title)
        plt.show(block=False)


# Good reference that explains each brain area (label) from the used atlas AAL3: https://beckycutler.medium.com/aal3-an-overview-of-rois-62be6aef2de1
# actual paper: https://www.sciencedirect.com/science/article/pii/S1053811919307803



##############

# Functions to get index for seen correct (SC), seen incorrect (SI), unseen correct (UC), and unseen incorrect (UI)

def get_index_trial_type(events_df):

    ''' This function get the index for SC, SI, UC, and UI trials from an events dataframe
        Input: events dataframe of one participant
        Output: dictionary with keys as trial type and values the indices
        NOTE: only trials with target and mask (and good) are selected'''
    
    events_df_filtered = events_df[(events_df['Mask'] == 1) & (events_df['Target'] == 1) & # select events that have mask and target
                                   (events_df['Type'].isin(['SC', 'SI', 'UC', 'UI']))] # select only 4 trial types (to avoid NaNs)
    idx_type_dict = events_df_filtered.groupby('Type').groups # keys will be trial type and values pd index object

    return idx_type_dict