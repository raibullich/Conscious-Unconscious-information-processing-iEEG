
# Functions for temporal decoders


import numpy as np
import pandas as pd
import mne
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from nilearn import plotting
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

###### FUNCTIONS TO DECODE TARGET LOCATION IN A SLIDING WINDOW AND PLOT RESULTS FOR ALL BRAIN AREAS
# function to decode target location with time window
def run_temporal_decoder_target_sliding_window(epochs, events, channels, twindow, trial_type_index, target='Location',control=False, clf='logisticregression', return_coefs=False, average_channels=False):
    
    """
        
        epochs = epoch object containing all channel names and trials
        events = behavioral dataframe containing labels of the trial types
        channels = specific channels of interest to decode the signal from. These channels will be pooled (as features) in the input matrix
        twindos = of the sliding window. how much datapoints are taken at each time to decode from.
        trial_type_index = provides the indices of the trial type to decode (e.g., idx SC)
        # label = (can be location or hemisphere) (not included in the function for now)
        control = if true, the function shuffles the labels before training the decoder
        clf = what classifier is being used, can be SVC or LR
        get_coefs = weights of each channel. Retunrs np.array with shape n_coefs(channels)Xtime
        average_channels = if true, it will average all chanels before running the sliding window decoder. Like this, only one feature (averaged channel) is inputed to the decoder.
        
        returns a list of scores of lenght t - twindow. This represents decoding scores at each timepoint
    """

    # np.random.seed(42) # to get reproducible results

    # initialize
    scores = np.zeros(len(epochs.times) - twindow +1)
    estimator_coefs = []

    if clf == 'svc':
        classifier = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True, class_weight='balanced'))
    elif clf == 'logisticregression':
        classifier = make_pipeline(StandardScaler(), LogisticRegression(solver="liblinear", class_weight='balanced')) # class weight to give more importance to unseen (less n) trials
        # pipeline = Pipeline([
        #             ('scaler', StandardScaler()),
        #             ('smote', SMOTE()),
        #             ('logistic', LogisticRegression(solver="liblinear", class_weight='balanced'))])
    else:
        raise KeyError("Classifier not found. Only 'svc' or 'logisticregression' allowed.")

    # labels = events.loc[(events['Target'] == 1) & (events['Mask'] == 1) & (events['Type'] == trial_type), label]
    labels = events.loc[trial_type_index, f'{target}'] # for now this is location, but changing variable target could be quadrant or hemispace
    # Select epochs that contain these labels
    data_matrix = np.squeeze(epochs.get_data(picks=channels))
    if average_channels and len(data_matrix.shape) == 3: # also check are multiple channels
        epochs_avg = data_matrix.mean(axis=1) # axis 1 is where channels are
        epochs = epochs_avg[labels.index]
    else:
        epochs = data_matrix[labels.index] 

    # Remove epochs and labels that have 'nan' in Type column
    idx_not_nans = labels != 'nan'
    labels, epochs = labels[idx_not_nans], epochs[idx_not_nans]

    # if control, shuffle the labels
    if control == True:
        labels = labels.sample(frac=1)

    # data
    X = epochs
    y = labels.values
    tw=twindow

    for i in range(X.shape[-1]- tw + 1): # loop through time
        if np.shape(channels) == (1,) or len(epochs.shape)!= 3: # check if there is a single channel (or averaged channels, acting as one channel)
            window_data = X[:, i:i + tw]
        else:
            window_data = X[:, :, i:i+twindow]

        # window_data = X[:, :, i:i + tw]
        window_data = window_data.reshape(X.shape[0], -1)

        # score = cross_val_score(classifier, window_data, y=y, cv=3, n_jobs=None, scoring="roc_auc_ovr_weighted", verbose=False).mean() # this was orignianlly set to 5 but there are classses with not enough labels
        # scores.append(score)

        score = cross_validate(classifier, window_data, y=y, cv=3, n_jobs=None, scoring="balanced_accuracy", verbose=False, return_estimator=True, return_train_score=False)# befre i had roc_auc_ovr_weighted
        # score = cross_validate(pipeline, window_data, y=y, cv=3, n_jobs=None, scoring="balanced_accuracy", verbose=False, return_estimator=True, return_train_score=False)# befre i had roc_auc_ovr_weighted
        # scores.append(score['test_score'].mean()) # store the score
        scores[i] = score['test_score'].mean() # store the score
        
        if return_coefs:
            est_coefs = np.mean([model.named_steps[f'{clf}'].coef_ for model in score['estimator']], axis=0) #  First I loop through each model fitted in the cv to get the coefs. Then average the coefs through crossval. Shape classes x features (channels)
            est_coefs = abs(est_coefs).mean(axis=0) # average the absoulte values of the coefficients across classes to get feature importance (in this case axis 1 is where the classes are)
            estimator_coefs.append(est_coefs) # store the coefficients, this has len(features). This will end up having shape time x features
        
    
    if return_coefs:
        return scores, np.array(estimator_coefs).T
    else:
        return scores
    
# function to decode target location for each brain region in a single subject
def decode_target_location_brain_area(epochs, events, dict_brain_regions, idx_trial_type, trial_types, tw, target='Location', average_channels=False):
    ''' EXPLAIN EACH INPUT VARIABLE
        trial_types (list of str) to be included in the decoding
        tw is time window.
        Output: dictionary. each key is a brain area, and each value is the results of the sliding decoder'''
    
    dict_results = {}

    for area in dict_brain_regions:
        print(area)
        area_results = {}

        # loop over trial types
        # for trial_type in idx_trial_type:
        for tt in trial_types:
            scores = run_temporal_decoder_target_sliding_window(epochs, events, dict_brain_regions[area], tw, idx_trial_type[tt], target, average_channels=average_channels)
            area_results[tt] = scores
        
        # Store the area_results dict in the main results dict with the current brain area as the key
        dict_results[area] = area_results

    return dict_results

# Function to plot decoding results one subject all brain areas
def plot_decoding_all_areas(dict_results, dict_brain_regions, times, labels, subj_num, target='Location', tw=1):
    ''' Labels refers to what trial types have been decoded
        Times is an array with the time values'''
    
    # Calculate the number of rows and columns for the subplot grid
    num_brain_areas = len(dict_results)
    # num_trial_types = len(next(iter(dict_results.values())))  # Assuming all brain areas have the same number of trial types
    num_rows = int(np.ceil(np.sqrt(num_brain_areas)))
    num_cols = int(np.ceil(num_brain_areas / num_rows))

    # Create subplots in a grid layout
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))

    # Track non-empty subplots
    non_empty_axes = []

    # Iterate over each brain area and its decoded data
    for i, (area, area_results) in enumerate(dict_results.items()):
        row = i // num_cols
        col = i % num_cols
        
        # Check if there is any data for the current brain area
        # if area_results:
        ax = axes[row, col] if num_brain_areas > 1 else axes  # Select subplot
        
        # Plot each trial type within the current brain area subplot
        for i, (trial_type, data) in enumerate(area_results.items()):
            if tw !=1:
                ax.plot(times[:-tw + 1], filter(data), label=trial_type, alpha=0.8)
            else: 
                if i ==0:
                    ax.plot(times, filter(data,1), label=trial_type, lw=2.1)
                else: ax.plot(times, filter(data,1), label=trial_type, alpha=0.7, lw=1.1)

        # ax.plot(times, filter(area_results['SC']), label='SC')
        
        
        ax.set(title = area, xlabel= 'Time (s)', ylabel='Accuracy')
        if target == 'Location':
            ax.axhline(1/8, ls='--', c='k')
            ax.text(0.65, 0.38, f'N_ch = {len(dict_brain_regions[area])}')
            ax.set(ylim=(0.05,0.42))
        if target == 'Quadrant':
            ax.axhline(1/4, ls='--', c='k')
            ax.text(1.5, 0.45, f'N_ch = {len(dict_brain_regions[area])}')
            ax.set(ylim=(0.05,0.52))
        if target == 'Hemispace':
            ax.axhline(1/2, ls='--', c='k')
            ax.set(ylim=(0.3,0.7))
            ax.text(1.5, 0.7, f'N_ch = {len(dict_brain_regions[area])}')
            
        ax.axvline(0.0, ls='--', c='k', lw=0.5)
        sns.despine()
        
        non_empty_axes.append(ax)
        # else:
        #     # If there is no data for the current brain area, do nothing
        #     pass

    # Remove empty subplots
    for ax in axes.flat:
        if ax not in non_empty_axes:
            fig.delaxes(ax)

    fig.legend(loc='center left', bbox_to_anchor=(1, 1), labels=labels, title='Trial Types')
    fig.suptitle(f'Decode target {target} Subject {subj_num}')
    # Adjust layout
    plt.tight_layout()
    plt.show()


##### FUNCTIONS TO RUN PERMUTATION CLUSTER BASED TEST ON DECODING RESULTS
# Here I am using the statistics used in Figure 2 in this paper: https://academic.oup.com/nc/article/2022/1/niac005/6535734.

def get_observed_and_permutation_scores(epochs, events, channels, twindow, trial_type_index, target='Location', clf='logisticregression', n_permutations=30, average_channels=False):
    # Compute observed accuracy
    observed_scores = run_temporal_decoder_target_sliding_window(epochs, events, channels, twindow, trial_type_index, target, control=False, clf=clf, average_channels=average_channels)
    # Perform permutations
    permuted_scores = np.zeros((n_permutations, len(observed_scores)))
    for i in range(n_permutations):
        permuted_scores[i] = run_temporal_decoder_target_sliding_window(epochs, events, channels, twindow, trial_type_index, target, control=True, clf=clf, average_channels=average_channels)
    return observed_scores, permuted_scores

# Function to compute t-values
def compute_t_values(observed_scores, permuted_scores):
    mean_permuted = np.mean(permuted_scores, axis=0)
    std_permuted = np.std(permuted_scores, axis=0)
    t_values = (observed_scores - mean_permuted) / std_permuted
    return t_values

# Function to compute p-values
def compute_p_values(observed_scores, permuted_scores):
    n_permutations = permuted_scores.shape[0]
    p_values = np.zeros(observed_scores.shape)
    for i in range(len(observed_scores)):
        p_values[i] = (np.sum(permuted_scores[:, i] >= observed_scores[i]) + 1) / (n_permutations + 1)
    return p_values

# Function to identify clusters
def identify_clusters(data, threshold):
    binary_data = data < threshold
    labeled_array, num_features = scipy.ndimage.label(binary_data)
    return labeled_array, num_features

# Function to compute cluster statistics
def compute_cluster_statistics(data, clusters, num_clusters):
    cluster_stats = np.zeros(num_clusters)
    for i in range(1, num_clusters + 1):
        cluster_stats[i - 1] = np.sum(data[clusters == i])
    return cluster_stats

# Cluster-based permutation test function
def cluster_based_permutation_test(observed_scores, permuted_scores, p_value_threshold=0.05):

    # Compute T-values for each timepoint
    observed_t_values = compute_t_values(observed_scores, permuted_scores)
    # print(observed_t_values)
    # Compute P-values for each timepoint
    p_values = compute_p_values(observed_scores, permuted_scores)
    # Cluster significant timepoints (pval < 0.05)
    observed_clusters, _ = identify_clusters(p_values, p_value_threshold) # Function to identify clusters

    # Compute sum of T-values for each cluster
    observed_cluster_stats = []
    unique_clusters = np.unique(observed_clusters)[1:]  # value 0 is not a cluster

    if len(unique_clusters) > 0:
        for cluster_label in unique_clusters:
            cluster_indices = np.where(observed_clusters == cluster_label)[0]
            cluster_t_values = observed_t_values[cluster_indices] # get tvals in cluster idx
            cluster_sum_t = np.sum(cluster_t_values)
            observed_cluster_stats.append(cluster_sum_t)

        # Compare with surrogate distribution
        n_permutations = permuted_scores.shape[0] # this should not be here tho
        surrogate_cluster_sums = np.zeros(len(unique_clusters))  # Initialize array for surrogate cluster sums
        # print(len(surrogate_cluster_sums), len(unique_clusters))
        for i in range(n_permutations):
            permuted_t_values = compute_t_values(permuted_scores[i], permuted_scores)
            # print(unique_clusters)
            for j, cluster_label in enumerate(unique_clusters):
                cluster_indices = np.where(observed_clusters == cluster_label)[0] 
                surrogate_cluster_sums[j] += np.sum(permuted_t_values[cluster_indices]) # create a distribution of tvals for each cluster (this distrb will be used to compute significance for the observed_cluster tvals). the idx cluster is obtained from observed_clusters

        # Compare cluster tval sums with surrogate distribution of tvals
        threshold_percentile = 99.9
        threshold_value = np.percentile(surrogate_cluster_sums, threshold_percentile)

        # Determine significant clusters
        significant_clusters = observed_clusters.copy()
        for cluster_id, cluster_sum in enumerate(observed_cluster_stats, start=1):
            if cluster_sum <= threshold_value: # look at NON significant clusters and set to 0
                significant_clusters[significant_clusters == cluster_id] = 0

        significant_cluster_indices = []
        for cluster_id in np.unique(significant_clusters): # get index of the significant clusters and return
            if cluster_id == 0:
                continue
            significant_cluster_indices.append(np.where(significant_clusters == cluster_id)[0])

        return significant_cluster_indices
    else: return []

# Function to run decoder + permutation + obtain significant clusters for each area and store the results
def run_temporal_decoder_all_areas_permutation(epochs, events, dict_brain_regions, idx_trial_type, trial_types, tw=1, target='Location', n_permutations=30):
  
    dict_results = {}

    for area in dict_brain_regions:
        print(area)
        area_results = {}

        # loop over trial types
        for tt in trial_types:
            observed_scores, permuted_scores = get_observed_and_permutation_scores(epochs, events, dict_brain_regions[area], tw, idx_trial_type[tt], n_permutations=n_permutations)
            # print(permuted_scores.shape)
            significant_clusters = cluster_based_permutation_test(observed_scores, permuted_scores)
            # Initialize area_results[tt] as a dictionary if it doesn't exist
            if tt not in area_results:
                area_results[tt] = {}
            
            area_results[tt]['observed'] = observed_scores
            area_results[tt]['permuted'] = permuted_scores
            area_results[tt]['significant clusters'] = significant_clusters

        dict_results[area] = area_results

    return dict_results

# plotting functions for results (with permutation test)
# Function to plot decoding score per area, trial type and statistics
def plot_decoding_all_areas_w_stats(dict_results, dict_brain_regions, times, labels, subj_num, target='Location', tw=1):
    # Calculate the number of rows and columns for the subplot grid
    num_brain_areas = len(dict_results)
    # num_trial_types = len(next(iter(dict_results.values())))  # Assuming all brain areas have the same number of trial types
    num_rows = int(np.ceil(np.sqrt(num_brain_areas)))
    num_cols = int(np.ceil(num_brain_areas / num_rows))

    # Create subplots in a grid layout
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))

    # Track non-empty subplots
    non_empty_axes = []

    # Iterate over each brain area and its decoded data
    for i, (area, area_results) in enumerate(dict_results.items()):
        row = i // num_cols
        col = i % num_cols
        
        # Check if there is any data for the current brain area
        # if area_results:
        ax = axes[row, col] if num_brain_areas > 1 else axes  # Select subplot
        
        # Plot each trial type within the current brain area subplot
        for i, (trial_type, data) in enumerate(area_results.items()):

            data_observed = filter(data['observed'],1)
            sig_clusters = data['significant clusters']

            all_idx = np.arange(0,len(data_observed))
            idx_clusters = [idx for cluster in sig_clusters for idx in cluster]
            idx_non_clusters = [idx for idx in all_idx if idx not in idx_clusters]
            data_clusters = data_observed.copy()
            data_clusters[idx_non_clusters] = np.nan

            if tw !=1: # if tw is greater than 1
                ax.plot(times[:-tw + 1], data_clusters, label=trial_type, alpha=1, lw=2)
                ax.plot(times[:-tw + 1], data_observed, label=trial_type, alpha=0.6, lw=1, c=f'C{i}')

                for cluster_idx in sig_clusters:
                    # ax.hlines(y=np.max(observed_scores) + 0.01- i*0.02, xmin=times[cluster_idx[0]], xmax=times[cluster_idx[-1]], color=f'C{i}', linewidth=3)
                    ax.hlines(y=np.max(data_observed) + 0.01- i*0.02, xmin=times[cluster_idx[0]], xmax=times[cluster_idx[-1]], color=f'C{i}', linewidth=3)
            else:
                ax.plot(times, data_clusters, label=trial_type, alpha=1, lw=2)
                ax.plot(times, data_observed, label=trial_type, alpha=0.6, lw=1, c=f'C{i}')

                for cluster_idx in sig_clusters:
                    # ax.hlines(y=np.max(observed_scores) + 0.01 - i*0.02, xmin=times[cluster_idx[0]], xmax=times[cluster_idx[-1]], color=f'C{i}', linewidth=3)
                    ax.hlines(y=0.4 + 0.01 - i*0.02, xmin=times[cluster_idx[0]], xmax=times[cluster_idx[-1]], color=f'C{i}', linewidth=3)

        # set labels and other plotting features
        ax.set(title = area, xlabel= 'Time (s)', ylabel='Accuracy (%)')
        if target == 'Location':
            ax.axhline(1/8, ls='--', c='k')
            # ax.text(0.65, 0.38, f'N_ch = {len(dict_brain_regions[area])}')
            ax.set(ylim=(0.05,0.42))
        if target == 'Quadrant':
            ax.axhline(1/4, ls='--', c='k')
            # ax.text(1.5, 0.45, f'N_ch = {len(dict_brain_regions[area])}')
            ax.set(ylim=(0.05,0.52))
        if target == 'Hemispace':
            ax.axhline(1/2, ls='--', c='k')
            ax.set(ylim=(0.3,0.7))
            # ax.text(1.5, 0.7, f'N_ch = {len(dict_brain_regions[area])}')
            
        ax.axvline(0.0, ls='--', c='k', lw=0.5)
        sns.despine()
        
        non_empty_axes.append(ax)
        
    # Remove empty subplots
    for ax in axes.flat:
        if ax not in non_empty_axes:
            fig.delaxes(ax)

    colors = [f'C{i}' for i in range(len(labels))]
    legend_handles = [Line2D([0], [0], color=color, lw=4) for color in colors]
    fig.legend(loc='center left', bbox_to_anchor=(1, 1), handles=legend_handles, labels=labels, title='Trial Types')
    fig.suptitle(f'Decode target {target} Subject {subj_num}', y=1.01)
    # Adjust layout
    plt.tight_layout()
    plt.show()


##### FUNCTIONS TO DECODE FROM FREQUENCY BANDS
# filter bands for all subjs
def create_dict_freq_bands(epochs):
    # init dictionary
    dict_freq_bands = {}

    freq_bands = [[1.5, 4], [4.1, 8], [8.1, 12], [12.1, 35], [35.1, 45], [55, 80]]
    freq_bands_labels = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma', 'High gamma']

    for fr, fr_label in zip(freq_bands, freq_bands_labels):
        dict_freq_bands[fr_label] = epochs.copy().filter(l_freq = fr[0], h_freq = fr[1])

    return dict_freq_bands
    # return dict of one subject, each key is the frequency band, and values the epochs with that freq band.

# hilbert transform the signal
def get_envelope_phase(epoch):
    '''Reurns EvokedArray object for both envelope and phase features of each epoch''' 
    epoch_hilbert = epoch.copy().apply_hilbert().get_data() # this transformation gives weird (assimptotic) data at the extremes of the x axis
    envelope = np.abs(epoch_hilbert)
    phase = np.angle(epoch_hilbert)

    epoch_envelope = mne.EpochsArray(envelope, epoch.info, tmin=epoch.tmin)
    epoch_phase = mne.EpochsArray(phase, epoch.info, tmin=epoch.tmin)

    return epoch_envelope, epoch_phase


###### FUNCTIONS TO DECODE FROM POWER SPECTRUM TIMESERIES

# Decoder function
def run_temporal_decoder_target_sliding_window_powwer_timeseries(epochs, epochs_power_array, events, channel, trial_type_index, target='Location',control=False, clf='logisticregression', return_coefs=False, average_channels=False):
    
    """
        epochs = epoch object containing all channel names and trials
        epochs_power_array = ndarray variable of shape (n_trials, n_channels, n_frequencies, n_times) 
        events = behavioral dataframe containing labels of the trial types
        channels = specific channels of interest to decode the signal from. ONLY ONE CHANNEL ALLOWED IN THIS FUNCTION (otherwise input features are too high)
        trial_type_index = provides the indices of the trial type to decode (e.g., idx SC)
        control = if true, the function shuffles the labels before training the decoder
        clf = what classifier is being used, can be SVC or LR
        get_coefs = weights of each channel. Retunrs np.array with shape n_coefs(channels)Xtime
        average_channels = if true, it will average all chanels before running the sliding window decoder. Like this, only one feature (averaged channel) is inputed to the decoder.
        
        returns a list of scores of lenght t. This represents decoding scores at each timepoint
    """

    # np.random.seed(42) # to get reproducible results

    # initialize
    twindow= 1 # how many time bins per each model fit (always should be one in this case)
    scores = np.zeros(epochs_power_array.shape[-1] - twindow +1)
    estimator_coefs = []

    if clf == 'svc':
        classifier = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True, class_weight='balanced'))
    elif clf == 'logisticregression':
        classifier = make_pipeline(StandardScaler(), LogisticRegression(solver="liblinear", class_weight='balanced')) # class weight to give more importance to unseen (less n) trials
    else:
        raise KeyError("Classifier not found. Only 'svc' or 'logisticregression' allowed.")

    # labels = events.loc[(events['Target'] == 1) & (events['Mask'] == 1) & (events['Type'] == trial_type), label]
    labels = events.loc[trial_type_index, f'{target}'] # for now this is location, but changing variable target could be quadrant or hemispace

    if type(channel) != str:
        raise ValueError('Number of channels should not be > 1')

    index_channel = epochs.info['ch_names'].index(channel)
    data_matrix = epochs_power_array[:,index_channel,:,:]    
    
    # Select epochs with trial type index
    epochs_array = data_matrix[labels.index] 

    # Remove epochs and labels that have 'nan' in Type column
    idx_not_nans = labels != 'nan'
    labels, epochs_array = labels[idx_not_nans], epochs_array[idx_not_nans]

    # if control, shuffle the labels
    if control == True:
        labels = labels.sample(frac=1)

    # data
    X = epochs_array
    y = labels.values
    tw=twindow

    # Decoding loop
    for i in range(X.shape[-1]- tw + 1): # loop through time
        window_data = X[:, :, i:i + tw]
        window_data = window_data.reshape(X.shape[0], -1) # remove last dimention

        # Data has 30 features
        score = cross_validate(classifier, window_data, y=y, cv=3, n_jobs=None, scoring="balanced_accuracy", verbose=False, return_estimator=True, return_train_score=False)# befre i had roc_auc_ovr_weighted
        scores[i] = score['test_score'].mean() # store the score
        
        if return_coefs:
            est_coefs = np.mean([model.named_steps[f'{clf}'].coef_ for model in score['estimator']], axis=0) #  First I loop through each model fitted in the cv to get the coefs. Then average the coefs through crossval. Shape classes x features (channels)
            est_coefs = abs(est_coefs).mean(axis=0) # average the absoulte values of the coefficients across classes to get feature importance (in this case axis 1 is where the classes are)
            estimator_coefs.append(est_coefs) # store the coefficients, this has len(features). This will end up having shape time x features

    if return_coefs:
        return scores, np.array(estimator_coefs).T
    else:
        return scores

# plotting function
def plot_power_timeseries_decoder(epochs, dict_power_results,title_area, probes, list_channels):

    fig = plt.figure(figsize=(12,7))

    # PLOT THE LOCATION OF THE ELECTRODES IN THE 3D BRAIN
    mni_locs = probes.loc[probes['Channel'].isin(list_channels), 'MNI'].values
    coordinates = [tuple(map(float, eval(coord))) for coord in mni_locs]
    # Create an empty connectivity matrix (for plotting purposes)
    n_regions = len(coordinates)
    connectivity = np.zeros((n_regions, n_regions))

    # Create a plain brain template and plot
    ax1 = fig.add_subplot(2,2,1)
    ax1 = plotting.plot_connectome(connectivity, coordinates, node_size=5, display_mode='ortho', node_color=[f'C{i}' for i in range(len(list_channels))], axes=ax1)

    # PLOT THE DECODING OF EACH CHANNEL
    for i, key in enumerate(dict_power_results.keys()):
        # print(i)
        ax = fig.add_subplot(2,2,i+2)

        for j in range(len(dict_power_results[key])):
            ax.plot(epochs.times, filter(dict_power_results[key][j]))
        ax.axhline(0.125, c='k', ls='--')
        ax.axvline(0, c='k', ls='--', lw=0.5)
        ax.set(title=f'{key}', xlabel='time (s)', ylabel='accuracy', ylim=(0,0.31))
        sns.despine()

    # Define the starting position for the labels and lines
    start_x = 0.1  # Initial x position for the first label
    y = 0.97  # Y position for the labels (fixed)

    # Loop over the list_channels to add colored lines and black text labels
    for idx, channel in enumerate(list_channels):
        color = f'C{idx}'  # Define the color based on the index
        
        # Draw a short horizontal line with the corresponding color
        fig.lines.append(plt.Line2D([start_x, start_x + 0.02], [y, y], color=color, lw=4, transform=fig.transFigure, clip_on=False))
        
        # Add the channel name in black, next to the colored line
        fig.text(start_x + 0.03, y, channel, color='black', fontsize=12, ha='left')
        
        # Update the starting x position for the next label
        start_x += 0.1  # Adjust spacing as needed
    # Set the overall figure title and layout

    fig.suptitle(f'{title_area}', y=1.1, fontsize=20)
    fig.tight_layout()
    plt.show()


##### OTHER FUNCTIONS
# function to smooth data with a gaussian filter
def filter(data, sigma=5.): # smoothes for 5 datapoints
    filt_scores = scipy.ndimage.gaussian_filter1d(data, sigma)
    return filt_scores