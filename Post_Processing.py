####Key functions to estimate the duration of MCC###
####Input: deep learning outputted 2-d (time-height) dimension classification####
####Output: duration, start and end time


def merge_periods(mcc_periods):

    if not mcc_periods:
        return []

    merged_periods = [mcc_periods[0]]
    for start, end, label in mcc_periods[1:]:
        last_end, last_label = merged_periods[-1][1], merged_periods[-1][2]

        # Determine the maximum allowed gap based on the cloud type
        if label == 2:
            max_gap = 2  # Closed MCC can merge if the gap is less than 2 hours
        else:
            max_gap = 4  # Open MCC can merge if the gap is less than 4 hours, change to 5 hour as sensitivity

        
        # Check if the current period can be merged with the last one
        if label == last_label and (start - last_end < max_gap):
            merged_periods[-1] = (merged_periods[-1][0], end, label)  # Merge periods by updating the end time
        else:
            merged_periods.append((start, end, label))  # Otherwise, add as a new period
    
    #print(merged_periods)
    
    return merged_periods


def check_dominance(merged_periods, cloud_labels, time_data):
    dominant_periods = []
    for start, end, label in merged_periods:
        
        #print(merged_periods)
        
        start_index = np.searchsorted(time_data, start)
        end_index = np.searchsorted(time_data, end, side='right') - 1

        # Extract the segment of cloud labels using indices
        segment = cloud_labels[start_index:end_index + 1]
        
        
        # Apply different dominance criteria based on the cloud type
        if label == 2:
            label_proportion = np.sum(segment == label) / len(segment)
            dominance_threshold = 0.6  #  dominance (0.6) threshold for cloud type 2
            if label_proportion > dominance_threshold:
                dominant_periods.append((start, end, label))

                
        if label == 3:
            #label_proportion = np.sum(segment == label) / np.sum((segment == label) | (segment == 0) )
            #dominance_threshold = 0.5  # Adjust threshold for cloud type 3 or others
            open_cells = np.sum(segment == 3)
            other_cells = np.sum(segment == 0)
            # open> 0.6 (>0.5)
            if open_cells > other_cells*1.0:
                dominant_periods.append((start, end, label))     


    return dominant_periods


def extract_details(periods, label_type):
    if not periods:
        return (np.nan, np.nan, np.nan)

    # Filter periods to include only those that match the specified label type
    specific_periods = [(start, end) for start, end, label in periods if label == label_type]
    if not specific_periods:
        return (np.nan, np.nan, np.nan)

    # Determine the period with the maximum duration and return it
    max_period = max(specific_periods, key=lambda x: x[1] - x[0], default=(np.nan, np.nan, np.nan))
    if max_period == (np.nan, np.nan, np.nan):
        return (np.nan, np.nan, np.nan)

    # Calculate duration based on the maximum period
    duration = max_period[1] - max_period[0]
    start = max_period[0]
    end = max_period[1]

    return (duration, start, end)


def calculate_mcc_durations(cloud_labels, time_data):
    if len(cloud_labels) == 0 or len(time_data) == 0:
        # If no data is provided, return NaNs for all values
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    
    mcc_periods = []
    current_label = None
    start_time = None
    label_counts = {2: 0, 3: 0, 'other': 0}

    # Process each time point
    for i in range(len(cloud_labels)):
        label = cloud_labels[i]
        time = time_data[i]

        if label in [2, 3]:  # Handling MCC labels
            if current_label != label:
                if current_label in [2, 3] and start_time is not None:
                    mcc_periods.append((start_time, time_data[i - 1], current_label))
                start_time = time
                current_label = label
                label_counts = {2: 0, 3: 0, 'other': 0}
            label_counts[label] += 1
        else:
            label_counts['other'] += 1
            if current_label in [2, 3] and start_time is not None:
                mcc_periods.append((start_time, time_data[i - 1], current_label))
                current_label = None
                start_time = None

    # Close the last period if still open
    if current_label in [2, 3] and start_time is not None:
        mcc_periods.append((start_time, time_data[-1], current_label))
        
    #print(mcc_periods)

    # Merge periods that are very close together
    merged_periods = merge_periods(mcc_periods)

    #print(merged_periods)
    
    # Check dominance of MCC labels over other labels
    final_periods = check_dominance(merged_periods, cloud_labels, time_data)
    
    #print(final_periods)

    open_details = extract_details(final_periods, 3)
    closed_details = extract_details(final_periods, 2)

    return (*open_details, *closed_details)



def input_2d_mask_output_1d_mask(y_pred_labels_post_merge_4hr):

    y_pred_labels_1d_4hr=np.zeros_like(new_time_data_4hr)+1
    for i in np.arange(768):
        unique_values_4hr=np.unique(y_pred_labels_post_merge_4hr[i,:])
        if 2 in unique_values_4hr or 3 in unique_values_4hr:
            y_pred_labels_1d_4hr[i] = 2 if 2 in unique_values_4hr else 3
        elif 0 in unique_values_4hr: ## If 0 is present and no 2 or 3
            y_pred_labels_1d_4hr[i] = 0    

    return y_pred_labels_1d_4hr



#### how to call above functions, example for 1 case###
for i in np.arange(1)+100:  

    #####Read in predicted values from deep learning#####
    file_name_array=f'DL_Output_{date_array[i]}.nc'
    
    if len(glob.glob(f'{outdir}{file_name_array}'))==1:
        
        output_file=glob.glob(f'{outdir}{file_name_array}')[0]
        dsout_DL=xr.open_dataset(output_file)
        y_pred_labels_post_merge_4hr=np.squeeze(dsout_DL.y_pred_labels_post_merge_4hr.values)
        new_time_data_4hr=np.squeeze(dsout_DL.new_time_data_4hr.values)
        y_pred_labels_1d_4hr=\
        input_2d_mask_output_1d_mask(y_pred_labels_post_merge_4hr)

        ### Your input is a time series of classification from deep learning (y_pred_labels_1d_4hr)###

        duration_open_4hr, start_open_4hr, end_open_4hr, duration_closed_4hr, start_closed_4hr, end_closed_4hr = calculate_mcc_durations(y_pred_labels_1d_4hr, new_time_data_4hr)
        




