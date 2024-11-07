

def handle_correlation(x_train, features):
    """
    Handling correlation between features
    :param x_train: training data
    :param features: features list
    :return: modified and correlation handled data
    """
    # Compute the correlation matrix
    features_correlation = np.corrcoef(x_train, rowvar=False)

    # Find the features that are highly correlated
    correlation_limit = 0.8
    correlation_tuple_list = []
    correlation_list = []

    for i in range(x_train.shape[1]) : 
        for j in range(i, x_train.shape[1]) : 
            if i != j and features_correlation[i,j] >= correlation_limit : 
                correlation_tuple_list.append((find_key_by_value(features, i) , find_key_by_value(features, j)))
                correlation_list.append(find_key_by_value(features, i))
                correlation_list.append(find_key_by_value(features, j))
    
    # Use np.unique to get counts of each element
    correlation_list = np.array(correlation_list)
    unique_elements, counts = np.unique(correlation_list, return_counts=True)
    count_elements = dict(zip(unique_elements, counts))

    features_to_remove = set()
    # Iterate through the correlation tuples
    for feature1, feature2 in correlation_tuple_list:
        # Compare the counts of the two features
        if count_elements[feature1] > count_elements[feature2]:
            features_to_remove.add(feature2)
        else:
            features_to_remove.add(feature1)

    # Remove the features from the features dictionary and x_train_modified
    features, x_train = remove_features(x_train, list(features_to_remove), features)

    # Update the features dictionary to reflect the new indices
    features = {feature: i for i, feature in enumerate(features.keys())}

    return x_train, features



def handling_data(x_train, features):
    """
    Handling and modifying data because of special values and scaling some values
    :param x_train: training data
    :return: modified data
    """
    #Normalizing data
    for feature in features.keys() :
    
        dict_special_value_handle = dictionary_features[feature]


        for special_value in dict_special_value_handle.keys() :
            #For special values meaning not sure, we change by the median
            if special_value == 'dont_know_not_sure' :
                x_train[x_train[:, features[feature]] == dict_special_value_handle[special_value], features[feature]] = np.nanmedian(x_train[:, features[feature]])
                
            #For special values meaning none, we change by 0
            elif special_value == 'never' or special_value == 'none'  :
                x_train[x_train[:, features[feature]] == dict_special_value_handle[special_value], features[feature]] = 0
                    
            #For special values meaning refused, we change by NaN
            elif special_value == 'refused' :
                x_train[x_train[:, features[feature]] == dict_special_value_handle[special_value], features[feature]] = np.nan
    
    x_train = day_week_month_rescale(x_train, 'STRENGTH', 1, features)
    x_train = day_week_month_rescale(x_train, 'ALCDAY5', 1, features)
    x_train = day_week_month_rescale(x_train, 'FRUIT1', 2, features)
    x_train = day_week_month_rescale(x_train, 'FVBEANS', 2, features)
    x_train = day_week_month_rescale(x_train, 'FVORANG', 2, features)
    x_train = day_week_month_rescale(x_train, 'FVGREEN', 2, features)
    x_train = day_week_month_rescale(x_train, 'VEGETAB1', 2, features)
    x_train = day_week_month_rescale(x_train, 'FRUITJU1', 2, features)

    return x_train

def day_week_month_rescale(x, feature_name, scaling_mode, features):
    """
    Rescale the values of the feature_name
    :param x: training data
    :param feature_name: feature name
    :param scaling_mode: scaling mode
    :param features: features list
    :return: modified data
    """
    if feature_name in features.keys() :
    
        mask_three_hundred = x[:, features[feature_name]] > 300
        mask_two_hundred = (x[:, features[feature_name]] >= 200) & (x[:, features[feature_name]] < 300)
        mask_one_hundred = (x[:, features[feature_name]] >= 100) &(x[:, features[feature_name]] < 200) 
            
        if scaling_mode == 1 : 
            x[mask_one_hundred , features[feature_name]] = (x[mask_one_hundred, features[feature_name]] -100)*4.33
            x[mask_two_hundred, features[feature_name]] = (x[mask_two_hundred, features[feature_name]] -200)
            
        elif scaling_mode == 2:
            x[x[:, features[feature_name]] == 300, features[feature_name]] = 0
            x[mask_one_hundred , features[feature_name]] = (x[mask_one_hundred, features[feature_name]] -100)*(4.33*7)
            x[mask_two_hundred, features[feature_name]] = (x[mask_two_hundred, features[feature_name]] -200)*(4.33)
            x[mask_three_hundred, features[feature_name]] = (x[mask_three_hundred, features[feature_name]] -300)

    return x
    