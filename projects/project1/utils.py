import numpy as np






def downsample_class_0(X, y, downsample_size_0):
    # Separate the samples with label 0 and label 1
    indices_class_0 = np.where(y == 0)[0]
    indices_class_1 = np.where(y == 1)[0]
    
    # Downsample the class 0 samples
    selected_indices_class_0 = np.random.choice(indices_class_0, size=downsample_size_0, replace=False)
    
    # Combine the downsampled class 0 samples with all class 1 samples
    combined_indices = np.concatenate([selected_indices_class_0, indices_class_1])
    
    # Shuffle the combined indices to mix the samples
    np.random.shuffle(combined_indices)
    
    # Select the corresponding rows from X and y
    X_downsampled = X[combined_indices]
    y_downsampled = y[combined_indices]
    
    return X_downsampled, y_downsampled


def remove_features(x, features_to_remove, features) :
    
    if not set(features_to_remove).intersection(features):
        return features, x
    
    for feature in features_to_remove:
        if feature in features :
            x = np.delete(x, features[feature], axis = 1)
            features = remove_and_update_indices(features, feature)
            
    return features, x

def find_key_by_value(d, value):
    for key, val in d.items():
        if val == value:
            return key
    return None

def remove_and_update_indices(d, remove_key):
    if remove_key in d:
        removed_index = d.pop(remove_key)
        for key in d:
            if d[key] > removed_index:
                d[key] -= 1

    return d



def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8
    you will have 80% of your data set dedicated to training
    and the rest dedicated to testing. If ratio times the number of samples is not round
    you can use np.floor. Also check the documentation for np.random.permutation,
    it could be useful.

    Args:
        x: numpy array of shape (N,), N is the number of samples.
        y: numpy array of shape (N,).
        ratio: scalar in [0,1]
        seed: integer.

    Returns:
        x_tr: numpy array containing the train data.
        x_te: numpy array containing the test data.
        y_tr: numpy array containing the train labels.
        y_te: numpy array containing the test labels.

    >>> split_data(np.arange(13), np.arange(13), 0.8, 1)
    (array([ 2,  3,  4, 10,  1,  6,  0,  7, 12,  9]), array([ 8, 11,  5]), array([ 2,  3,  4, 10,  1,  6,  0,  7, 12,  9]), array([ 8, 11,  5]))
    """

    # set seed
   
    np.random.seed(seed)
    x = np.random.permutation(x)
    np.random.seed(seed)
    y = np.random.permutation(y)
    # ***************************************************
    # INSERT YOUR CODE HERE
    # split the data based on the given ratio: TODO
    # ***************************************************
    sample_size = y.shape[0]
    training_size = (int)(np.floor(ratio*sample_size))
    x_tr = x[:training_size]
    x_te = x[training_size :]
    y_tr = y[:training_size]
    y_te = y[training_size:]
    
    return x_tr, x_te, y_tr, y_te





def stratified_train_test_split(X, y, test_size=0.2, random_seed=None):
    """
    Split X and y into train and test sets while maintaining the distribution of classes.
    
    Parameters:
    - X: Feature matrix, shape (n_samples, n_features)
    - y: Labels, shape (n_samples,)
    - test_size: Proportion of the dataset to include in the test split (default=0.2)
    - random_seed: Seed for reproducibility (default=None)
    
    Returns:
    - X_train, X_test, y_train, y_test
    """

       
    
    # Get unique classes and their indices
    classes, y_indices = np.unique(y, return_inverse=True)
    
    # Initialize lists for stratified splits
    X_train, X_test = [], []
    y_train, y_test = [], []
    
    # Loop through each class and split data accordingly
    for class_label in classes:
        # Get the indices of the samples with the current class label
        class_indices = np.where(y == class_label)[0]
        
        # Shuffle the class-specific data
        np.random.seed(random_seed)
        np.random.shuffle(class_indices)
        
        # Split based on test_size
        split_idx = int(len(class_indices) * (1 - test_size))
        
        train_idx = class_indices[:split_idx]
        test_idx = class_indices[split_idx:]
        
        # Append stratified data to train and test lists
        X_train.append(X[train_idx])
        X_test.append(X[test_idx])
        y_train.append(y[train_idx])
        y_test.append(y[test_idx])
    
    # Concatenate lists into arrays
    X_train = np.concatenate(X_train)
    X_test = np.concatenate(X_test)
    y_train = np.concatenate(y_train)
    y_test = np.concatenate(y_test)
    
    return X_train, X_test, y_train, y_test