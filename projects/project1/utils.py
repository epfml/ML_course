import numpy as np

def convert_number_to_string(num):
    # If the number is an integer, return it as a string
    if num.is_integer():
        return str(int(num))
    else:
        # Otherwise, return it as a string without unnecessary zeros
        return str(num)
    
def modf(number) :
    decimal = number - int(number)
    return round(decimal, 2)

def has_fractional_part(num):
    if np.isnan(num) :
        return False
    fractional_part = modf(num)
    return fractional_part != 0.0


def remove_features(x, features_to_remove, features) :
    
    if not set(features_to_remove).intersection(features):
        return features, x
    
    for feature in features_to_remove:
        x = np.delete(x, features[feature], axis = 1)
        features = remove_and_update_indices(features, feature)
        
    return features, x

def find_key_by_value(d, value):
    for key, val in d.items():
        if val == value:
            return key
    return None  # Return None if no match is found

def remove_and_update_indices(d, remove_key):
    if remove_key in d:
        removed_index = d.pop(remove_key)
        for key in d:
            if d[key] > removed_index:
                d[key] -= 1

    return d