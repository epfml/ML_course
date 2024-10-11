import numpy as np
import matplotlib.pyplot as plt
from helpers import *
from utils import remove_features, find_key_by_value

#Defining some constants
import numpy as np
import matplotlib.pyplot as plt
from helpers import *
from utils import remove_features, find_key_by_value

#Defining some constants
ACCEPTABLE_NAN_PERCENTAGE = 0.3
dictionary_features = {
    **dict.fromkeys(['_STATE', 'FMONTH', 'IDATE', 'IMONTH', 'IDAY', 'IYEAR',
                     'DISPCODE', 'SEQNO', '_PSU', 'SEX', 'QSTVER', '_STSTR',
                     '_STRWT', '_RAWRAKE', '_WT2RAKE', '_DUALUSE', '_LLCPWT',
                     '_DRDXAR1', '_RACE_G1', '_AGE80', '_AGE_G', 'HTIN4', 'HTM4', '_BMI5', '_BMI5CAT',
                     'FTJUDA1_', 'BEANDAY_', 'GRENDAY_', 'ORNGDAY_', 'VEGEDA1_', '_MISFRTN', '_MISVEGN',
                     '_FRTRESP', '_VEGRESP', '_FRUTSUM', '_FRUTSUM', '_FRT16', '_VEG23', '_FRUITEX',
                     '_VEGETEX', 'QSTLANG', 'FRUTDA1_', '_VEGESUM'
                   
                    ], {}),
    
    **dict.fromkeys(['ALCDAY5', 'STRENGTH'], {'dont_know_not_sure' : 777, 'none' : 888, 'refused' : 999}),
    
    **dict.fromkeys(['PHYSHLTH', 'MENTHLTH'], {'dont_know_not_sure' : 77, 'none' : 88, 'refused' : 99}),
    
    **dict.fromkeys(['CHILDREN'], {'none' : 88, 'refused' : 99}),
    
     **dict.fromkeys(['INCOME2', '_PRACE1', '_MRACE1'], {'dont_know_not_sure' : 77, 'refused' : 99}),

    **dict.fromkeys(['FRUITJU1', 'FRUIT1', 'FVBEANS', 'FVORANG', 'FVGREEN', 'VEGETAB1'], {'never' : 555, 'dont_know_not_sure' : 777, 'refused' : 999}),
    
     **dict.fromkeys(['GENHLTH', 'HLTHPLN1', 'PERSDOC2', 'MEDCOST', 'BPHIGH4',
                      'BLOODCHO', 'CHOLCHK', 'TOLDHI2', 'CVDSTRK3', 'ASTHMA3',
                      'CHCSCNCR', 'CHCOCNCR', 'CHCCOPD1', 'HAVARTH3', 'ADDEPEV2',
                      'CHCKIDNY', 'DIABETE3', 'RENTHOM1', 'VETERAN3', 'INTERNET',
                     'QLACTLM2', 'USEEQUIP', 'BLIND', 'DECIDE', 'DIFFWALK', 'DIFFDRES',
                      'DIFFALON', 'SMOKE100', 'USENOW3', 'EXERANY2', 'FLUSHOT6', 'PNEUVAC3',
                      'HIVTST6', 'DRNKANY5'
                     ], {'dont_know_not_sure' : 7, 'refused' : 9}),
    
    **dict.fromkeys(['CHECKUP1', 'SEATBELT'], {'dont_know_not_sure' : 7, 'never' : 8, 'refused' : 9}),
    
    **dict.fromkeys(['MARITAL', 'EDUCA', 'EMPLOY1', '_CHISPNC', '_RFHLTH',
                      '_HCVU651', '_RFHYPE5', '_CHOLCHK', '_RFCHOL', '_LTASTH1',
                      '_CASTHM1', '_ASTHMS1', '_HISPANC', '_RACE', '_RACEG21',
                      '_RACEGR3', '_RFBMI5', '_CHLDCNT', '_EDUCAG', '_INCOMG',
                      '_SMOKER3', '_RFSMOK3', '_RFBING5', '_RFDRHV5', '_FRTLT1',
                     '_VEGLT1', '_TOTINDA', 'PAMISS1_', '_PACAT1', '_PAINDX1', '_PA150R2',
                     '_PA300R2', '_PA30021', '_PASTRNG', '_PAREC1', '_PASTAE1', '_LMTACT1',
                     '_LMTWRK1', '_LMTSCL1', '_RFSEAT2', '_RFSEAT3', '_AIDTST3'
                     ], {'refused' : 9}),
    
    **dict.fromkeys(['HEIGHT3', 'WEIGHT2'], {'dont_know_not_sure' : 7777, 'refused' : 9999}),
    
    **dict.fromkeys(['_AGEG5YR'], {'refused' : 14}),

    **dict.fromkeys(['_AGE65YR'], {'refused' : 3}),

    **dict.fromkeys(['WTKG3'], {'refused' : 99999}),
    **dict.fromkeys(['DROCDY3_'], {'refused' : 900}),
    **dict.fromkeys(['_DRNKWEK'], {'refused' : 99900}),
     **dict.fromkeys(['FC60_', 'MAXVO2_'], {'refused' : 999}),
     **dict.fromkeys(['STRFREQ_'], {'refused' : 99}),
   
}

category_features = {'categorical' : ['HLTHPLN1','_STATE','FMONTH','IDATE', 'IMONTH', 'DISPCODE', 'PERSDOC2',
                             'MEDCOST', 'CHECKUP1', 'BPHIGH4', 'BLOODCHO', 'CHOLCHK', 'TOLDHI2', 'CVDSTRK3',
                            'ASTHMA3', 'CHCSCNCR', 'CHCOCNCR', 'CHCCOPD1', 'HAVARTH3', 'ADDEPEV2', 'CHCKIDNY'
                             'DIABETE3','SEX', 'MARITAL', 'EDUCA', 'RENTHOM1', 'VETERAN3', 'EMPLOY1', 'INCOME2', 'INTERNET', 'QLACTLM2',
                             'USEEQUIP', 'BLIND', 'DECIDE', 'DIFFWALK', 'DIFFDRES', 'DIFFALON', 'SMOKE100', 'USENOW3', 'EXERANY2', 'SEATBELT',
                             'FLUSHOT6', 'PNEUVAC3', 'HIVTST6', 'QSTVER', 'QSTLANG','_CHISPNC', '_DUALUSE', '_RFHLTH', '_HCVU651', '_RFHYPE5',
                             '_CHOLCHK', '_RFCHOL', '_LTASTH1', '_CASTHM1', '_ASTHMS1', '_DRDXAR1', '_PRACE1', '_MRACE1', '_HISPANC', '_RACE',
                             '_RACEG21', '_RACEGR3',  '_RACE_G1', '_AGEG5YR', '_AGE65YR', '_AGE_G', '_BMI5CAT', '_RFBMI5', '_CHLDCNT', '_EDUCAG',
                             '_INCOMG', '_SMOKER3', '_RFSMOK3', 'DRNKANY5', '_RFBING5', '_RFDRHV5', '_MISFRTN', '_MISVEGN', '_FRTRESP', '_VEGRESP',
                             '_FRTLT1', '_VEGLT1', '_FRT16', '_VEG23', '_FRUITEX', '_VEGETEX', '_TOTINDA', 'PAMISS1_', '_PACAT1', '_PAINDX1',
                             '_PA150R2', '_PA300R2', '_PA30021', '_PASTRNG', '_PAREC1', '_PASTAE1', '_LMTACT1', '_LMTWRK1', '_LMTSCL1',
                             '_RFSEAT2', '_RFSEAT3', '_AIDTST3', 'IYEAR', '_RFSMOK3', 'FMONTH', 'GENHLTH', 'CHCKIDNY', 'DIABETE3'
                             
                             
                            ],

            
            'continuous' : ['WEIGHT2', 'HEIGHT3', 'ALCDAY5', 'FRUITJU1', 'FRUIT1', 'FVBEANS', 'FVGREEN',
                            'FVORANG', 'VEGETAB1', 'STRENGTH', '_AGE80', 'HTIN4', 'WTKG3', 'HTM4', '_BMI5', 'DROCDY3_',
                           '_DRNKWEK', 'FTJUDA1_', 'BEANDAY_', 'GRENDAY_', 'ORNGDAY_', 'VEGEDA1_', '_FRUTSUM', '_FRUTSUM',
                           'MAXVO2_', 'FC60_', 'STRFREQ_', 'PHYSHLTH', 'IDAY', 'CHILDREN', 'MENTHLTH', 'FRUTDA1_', '_STRWT', '_VEGESUM'],
            
            'not_sure' : ['SEQNO', '_PSU' , '_STSTR', '_STRWT', '_RAWRAKE', ' _WT2RAKE', '_LLCPWT']
           }

def clean_data_x(x_train, labels):
    """
    Cleaning data
    :param x_train: training data
    :return: cleaned data
    """
    #Removing the first label which is the id
    features_number = x_train.shape[1]
    features = list(dict.fromkeys(labels))  
    features = [features[i]  for i in range(features_number)]
    features = {word: index for index, word in enumerate(features)}

    #Removing columns with more than ACCEPTABLE_NAN_PERCENTAGE of NaN values
    mask_nan_columns = [(np.count_nonzero(np.isnan(x_train[:, i]))/x_train.shape[0]) < ACCEPTABLE_NAN_PERCENTAGE for i in range (0, features_number)]
    x_train = x_train[:, mask_nan_columns]
    features = list(dict.fromkeys(labels))  
    features = [features[i]  for i in range (features_number) if mask_nan_columns[i]]
    features = {word: index for index, word in enumerate(features)}
    #Creating features list
    
    print(len(features))
    #We handle the date and rescale some of the features
    x_train = handling_data(x_train, features)

    #We remove the features that are not useful
    features, x_train = remove_features(x_train, ['WEIGHT2', 'HEIGHT3'], features)
   
    x_train, features = handle_correlation(x_train, features)
    print(np.sum(np.isnan(x_train)))
    x_train = apply_pca(x_train)

    return x_train

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

def handle_correlation(x_train, features):
    """
    Handling correlation between features
    :param x_train: training data
    :param features: features list
    :return: modified and correlation handled data
    """

    # Replace NaN in categorical features with the median value
    for feature in category_features['continuous']:
        if feature in features.keys() :
            median_value = np.nanmedian(x_train[:, features[feature]])
            x_train[: ,features[feature]] = np.nan_to_num(x_train[:,features[feature]], nan = median_value)

    # Replace NaN in categorical features with the most frequent (mode) value
    for feature in category_features['categorical']:
    
        if feature in features.keys() :
            values, counts = np.unique(x_train[:, features[feature]], return_counts=True)
            most_represented_class = np.argmax(counts)
            x_train[:,features[feature]] = np.nan_to_num(x_train[:,features[feature]], nan = most_represented_class)
    
    # Remove rows with NaN values
    no_nan_in_row_mask = ~np.isnan(x_train).any(axis=1)
    x_train_no_nan = x_train[no_nan_in_row_mask, :]

    # Compute the correlation matrix
    features_correlation = np.corrcoef(x_train_no_nan, rowvar=False)

    # Find the features that are highly correlated
    correlation_limit = 0.7
    correlation_tuple_list = []
    correlation_list = []

    for i in range(x_train_no_nan.shape[1]) : 
        for j in range(i, x_train_no_nan.shape[1]) : 
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

def apply_pca(x_train):
    """
    Apply PCA to the data
    :param x_train: training data
    :return: pca applied data
    """
    mean = np.nanmean(x_train, axis=0)
    x_tilde = x_train - mean
    print(x_tilde)
    cov_matrix = np.cov(x_tilde, rowvar=False)
    print(cov_matrix)
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)
    print(eigvals)
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]


    num_dimensions = 35
    W = eigvecs[:, 0 : num_dimensions]
    eg = eigvals[0 : num_dimensions]

    x_pca = np.dot(x_tilde, W)
    return x_pca