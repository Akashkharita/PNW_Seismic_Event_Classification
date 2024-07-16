import numpy as np
import pandas as pd
from glob import glob 
from tqdm import tqdm
import seaborn as sns 

# for converting the text file containing the quarry locations into csv file
import csv

# for computing the geographical distance between two points 
import math


from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, auc, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score
from datetime import datetime
import h5py
from sklearn.preprocessing import LabelEncoder
from scipy import stats, signal
from sklearn.preprocessing import StandardScaler
import obspy
from obspy.geodetics.base import gps2dist_azimuth, gps2dist_azimuth
from obspy.clients.fdsn import Client
import time
pd.set_option('display.max_columns', None)
from joblib import dump, load
from obspy.signal.filter import envelope
import tsfel


import sys
sys.path.append('../feature_extraction_scripts/physical_feature_extraction_scripts')
import seis_feature

sys.path.append('../src')

#from seis_feature import compute_physical_features
from tsfel import time_series_features_extractor, get_features_by_domain
from datetime import timedelta
import os


from utils import apply_cosine_taper
from utils import butterworth_filter
from utils import plot_confusion_matrix
from utils import plot_classification_report
from utils import calculate_distance


import pickle
from zenodo_get import zenodo_get

#import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse

# These waveforms are filtered between 1-10 Hz
# extracting features of surface events, thunder and sonic booms





parser = argparse.ArgumentParser(description="Seismic Event Classification Script")
parser.add_argument("--filename", type=str, required=True, help="Filename to process")


args = parser.parse_args()

filename = args.filename

## Loading tsfel features. 
print('Loading tsfel features')



features_exotic_tsfel = pd.read_csv('../extracted_features/tsfel_features_surface event_'+filename+'_part_1.csv')

# features of noise
features_noise_tsfel = pd.read_csv('../extracted_features/tsfel_features_noise_'+filename+'_part_1.csv')


# features of explosion
features_explosion_tsfel = pd.read_csv('../extracted_features/tsfel_features_explosion_'+filename+'_part_1.csv')


# features of earthquake (had to extract it in four parts because of memory constraints)
features_eq1 = pd.read_csv('../extracted_features/tsfel_features_earthquake_'+filename+'_part_1.csv')
features_eq2 = pd.read_csv('../extracted_features/tsfel_features_earthquake_'+filename+'_part_2.csv')
features_eq3 = pd.read_csv('../extracted_features/tsfel_features_earthquake_'+filename+'_part_3.csv')
features_eq4 = pd.read_csv('../extracted_features/tsfel_features_earthquake_'+filename+'_part_4.csv')

# features of earthquakes
features_earthquake_tsfel = pd.concat([features_eq1, features_eq2, features_eq3, features_eq4])



print(f'No. of tsfel features: {features_exotic_tsfel.shape[1]-2}')
print('In addition to 390 features, it contains two more columns, serial no. and source type')

print('Loading physical features')
# extracting features of surface events, thunder and sonic booms
features_exotic_physical = pd.read_csv('../extracted_features/physical_features_surface event_'+filename+'_part_1.csv')

features_surface_physical = features_exotic_physical[features_exotic_physical['source'] == 'surface event']
features_sonic_physical = features_exotic_physical[features_exotic_physical['source'] == 'sonic']
features_thunder_physical = features_exotic_physical[features_exotic_physical['source'] == 'thunder']



# features of noise
features_noise_physical = pd.read_csv('../extracted_features/physical_features_noise_'+filename+'_part_1.csv')


# features of explosion
features_explosion_physical = pd.read_csv('../extracted_features/physical_features_explosion_'+filename+'_part_1.csv')

# features of earthquakes
features_eq1 = pd.read_csv('../extracted_features/physical_features_earthquake_'+filename+'_part_1.csv')
features_eq2 = pd.read_csv('../extracted_features/physical_features_earthquake_'+filename+'_part_2.csv')
features_eq3 = pd.read_csv('../extracted_features/physical_features_earthquake_'+filename+'_part_3.csv')
features_eq4 = pd.read_csv('../extracted_features/physical_features_earthquake_'+filename+'_part_4.csv')

features_earthquake_physical = pd.concat([features_eq1, features_eq2, features_eq3, features_eq4])

print(f'No. of physical features: {features_exotic_physical.shape[1]-2}')




print('Merging physical and tsfel features')

features_noise = pd.merge(features_noise_physical, features_noise_tsfel, on = ['serial_no', 'source'])
features_earthquake = pd.merge(features_earthquake_physical, features_earthquake_tsfel, on = ['serial_no', 'source'])
features_explosion = pd.merge(features_explosion_physical, features_explosion_tsfel, on = ['serial_no', 'source'])


features_surface_tsfel = features_exotic_tsfel[features_exotic_tsfel['source'] == 'surface event']
features_sonic_tsfel = features_exotic_tsfel[features_exotic_tsfel['source'] == 'sonic']
features_thunder_tsfel = features_exotic_tsfel[features_exotic_tsfel['source'] == 'thunder']

features_surface_physical = features_exotic_physical[features_exotic_physical['source'] == 'surface event']
features_sonic_physical = features_exotic_physical[features_exotic_physical['source'] == 'sonic']
features_thunder_physical = features_exotic_physical[features_exotic_physical['source'] == 'thunder']


features_surface = pd.merge(features_surface_physical, features_surface_tsfel, on = ['serial_no', 'source'])
features_sonic = pd.merge(features_sonic_physical, features_sonic_tsfel, on = ['serial_no', 'source'])
features_thunder = pd.merge(features_thunder_physical, features_thunder_tsfel, on = ['serial_no', 'source'])



features_all = pd.concat([features_surface, features_sonic, features_thunder, features_noise, features_explosion, features_earthquake])


tsfel_features = features_exotic_tsfel.columns[:-2]
physical_features = features_exotic_physical.columns[:-2]
final_features_list = np.concatenate([tsfel_features,physical_features,['source', 'serial_no']])

features_all = features_all.loc[:, final_features_list]
print(f'So we have {features_all.shape[0]} events and each event have {features_all.shape[1]} features')



print('Removing highly correlated redundant features with a threshold of 0.95')
serial_nos = features_all['serial_no'].values
features_all = features_all.drop(['Unnamed: 0_x','Unnamed: 0_y', 'source_x', 'serial_no'], axis = 1, errors = 'ignore')
features_all.rename(columns={'source_y': 'source'}, inplace=True)
corr_features = tsfel.correlated_features(features_all.iloc[:, 1:453])

features_all.drop(corr_features, axis=1, inplace=True)
features_all['serial_no'] = serial_nos
print(f'So we have {features_all.shape[0]} events and each event have {features_all.shape[1]} features')




print('Dropping the columns (features) that contain infinity or nan values for any event')

# dropping the columns that contain NaNs
features_all = features_all.dropna(axis = 1)

# dropping the rows that contains NaNs
features_all = features_all.dropna()


## dropping all the rows containing infinity values
features_all = features_all.replace([np.inf, -np.inf], np.nan).dropna()


## dropping sonic boom and thunder events
features_all = features_all[features_all['source'] != 'sonic']
features_all = features_all[features_all['source'] != 'thunder']

print(f'So we have {features_all.shape[0]} events and each event have {features_all.shape[1]} features')



print(f'Removing the features that have same values for each event')
# Check unique values in each column
unique_counts = features_all.nunique()

# Identify columns with only one unique value (same value for all rows)
single_value_columns = unique_counts[unique_counts == 1].index


# Drop columns with the same value for all rows
features_all = features_all.drop(columns=single_value_columns)

print(f'So we have {features_all.shape[0]} events and each event have {features_all.shape[1]} features')



print(f'Removing the outliers based on the Z score')
df = features_all.drop(['serial_no', 'source'], axis = 1)
# Calculate Z-scores for each feature
z_scores = np.abs(stats.zscore(df))


# Define a threshold for Z-score beyond which data points are considered outliers
threshold = 10

# Filter out rows with any Z-score greater than the threshold
# Temporarily removing this 
outliers_removed_df =   features_all[(z_scores < threshold).all(axis=1)]

print(f'So we have {outliers_removed_df.shape[0]} events and each trace have {outliers_removed_df.shape[1]} features')



print('Standardizing the features')

## defining the global variables X and y
X = outliers_removed_df.drop(['serial_no','source'], axis = 1)
y = outliers_removed_df['source']


# Initialize the StandardScaler
scaler = StandardScaler()



# Apply standard scaling to the DataFrame
scaled_features = scaler.fit_transform(X)


# Access the mean and standard deviation for each feature
means = scaler.mean_
std_devs = scaler.scale_

# Create a DataFrame to display the means and standard deviations
scaler_params = pd.DataFrame({'Feature': X.columns, 'Mean': means, 'Std Dev': std_devs})
print(scaler_params)


# Create a new DataFrame with scaled features
X_scaled = pd.DataFrame(scaled_features, columns=X.columns)


## We are not standardizing at this stage. We will rather wait when the outlier are removed, then we will
## standardize and save the standard scaler parameters. 
#X_scaled = X

X_scaled['serial_no'] = outliers_removed_df['serial_no'].values
X_scaled['source'] = outliers_removed_df['source'].values





print('saving the scalar params')
scaler_params.to_csv('../results/scaler_params_'+filename+'.csv', index = False)



print('Merging the metadata information with the events')

# extracting the stored data
comcat_file_name = h5py.File("/data/whd01/yiyu_data/PNWML/comcat_waveforms.hdf5",'r')
exotic_file_name = h5py.File("/data/whd01/yiyu_data/PNWML/exotic_waveforms.hdf5",'r')
noise_file_name = h5py.File("/data/whd01/yiyu_data/PNWML/noise_waveforms.hdf5",'r')


# extracting the catalog
comcat_file_csv = pd.read_csv("/data/whd01/yiyu_data/PNWML/comcat_metadata.csv")
exotic_file_csv = pd.read_csv("/data/whd01/yiyu_data/PNWML/exotic_metadata.csv")
noise_file_csv = pd.read_csv("/data/whd01/yiyu_data/PNWML/noise_metadata.csv")



# extracting the metadata corresponding to individual events
cat_exp = comcat_file_csv[comcat_file_csv['source_type'] == 'explosion']
cat_eq = comcat_file_csv[comcat_file_csv['source_type'] == 'earthquake']
cat_no = noise_file_csv
cat_su = exotic_file_csv[exotic_file_csv['source_type'] == 'surface event']


# extracting the index 
ind_exp = X_scaled[X_scaled['source'] == 'explosion']['serial_no'].values
ind_eq = X_scaled[X_scaled['source'] == 'earthquake']['serial_no'].values
ind_no = X_scaled[X_scaled['source'] == 'noise']['serial_no'].values
ind_su = X_scaled[X_scaled['source'] == 'surface event']['serial_no'].values

df_exp = X_scaled[X_scaled['source'] == 'explosion']
exp_df = cat_exp.iloc[ind_exp]
exp_df['serial_no'] = ind_exp

df_eq = X_scaled[X_scaled['source'] == 'earthquake']
eq_df = cat_eq.iloc[ind_eq]
eq_df['serial_no'] = ind_eq

df_no = X_scaled[X_scaled['source'] == 'noise']
no_df = cat_no.iloc[ind_no]
no_df['serial_no'] = ind_no

df_su = X_scaled[X_scaled['source'] == 'surface event']
su_df = cat_su.iloc[ind_su]
su_df['serial_no'] = ind_su



new_exp = pd.merge(df_exp,exp_df, on = 'serial_no')
new_eq = pd.merge(df_eq,eq_df, on = 'serial_no')
new_su = pd.merge(df_su,su_df, on = 'serial_no')
new_no = pd.merge(df_no,no_df, on = 'serial_no')
new_no['event_id'] = np.array(['noise'+str(i) for i in np.arange(len(new_no))])



X_final = pd.concat([new_exp, new_eq, new_su, new_no])
y = ['explosion']*len(new_exp)+['earthquake']*len(new_eq)+['surface']*len(new_su)+['noise']*len(new_no)


print('Adding manual features such as HOD, DOW, and MOY')

# new_exp contains the features and the corresponding metadata information. 
datetimes = X_final['trace_start_time'].values

hour_of_day = []
days_of_week = []
month_of_year = []
for dt_str in tqdm(datetimes):
        
    # Parse the datetime string
        dt = datetime.strptime(dt_str, '%Y-%m-%dT%H:%M:%S.%fZ')
        hod = dt.hour - 8.  # converting to local time. 
        moy = dt.month
        
        
        days_of_week.append(dt.weekday())
        hour_of_day.append(hod)
        month_of_year.append(moy)
        
X_final['hour_of_day'] = hour_of_day
X_final['day_of_week'] = days_of_week
X_final['month_of_year'] = month_of_year



print('Hyperparameter Tuning using 10 fold cross validation over 50 iterations on the dataset that contains 3000 random samples per class.')


temp_X = X_final.iloc[:,0:int(np.where(X_final.columns == 'serial_no')[0])]
#temp_X = temp_X.assign(hod=X_final['hour_of_day'].values, dow=X_final['day_of_week'].values, moy=X_final['month_of_year'].values)

## X_final is already standardized. 

# Apply standard scaling to the DataFrame
scaled_features =  temp_X  #scaler.fit_transform(temp_X)

# Create a new DataFrame with scaled features
temp_X = pd.DataFrame(scaled_features, columns= temp_X.columns)






# Apply random undersampling using imbalanced-learn library
rus = RandomUnderSampler(sampling_strategy={'earthquake':3000, 'explosion':3000, 'surface':3000, 'noise':3000})
X_resampled, y_resampled = rus.fit_resample(temp_X, y)

# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Fit the LabelEncoder on the text labels and transform them to numeric labels
y_num = label_encoder.fit_transform(y_resampled)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_num, test_size=0.2, stratify = y_num)



# Define the hyperparameter grid for randomized search
param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False]
}

# Initialize the Random Forest model
rf_model = RandomForestClassifier(random_state=42)

# Initialize the RandomizedSearchCV with 5-fold cross-validation
random_search = RandomizedSearchCV(
    rf_model, param_distributions=param_dist, n_iter=50, scoring='f1_macro', cv=10, verbose=0, random_state=42, n_jobs=-1
)

# Perform randomized grid search cross-validation
random_search.fit(X_train, y_train)

# Print the best parameters and their corresponding accuracy score
print("Parameters considered:", param_dist)
print("Best Parameters:", random_search.best_params_)
print("Best Accuracy:", random_search.best_score_)

# Evaluate the best model on the test set
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = np.sum(y_pred == y_test) / len(y_test)
print("Test Accuracy:", accuracy)


## saving the best model in the disk..
print('saving the best model in the disk')
dump(best_model, '../results/best_rf_model_all_features_'+filename+'.joblib')


print('loading the best model from the disk')
bf = load('../results/best_rf_model_all_features_'+filename+'.joblib')


print('Plotting and saving confusion matrix and classification report for the best model from hyperparameter tuning')
# Reload the module
import importlib
import utils
importlib.reload(utils)
from utils import plot_confusion_matrix
from utils import plot_classification_report


## plotting confusion matrix
cm = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm, figure_name = '../figures/confusion_matrix_hyp_tuning_'+filename+'.png')


# Calculate the classification report
report = classification_report(y_test, y_pred, output_dict=True)
plot_classification_report(report, figure_name = '../figures/classification_report_hyp_tuning_'+filename+'.png')






print('doing testing and training over all of the data')

print('Training the model on 5000 randomly selected events per class (Not traces)')

## extracting metadata information for each kind of source along with features
## source_type_pnsn_label is more reliable label. 


# selecting all the earthquakes and their metadata information. 
a_eq = X_final[X_final['source_type_pnsn_label'] == 'eq']



## removing the ambiguous events, these are the events that were assigned as earthquakes by pnsn but labelled as
## explosion in USGS ANSS catalog. 
a_eq = a_eq[a_eq['source_type'] != 'explosion']


## selecting all the explosions specifically px, (which is mainly quarry blasts)
a_px = X_final[X_final['source_type_pnsn_label'] == 'px']
# removing the ambiguous events. 
a_px = a_px[a_px['source_type'] != 'earthquake']

a_su = X_final[X_final['source_type'] == 'surface event']
a_no = X_final[X_final['source_type'] == 'noise']



# Extract event IDs for each source type
eq_ids, px_ids, su_ids, no_ids = (
    np.unique(a['event_id'].values) for a in [a_eq, a_px, a_su, a_no]
)



## Specifying a random seed for the reproducibility. 
np.random.seed(123) 


## randomizing along the time. 
r1 = np.random.randint(0, len(eq_ids), 5000)
train_eq = eq_ids[r1]

## randomizing along the time. 
r2 = np.random.randint(0, len(px_ids), 5000)
train_px = px_ids[r2]

## randomizing along the time. 
r3 = np.random.randint(0, len(su_ids), 5000)
train_su = su_ids[r3]

## randomizing along the time
r4 = np.random.randint(0, len(no_ids), 5000)
train_no = no_ids[r4]



mask_eq = np.ones(eq_ids.shape, dtype = bool)
mask_eq[r1] = False

mask_px = np.ones(px_ids.shape, dtype = bool)
mask_px[r2] = False

mask_su = np.ones(su_ids.shape, dtype = bool)
mask_su[r3] = False

mask_no = np.ones(no_ids.shape, dtype = bool)
mask_no[r4] = False

test_eq = eq_ids[mask_eq]
test_px = px_ids[mask_px]
test_su = su_ids[mask_su]
test_no = no_ids[mask_no]



# concatenating training ids
all_train_ids = np.concatenate([train_eq,train_px, train_su, train_no])

# concatenating testing ids
all_test_ids = np.concatenate([test_eq,test_px, test_su, test_no])

# allocating event id as index
X_final.index = X_final['event_id'].values


# extracting training and testing values
X_train = X_final.loc[all_train_ids]
X_test = X_final.loc[all_test_ids]


Y_train = X_train['source_type'].values
Y_test = X_test['source_type'].values



print('Computing performance of physical+tsfel features without adding manual features')
# Case 1: without adding anything manual
## Check the performance 
x_train = X_train.iloc[:, 0:int(np.where(X_train.columns == 'serial_no')[0])]
#x_train = x_train.assign(hod=X_train['hour_of_day'].values, dow=X_train['day_of_week'].values, moy=X_train['month_of_year'].values)

x_test = X_test.iloc[:, 0:int(np.where(X_train.columns == 'serial_no')[0])]
#x_test = x_test.assign(hod=X_test['hour_of_day'].values, dow=X_test['day_of_week'].values, moy=X_test['month_of_year'].values)




print('Training our model on 5000 events per class')
# initiating a random undersampler
# we have also specified a random state for reproducibility


# number of samples per each event. 
nus = 5000
rus = RandomUnderSampler(sampling_strategy={'earthquake':nus, 'explosion':nus,'surface event':nus,'noise':nus}, random_state = 42)

# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Converting the textual labels into numerical labels
y_num_test = label_encoder.fit_transform(Y_test)


# randomly taking 5000 samples per class from the training dataset
X_resampled, y_resampled = rus.fit_resample(x_train, Y_train)


# Fit the LabelEncoder on the text labels and transform them to numeric labels
y_num_res = label_encoder.fit_transform(y_resampled)



best_model.class_weight  = None
best_model.fit(X_resampled, y_num_res)


print('plotting and saving trace wise performance based on tsfel + physical features')
y_pred = best_model.predict(x_test)
plt.style.use('seaborn')
trace_cm_phy_tsf = confusion_matrix(y_num_test, y_pred)
plot_confusion_matrix(trace_cm_phy_tsf, figure_name = '../figures/trace_conf_matrix_phy_tsfel_'+filename+'.png')


# Calculate the classification report
trace_report_phy_tsf = classification_report(y_num_test, y_pred, output_dict=True)
plot_classification_report(trace_report_phy_tsf, figure_name = '../figures/trace_class_report_phy_tsfel_'+filename+'.png')



print('plotting and saving event wise performance based on tsfel+physical features')
probs_all = best_model.predict_proba(x_test)

X_test['labelled'] = y_num_test
X_test['classified'] = y_pred
X_test['eq_probability'] = probs_all[:,0]
X_test['px_probability'] = probs_all[:,1]
X_test['no_probability'] = probs_all[:,2]
X_test['su_probability'] = probs_all[:,3]


mean_labels = X_test.groupby('event_id').mean()['labelled'].values
mean_ids = X_test.groupby('event_id').mean().index.values



mean_eq_prob = X_test.groupby('event_id').mean()['eq_probability'].values
mean_px_prob = X_test.groupby('event_id').mean()['px_probability'].values
mean_no_prob = X_test.groupby('event_id').mean()['no_probability'].values
mean_su_prob = X_test.groupby('event_id').mean()['su_probability'].values



temp_class = np.argmax(np.vstack([mean_eq_prob, mean_px_prob, mean_no_prob, mean_su_prob]), axis = 0)
temp_probs = np.max(np.vstack([mean_eq_prob, mean_px_prob, mean_no_prob, mean_su_prob]), axis = 0)



cf_events_phy_tsf = confusion_matrix(mean_labels, temp_class)
plot_confusion_matrix(cf_events_phy_tsf,  figure_name = '../figures/event_conf_matrix_phy_tsfel_'+filename+'.png')


# Calculate the classification report
report_event_phy_tsf = classification_report(mean_labels, temp_class, output_dict=True)
plot_classification_report(report_event_phy_tsf,  figure_name = '../figures/event_class_report_phy_tsfel_'+filename+'.png')





print('Including manual features HOD, DOW and MOY in addition to physical+tsfel features')
# Case 1: without adding anything manual
## Check the performance 
x_train = X_train.iloc[:, 0:int(np.where(X_train.columns == 'serial_no')[0])]
x_train = x_train.assign(hod=X_train['hour_of_day'].values, dow=X_train['day_of_week'].values, moy=X_train['month_of_year'].values)

x_test = X_test.iloc[:, 0:int(np.where(X_train.columns == 'serial_no')[0])]
x_test = x_test.assign(hod=X_test['hour_of_day'].values, dow=X_test['day_of_week'].values, moy=X_test['month_of_year'].values)


print('Model is being trained on 5000 traces per class')
# initiating a random undersampler
rus = RandomUnderSampler(sampling_strategy={'earthquake':5000, 'explosion':5000,'surface event':5000,'noise':5000}, random_state = 42)

# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Converting the textual labels into numerical labels
y_num_test = label_encoder.fit_transform(Y_test)


# randomly taking 5000 samples per class from the training dataset
X_resampled, y_resampled = rus.fit_resample(x_train, Y_train)


# Fit the LabelEncoder on the text labels and transform them to numeric labels
y_num_res = label_encoder.fit_transform(y_resampled)



best_model.class_weight  = None
best_model.fit(X_resampled, y_num_res)


print('Plotting and saving trace wise performance for physical, tsfel and manual features')
## Plotting the confusion matrix
y_pred = best_model.predict(x_test)
plt.style.use('seaborn')
trace_cm_phy_tsf_man = confusion_matrix(y_num_test, y_pred)
plot_confusion_matrix(trace_cm_phy_tsf_man,  figure_name = '../figures/trace_conf_matrix_phy_tsfel_man_'+filename+'.png')



# Plotting the classification report
trace_report_phy_tsf_man = classification_report(y_num_test, y_pred, output_dict=True)
plot_classification_report(trace_report_phy_tsf_man,  figure_name = '../figures/trace_class_report_phy_tsfel_man_'+filename+'.png')



print('Plotting and saving event wise performance for physical+tsfel+manual features')

probs_all = best_model.predict_proba(x_test)

X_test['labelled'] = y_num_test
X_test['classified'] = y_pred
X_test['eq_probability'] = probs_all[:,0]
X_test['px_probability'] = probs_all[:,1]
X_test['no_probability'] = probs_all[:,2]
X_test['su_probability'] = probs_all[:,3]


mean_labels = X_test.groupby('event_id').mean()['labelled'].values
mean_ids = X_test.groupby('event_id').mean().index.values



mean_eq_prob = X_test.groupby('event_id').mean()['eq_probability'].values
mean_px_prob = X_test.groupby('event_id').mean()['px_probability'].values
mean_no_prob = X_test.groupby('event_id').mean()['no_probability'].values
mean_su_prob = X_test.groupby('event_id').mean()['su_probability'].values



## Assigning an event class based on the maximum average probability across the stations. 
temp_class = np.argmax(np.vstack([mean_eq_prob, mean_px_prob, mean_no_prob, mean_su_prob]), axis = 0)
## Computing the maximum averaged probability. 
temp_probs = np.max(np.vstack([mean_eq_prob, mean_px_prob, mean_no_prob, mean_su_prob]), axis = 0)



## Plotting the confusion matrix
cf_events_phy_tsf_man = confusion_matrix(mean_labels, temp_class)
#cf_norm = cf_events/np.sum(cf_events, axis = 1, keepdims = True)
plot_confusion_matrix(cf_events_phy_tsf_man,  figure_name = '../figures/event_conf_matrix_phy_tsfel_man_'+filename+'.png')



# Plotting the classification report
report_event_phy_tsf_man = classification_report(mean_labels, temp_class, output_dict=True)
plot_classification_report(report_event_phy_tsf_man,   figure_name = '../figures/event_class_report_phy_tsfel_man_'+filename+'.png')


print('saving individual results')
# Saving every result into disk

# Saving trace results

## physical + tsfel
# Save to a file
with open('../results/trace_report_phy_tsf_'+filename+'.pkl', 'wb') as pickle_file:
    pickle.dump(trace_report_phy_tsf, pickle_file)

    
# Save to a file
with open('../results/trace_confusion_matrix_phy_tsf_'+filename+'.pkl', 'wb') as pickle_file:
    pickle.dump(trace_cm_phy_tsf, pickle_file)
    
    

# Saving event results

with open('../results/event_report_phy_tsf_'+filename+'.pkl', 'wb') as pickle_file:
    pickle.dump(report_event_phy_tsf, pickle_file)

    

with open('../results/event_confusion_matrix_phy_tsf_'+filename+'.pkl', 'wb') as pickle_file:
    pickle.dump(cf_events_phy_tsf, pickle_file)

    
    
## physical + tsfel + manual
    
    
# Save to a file
with open('../results/trace_report_phy_tsf_man_'+filename+'.pkl', 'wb') as pickle_file:
    pickle.dump(trace_report_phy_tsf_man, pickle_file)

    
# Save to a file
with open('../results/trace_confusion_matrix_phy_tsf_man_'+filename+'.pkl', 'wb') as pickle_file:
    pickle.dump(trace_cm_phy_tsf_man, pickle_file)
    
    

# Saving event results

with open('../results/event_report_phy_tsf_man_'+filename+'.pkl', 'wb') as pickle_file:
    pickle.dump(report_event_phy_tsf_man, pickle_file)

    

with open('../results/event_confusion_matrix_phy_tsf_man_'+filename+'.pkl', 'wb') as pickle_file:
    pickle.dump(cf_events_phy_tsf_man, pickle_file)
    
    
print('Saving trained model')
dump(best_model, '../results/best_rf_model_all_features_'+filename+'.joblib')



















print('Computing and Plotting Feature Importances')
num_iter = 10
f_imp = []

for i in tqdm(range(num_iter)):

    # Apply random undersampling using imbalanced-learn library
    rus = RandomUnderSampler(sampling_strategy={'earthquake':3000, 'explosion':3000, 'surface':3000, 'noise':3000})
    X_resampled, y_resampled = rus.fit_resample(temp_X, y)

    # Initialize the LabelEncoder
    label_encoder = LabelEncoder()

    # Fit the LabelEncoder on the text labels and transform them to numeric labels
    y_num = label_encoder.fit_transform(y_resampled)


    # Split the data into training and testing sets
    samp_X_train, samp_X_test, samp_y_train, samp_y_test = train_test_split(X_resampled, y_num, test_size=0.2, stratify = y_num)

    # Perform randomized grid search cross-validation
    rf_model.fit(samp_X_train, samp_y_train)


    f_imp.append(rf_model.feature_importances_)


final_imp = np.mean(f_imp, axis = 0)
features = X_train.columns.values


# Create a boolean list where each element indicates if the corresponding feature starts with '0_'
ind_str = [features[j][0:2] == '0_' for j in range(len(features))]

# Initialize a numpy array with the color 'darkred' for each feature, specifying a string length of 8
feature_colors = np.array(['darkred'] * len(features), dtype='<U8')

# Update the color to 'darkblue' for features that start with '0_'
feature_colors[ind_str] = 'darkblue'

feature_names = features



# Set rc parameters for font size
plt.rcParams['xtick.labelsize'] = 16  # Font size for xtick labels
plt.rcParams['ytick.labelsize'] = 20  # Font size for ytick labels
# Sample feature importances and feature labels
feature_importances = final_imp
feature_labels = features

# Sort feature importances and feature labels together
sorted_indices = sorted(range(len(feature_importances)), key=lambda k: feature_importances[k], reverse=True)
sorted_feature_importances = [feature_importances[i] for i in sorted_indices]
sorted_feature_labels = [feature_labels[i] for i in sorted_indices]
colors = [feature_colors[i] for i in sorted_indices]

# Plotting
plt.figure(figsize=(20, 15))
bars = plt.barh(sorted_feature_labels[0:50], sorted_feature_importances[0:50])

# Color bars to match yticklabels
for bar, color in zip(bars, colors):
    bar.set_color(color)

# Color yticklabels and increase font size
for label, color in zip(plt.gca().get_yticklabels(), colors):
    label.set_color(color)
    #label.set_fontsize(20)  # Set desired font size here

# Create legend handles and labels
legend_handles = [plt.Rectangle((0,0),1,1, color='darkblue', ec='black'), plt.Rectangle((0,0),1,1, color='darkred', ec='black')]
legend_labels = ['Tsfel Features', 'Physical Features']

plt.legend(legend_handles, legend_labels, title='Features', title_fontsize=20, frameon=True, fontsize = 20, facecolor='white', edgecolor='black')

plt.xlabel('Feature Importance', fontsize=20)
plt.ylabel('Feature', fontsize=20)
#plt.title('Top 50 Feature Importances', fontsize=20)
plt.gca().invert_yaxis()  # Invert y-axis to display highest importance at the top

# Set y-axis tick label size
plt.yticks(fontsize=14)

plt.savefig('../figures/Feature_Importances_'+filename+'_.png')





""""
print('Computing performance variation with cumulative number of most important features in the strides of 10')

## based on the X_train and X_test computed previously. They should contain 2400 and 600 events per class respectively. 
results_dict = []
selected_features = []

for i in tqdm(range(1, len(sorted_feature_labels), 10)):
    selected_features = sorted_feature_labels[0:i]
    X_temp_train = samp_X_train[selected_features].copy()
    X_temp_test = samp_X_test[selected_features].copy()
    bf.fit(X_temp_train, y_train)
    
    y_pred = bf.predict(X_temp_test)
    results_dict.append(classification_report(y_test, y_pred, output_dict=True))

    
# Define the labels for surface events, explosions, and earthquakes
labels = ['3', '1', '0']

# Define a function to extract metrics based on the label and metric type
def extract_metric(results_dict, label, metric):
    return [results_dict[i][label][metric] for i in range(len(results_dict))]

# Extract accuracy, f1, precision, and recall for individual and group assessment
acc_features = [results_dict[i]['accuracy'] for i in range(len(results_dict))]
f1_features, prec_features, rec_features = (
    extract_metric(results_dict, 'macro avg', metric)
    for metric in ['f1-score', 'precision', 'recall']
)

# Extract f1, precision, and recall for surface events, explosions, and earthquakes
f1_su, prec_su, rec_su = (extract_metric(results_dict, '3', metric) for metric in ['f1-score', 'precision', 'recall'])
f1_exp, prec_exp, rec_exp = (extract_metric(results_dict, '1', metric) for metric in ['f1-score', 'precision', 'recall'])
f1_eq, prec_eq, rec_eq = (extract_metric(results_dict, '0', metric) for metric in ['f1-score', 'precision', 'recall'])


# Create a figure and axis for the main plot
fig, ax = plt.subplots(figsize=[12, 8])



cb_palette = ['#1f77b4','#9467bd',   '#d62728',  '#2ca02c',  ]


# Main plot
ax.plot(np.arange(1, len(sorted_feature_labels), 10), f1_features, marker='o', label='F1-score (Average)', color='k', linestyle='-')
ax.plot(np.arange(1, len(sorted_feature_labels), 10), f1_su, marker='o', label='F1-score (Surface)', color= '#2ca02c', linestyle='-')
ax.plot(np.arange(1, len(sorted_feature_labels), 10), f1_eq, marker='o', label='F1-score (Earthquake)', color='#1f77b4', linestyle='-')
ax.plot(np.arange(1, len(sorted_feature_labels), 10), f1_exp, marker='o', label='F1-score (Explosion)', color= '#9467bd', linestyle='-')
#ax.plot(np.arange(1, len(sorted_feature_labels), 10), f1_features, marker='o', label='F1-score', color=colors(0), linestyle='-')


#ax.plot(np.arange(len(f1_features)), acc_features, marker='s', label='Accuracy', color=colors(1), linestyle='--')
#ax.plot(np.arange(len(f1_features)), prec_features, marker='^', label='Precision', color=colors(2), linestyle='-.')
#ax.plot(np.arange(len(f1_features)), rec_features, marker='d', label='Recall', color=colors(3), linestyle=':')
ax.legend(fontsize= 20, loc='lower right')
ax.set_xlabel('Cumulative number of most important features', fontsize=25)
ax.set_ylabel('Performance', fontsize=25)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.grid(True, linestyle='--', alpha=0.5)
#ax.set_title('Performance Metrics vs. Number of Features', fontsize=25)



plt.tight_layout()
plt.savefig('../figures/Performance_with_cumulative_features_'+filename+'.png')

"""

