# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 16:01:28 2020

@authors: Kevin Söderberg and Anders Wrethem
"""
import pandas as pd
from sklearn import preprocessing

import plaidml.keras
import os
plaidml.keras.install_backend()
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import keras
from keras import regularizers
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras import regularizers
from sklearn.metrics import accuracy_score

def create_3d_plot(data, target, figsize, class_list):
    plt.clf()
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    for c, dispo, a in class_list:
        tmp_df = data[data[target] == dispo]
        xs = tmp_df['X']
        ys = tmp_df['Y']
        zs = tmp_df['Z']
        ax.scatter(xs, ys, zs, s=50, alpha=a, edgecolors='w', c=c)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # ax.view_init(30, 90)
    
    plt.show()

def create_autoencoder(data, batch = 16, nr_epochs = 100, dim=2):
    X = data
    y = data[data.columns[0]]
    n_cols = X.shape[1]
    
    #Train-Test split
    from sklearn.model_selection import train_test_split as tts
    X_train, X_test, y_train, y_test = tts(X, y, test_size = 0.1, random_state = 0)
    
    input_img = Input(shape=(n_cols,))
    encoded = Dense(35, activation='relu')(input_img)
    encoded = Dense(30, activation='relu')(encoded)
    encoded = Dense(25, activation='relu')(encoded)
    encoded = Dense(20, activation='relu')(encoded)
    encoded = Dense(15, activation='relu')(encoded)
    encoded = Dense(10, activation='relu')(encoded)
    encoded = Dense(5, activation='relu')(encoded)
    
    encoded = Dense(dim, activation='relu')(encoded)

    decoded = Dense(5, activation='relu')(encoded)
    decoded = Dense(10, activation='relu')(decoded)
    decoded = Dense(15, activation='relu')(decoded)
    decoded = Dense(20, activation='relu')(decoded)
    decoded = Dense(25, activation='relu')(decoded)
    decoded = Dense(30, activation='relu')(decoded)
    decoded = Dense(35, activation='relu')(decoded)
    decoded = Dense(n_cols)(decoded)
    
    # Autoencoder, the entire network
    autoencoder = Model(input_img, decoded)
    
    #The encoder & decoder parts of the network
    encoder = Model(input_img, encoded)
    
    # create a placeholder for an encoded (2-dimensional) input
    encoded_input = Input(shape=(dim,))

    deco = autoencoder.layers[-8](encoded_input)
    deco = autoencoder.layers[-7](deco)
    deco = autoencoder.layers[-6](deco)
    deco = autoencoder.layers[-5](deco)
    deco = autoencoder.layers[-4](deco)
    deco = autoencoder.layers[-3](deco)
    deco = autoencoder.layers[-2](deco)
    deco = autoencoder.layers[-1](deco)
    decoder = Model(encoded_input, deco)
    
    # Train the entire Autoencoder
    autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
    autoencoder.fit(X_train, X_train,
                epochs=nr_epochs,
                batch_size=batch,
                shuffle=True,
                validation_data=(X_test, X_test), verbose = 1)
    
    # Test reconstruction
    #encoded_imgs = encoder.predict(X_test)    
    #decoded_imgs = decoder.predict(encoded_imgs)
    
    return encoder, decoder, autoencoder

def fill_median(data):
    for column in data.columns:
        print("Current columns: ", column)
        tmp = data[column].dtypes
        
        if tmp == 'int64' or tmp == 'float64':
            print("Number of NaN: ", data[column].isna().sum())
            print("Total length: ", len(data[column]))
            median = data[column].median()
            data[column] = data[column].fillna(median)
            
    return data
            

data = pd.read_csv('cumulative.csv')

# Removing these columns due to no information for model building
drop_columns = ['rowid', 'kepid', 'kepoi_name', 'kepler_name', 'koi_score']
data_cleaned = data.drop(drop_columns, axis = 1)

# Removing these columns due to being 100% NaN
data_cleaned = data_cleaned.drop(['koi_teq_err1', 'koi_teq_err2'], axis = 1)
data_cleaned.drop(['koi_tce_delivname'], axis = 1, inplace=True)

# Drop rows that have four or more NaN values
rows_nan = data_cleaned.isna().sum(axis=1)
rows_nan = rows_nan[rows_nan >= 4]
rows_nan = rows_nan.reset_index()
data_cleaned = data_cleaned.drop(index = rows_nan["index"].to_numpy(), axis = 0)

# Seperate False Positives from Candidates and Confirmed
# This is to run seperate Data Cleaning processes on the sets
data_fp = data_cleaned[data_cleaned['koi_disposition'] == 'FALSE POSITIVE']
data_cc = data_cleaned[data_cleaned['koi_disposition'] != 'FALSE POSITIVE']
data_fp = data_fp.dropna(axis = 0) # Drop every row that contains atleast one NaN
data_cc = fill_median(data_cc) # Fill each NaN with the median of the column

# Merge the two datasets back together after cleaning
data_merged = pd.concat([data_fp, data_cc], axis = 0)

# Filter out the samples that went 'False Positive' -> 'Confirmed'
mask = (data_merged['koi_disposition'] == 'CONFIRMED') & (data_merged['koi_pdisposition'] == 'FALSE POSITIVE')
data_merged['koi_disposition'][mask] = 'FP_CONFIRMED'

# Checking what type of data processing is suitable for the dataset
# data_merged['koi_period'].plot()

# data_period = pd.DataFrame(data_merged['koi_period'])

# data_period.drop(index=341, inplace=True)

# # Normalize / Scale each column
# minmax = preprocessing.MinMaxScaler()
# data_period['Scaled'] = minmax.fit_transform(data_period['koi_period'].values.reshape(-1, 1))

# robust = preprocessing.RobustScaler()
# data_period['Robust'] = robust.fit_transform(data_period['koi_period'].values.reshape(-1, 1))

# removing one extreme outlier
data_merged.drop(index=341, inplace=True)

# Reset index to get back ordinary range 0 - len(dataframe)
data_merged.reset_index(inplace=True, drop=True)

# Selected Standardization as the first attempt
standard = preprocessing.StandardScaler()
standardized_data = standard.fit_transform(data_merged[data_merged.columns[2:]]) # Numpy Array
minmax = preprocessing.MinMaxScaler()
standardized_data = minmax.fit_transform(standardized_data)

standardized_columns = data_merged.columns[2:] # List with column names
data_split = pd.DataFrame(data = standardized_data, columns = standardized_columns)

data_split_y = data_merged[data_merged.columns[0:2]]

data_split_merged = pd.concat([data_split_y, data_split], axis=1)

data_split_merged[data_split_merged.columns[2:]].plot(figsize = (12,8))

# data_merged[data_merged.columns[2:-1]].plot(figsize=(12,8))

# Visualize the data in 2D using PCA
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
pca = PCA(n_components = 2)

data_pca = pd.DataFrame(pca.fit_transform(data_split), columns = ['X', 'Y'])

data_pca = pd.concat([data_split_y, data_pca], axis = 1)

lda = LinearDiscriminantAnalysis(n_components = 2)
data_split_lda = data_split_y.replace({'FALSE POSITIVE': 0, 'CANDIDATE': 1, 'CONFIRMED': 2, 'FP_CONFIRMED': 3})

data_lda = pd.DataFrame(lda.fit_transform(data_split, data_split_lda['koi_disposition']), columns = ['X', 'Y'])
data_lda = pd.concat([data_split_y, data_lda], axis = 1)

lda = LinearDiscriminantAnalysis(n_components = 2)
data_lda_org = pd.DataFrame(lda.fit_transform(data_merged[data_merged.columns[2:]], data_split_lda['koi_disposition']), columns = ['X', 'Y'])
data_lda_org = pd.concat([data_split_y, data_lda_org], axis = 1)

# data_split_y.koi_disposition[data_lda.koi_disposition == 'False Positive'] = 0
# data_split_y.koi_disposition[data_lda.koi_disposition == 'Candidate'] = 1
# data_split_y.koi_disposition[data_lda.koi_disposition == 'Confirmed'] = 2

import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
sns.lmplot(x='X', y='Y', hue = 'koi_disposition', data=data_lda, scatter_kws={'alpha':0.3}, size=10)
sns.lmplot(x='X', y='Y', hue = 'koi_disposition', data=data_lda_org, scatter_kws={'alpha':0.3}, size=10)
sns.lmplot(x='X', y='Y', hue = 'koi_disposition', data=data_pca, scatter_kws={'alpha':0.3}, size=12)

pca = PCA(n_components = 3)

data_pca_3 = pd.DataFrame(pca.fit_transform(data_split), columns = ['X', 'Y', 'Z'])

data_pca_3 = pd.concat([data_split_y, data_pca_3], axis = 1)

dispo_list = data_pca_3['koi_disposition'].unique()


classes = [('b', 'FALSE POSITIVE', 0.025), ('r', 'CONFIRMED', 0.8), ('g', 'CANDIDATE', 0.8), ('purple', 'FP_CONFIRMED', 1)]
create_3d_plot(data_pca_3,'koi_disposition',(16,12),classes)

classes = [('r', 'CONFIRMED', 0.6), ('g', 'CANDIDATE', 0.6), ('purple', 'FP_CONFIRMED', 1)]
create_3d_plot(data_pca_3,'koi_disposition',(16,12),classes)

encoder_3d = keras.models.load_model('3D_encoder.h5')
enc_3d2 = encoder_3d.predict(data_split)

enc_df_3d = data_split_y
enc_df_3d['X'] = enc_3d2[:,0]
enc_df_3d['Y'] = enc_3d2[:,1]
enc_df_3d['Z'] = enc_3d2[:,2]

classes = [('b', 'FALSE POSITIVE', 0.025), ('r', 'CONFIRMED', 0.8), ('g', 'CANDIDATE', 0.8), ('purple', 'FP_CONFIRMED', 1)]
create_3d_plot(enc_df_3d,'koi_disposition',(16,12),classes)

encoder = keras.models.load_model('2D_encoder.h5')
enc_data = encoder.predict(data_split)
enc_df = data_split_y
enc_df["X"] = enc_data[:,0]
enc_df["y"] = enc_data[:,1]

sns.lmplot(x='X', y='y', hue = 'koi_disposition', data=enc_df, scatter_kws={'alpha':0.3}, size=12)


# IsolationForest on 'Confirmed', run on whole dataset
data_iso_con = data_split_merged
test_iso_con = data_split_merged
from sklearn.ensemble import IsolationForest
iso_con = IsolationForest(random_state=0, n_estimators=300,verbose=1, n_jobs=16, contamination=0.5, max_features=1)
iso_con.fit(data_iso_con[data_iso_con.columns[2:]])
iso_result= iso_con.predict(test_iso_con[test_iso_con.columns[2:]])

result_series = pd.Series(iso_result)
test_iso_3d = enc_df_3d
test_iso_3d.reset_index(inplace=True, drop=True)
test_iso_3d["Isolation"] = result_series
classes = [('b', -1, 0.3), ('r', 1, 0.8)]
create_3d_plot(test_iso_3d,'Isolation',(16,12),classes)
# IsolationsForest on 'False Positive', compare on 'FP_CONFIRMED'