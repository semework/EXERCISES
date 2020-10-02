#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thursday Oct 1 2020
@author: mulugetasemework
"""
#%% import relevant packages
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math
from scipy import stats
import seaborn as sns

#%% data path

work_path = "/Volumes/MuluData/TDI/"

#%% data importing, from command line

# curl -s  'https://wwwdasis.samhsa.gov/dasis2/teds.htm' | head -n 10
# curl url > /Volumes/MuluData/TDI/tedsa_puf_2017.csv

# %% comment this cell if you wish to see all warnings
import warnings
warnings.filterwarnings("ignore")

#%% parameters and direcotries

# get data from here:
data_path = os.path.join(work_path,'tedsa_puf_2017.csv')

# results and figures saving paths
results_path = os.path.join(work_path,'output')
models_path = os.path.join(results_path,'models')

# create folders if they don't exist
if not os.path.exists(results_path):
    os.makedirs(results_path)

if not os.path.exists(models_path):
    os.makedirs(models_path)
# %% plots house keeping

plt.rcdefaults()
plt.close("all")
np.random.seed(0)
# %% important parameters to play with
sampling_percentage = 1 # how much of the data we would like to use,
# in fractions, upto "1", meaning use all
predict_binary = 0 # change to 1 if the column to be predicted should be binarized
corr_thresh = 0.95# drop one correlated column
arrest_thresh = 0.6 # we somehow decided here 60 %is good outcome
sparsity_thresh = .5 # any column with more than this percentage of redundancy is discarded
# i.e. it has too many redundant values
NaN_thresh = 0.95 # any column with  this percentage of Nans is discarded
# intentionally kept high for this exercise since the few rows
# with data in sparse columns are still important

# %% create a y-label for figures, based on what is predicted,
if predict_binary ==1:
    ylabel_text = "binarized_outcome"
else:
    ylabel_text = "continuous_outcome"

# add sample size to file name for ease of recall
if sampling_percentage < 1:
    ylabel_text = (ylabel_text +"_" + str(Path(data_path).stem) + str(int(math.ceil(sampling_percentage*100))) + "_percent_data_used")
else:
    ylabel_text = (ylabel_text +"_" + str(Path(data_path).stem)  + "_All_data_used")

# %% import data
All_data = pd.read_csv(data_path, low_memory=False, error_bad_lines=False)

#peek at data size
print('\n your data size is:', All_data.shape)

#keep the original data for possible future use, instead of reuploading it
All_data_orig = All_data.copy()

# %% shuffle data to make sure there is no order bias
All_data = All_data.sample(frac=1)

#%% what does the data look like?, just look at the first 5 rows
All_data_view = All_data.head()

#%%  peek at data types
All_data.info()

# get basic statistics
All_data.describe()

#%% relevamt columns
""" These are all of the columns
['ADMYR', 'CASEID', 'STFIPS', 'CBSA2010', 'EDUC', 'MARSTAT', 'SERVICES',
       'DETCRIM', 'NOPRIOR', 'PSOURCE', 'ARRESTS', 'EMPLOY', 'METHUSE',
       'PSYPROB', 'PREG', 'GENDER', 'VET', 'LIVARAG', 'DAYWAIT', 'DSMCRIT',
       'AGE', 'RACE', 'ETHNIC', 'DETNLF', 'PRIMINC', 'SUB1', 'SUB2', 'SUB3',
       'ROUTE1', 'ROUTE2', 'ROUTE3', 'FREQ1', 'FREQ2', 'FREQ3', 'FRSTUSE1',
       'FRSTUSE2', 'FRSTUSE3', 'HLTHINS', 'PRIMPAY', 'FREQ_ATND_SELF_HELP',
       'ALCFLG', 'COKEFLG', 'MARFLG', 'HERFLG', 'METHFLG', 'OPSYNFLG',
       'PCPFLG', 'HALLFLG', 'MTHAMFLG', 'AMPHFLG', 'STIMFLG', 'BENZFLG',
       'TRNQFLG', 'BARBFLG', 'SEDHPFLG', 'INHFLG', 'OTCFLG', 'OTHERFLG',
       'DIVISION', 'REGION', 'IDU', 'ALCDRUG']


As it is meaningless for outcome, we will drop 'CASEID'

    ... although year ('ADMYR') is the same (2017) we leave it in for now to
    demonstrate that our data cleanup function works fine (i.e. will remove it)

"""

All_data = All_data.drop(['CASEID'], axis=1)

#%%
def clean_up_data(All_data, sparsity_thresh, NaN_thresh, corr_thresh):
    # cleanup: NANs and zero standard deviation columns and sparse rows
    All_data = All_data.dropna(axis=1, how='all')
    All_data = All_data.dropna(axis=0, how='all')
    All_data = All_data.loc[:, (All_data != All_data.iloc[0]).any()]
    All_data = All_data.reset_index(drop=True)

    # first  clean up data with sparsity analysis
    # remove columns which are very sparsly populated as they might cause false results
    # such as becoming very important in predictions despite having few real data points
    # column 1 for this data is ID, so it can be repeated
    sparse_cols = [(len(np.unique(All_data.iloc[:,i]))/len(All_data))*100  <
                   int((1-sparsity_thresh)*len(All_data)) for i in range(0, All_data.shape[1])]

    #remove sparse columns (i.e. with too many redundant values)
    All_data = All_data.iloc[:,sparse_cols]

    #remove too-many NaN columns
    non_NaN_cols = [All_data.iloc[:,i].isna().sum() < int(NaN_thresh*len(All_data)) for i in range(All_data.shape[1])]
    All_data = All_data.iloc[:, non_NaN_cols]

    # drop the pesky "Unnamed: 0" column, if exists
    # this happens sometimes depending on which packes are used or the quality of the .CSV file

    unNamedCols =   All_data.filter(regex='Unnamed').columns

    if not unNamedCols.empty:
        All_data = All_data.drop(unNamedCols, axis=1, inplace=True)

    # drop highly correlated columns

        # to dot that, first Create correlation matrix
    corr_matrix = All_data.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find index of feature columns with correlation greater than 0.95
    to_drop = [column  for column in upper.columns if any(upper[column] > corr_thresh)]

    # Drop Marked Features
    All_data.drop(All_data[to_drop], axis=1)

    return All_data

# %% clean up data, such as removing 0 STD and sparse columns
# for this data set, cleaning up will get rid off 'ADMYR', since it is all 2017
All_data_populated = clean_up_data(All_data, sparsity_thresh, NaN_thresh, corr_thresh)

# %%
colNames = list(All_data_populated)
#% Move the most important column to the last column, to make predictions easy

colNames.insert(len(colNames), colNames.pop(colNames.index('ARRESTS')))
All_data_populated.columns = colNames
# %%
""" aggregate data """
#
# let's aggregate it for different analysis, such as ranking by age,gender, etc
# We will be taking sums and median values to as aggregation measures
# Here, most are numeric types.
All_data_populated_agg = All_data_populated.groupby('ALCDRUG').agg({
        'ETHNIC': lambda x:  stats.mode(x)[0],
        'AGE': lambda x: x.median(),
        'ALCDRUG': lambda x: stats.mode(x)[0],
       'EMPLOY': lambda x: stats.mode(x)[0],
        'RACE': lambda x: stats.mode(x)[0],
        'MARSTAT': lambda x: stats.mode(x)[0]})

# %%
plt.close("all")
plt.figure(figsize=(10,8), dpi= 80)
All_data_populated.groupby(['AGE', 'GENDER']).size().unstack().plot(kind='bar',stacked=True)
        # Decorations
plt.title(('AGE Group (x-axis) and Gender (color) \n demographic of the 2017 Treatment Episode Data Set (TEDS) '),
          fontsize=14)
plt.xlabel('AGE group      (1 is 12-14 years,  5 (max) is 25–29 years,  12 is 65 years and older)')
plt.ylabel('Count')
plt.legend(['Unkown/unvailable', 'Male','Female'],loc="best",fontsize='small')
plt.tight_layout()
plt.savefig(os.path.join(results_path, str('Demographics_' + ylabel_text + '.pdf')), bbox_inches = "tight")
plt.show()

#%% lets take some columns out and see correlations

df = All_data_populated[['EDUC', 'MARSTAT',
        'NOPRIOR', 'EMPLOY', 'METHUSE',
        'PSYPROB','GENDER', 'VET', 'DAYWAIT', 'DSMCRIT',
        'AGE', 'RACE', 'ETHNIC', 'SUB1',
        'ROUTE1', 'FREQ1', 'FRSTUSE1',
        'HLTHINS', 'PRIMPAY', 'FREQ_ATND_SELF_HELP',
        'ALCFLG', 'COKEFLG', 'MARFLG', 'HERFLG', 'METHFLG', 'OPSYNFLG',
        'PCPFLG', 'HALLFLG', 'MTHAMFLG', 'AMPHFLG', 'STIMFLG', 'BENZFLG',
        'TRNQFLG', 'BARBFLG', 'SEDHPFLG', 'INHFLG', 'OTCFLG', 'OTHERFLG',
        'IDU', 'ALCDRUG','ARRESTS' ]]

# corr matrix
corr = df.corr()
corr = corr[corr.columns[::-1]]
corr = corr.sort_values(by=['ARRESTS'], ascending=[False ])
# lower triangle mask
mask = np.triu(np.ones_like(corr, dtype=bool))

plt.close("all")
plt.figure(figsize=(12,10), dpi= 80)
# colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

ax = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.title('Correlogram (correlation statistics) of Treatment Episode Data Set (TEDS, 2017) parameters. \n \
          (coefficients sorted by ARRESTS (left column)). \n Order in left vertical axis shows correlation with ARRESTS \n (i.e. Next to ARRESTS themselves, top parameters are highly correlated with arrests) \n HERFLG: Heroin reported at admission',
          fontsize=14)

plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.savefig(os.path.join(results_path, str('Correlogram_' + ylabel_text + '.pdf')), bbox_inches = "tight")
plt.show()
