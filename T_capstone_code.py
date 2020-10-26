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
from scipy import stats
import seaborn as sns
import plotly.io as pio
pio.renderers.default = 'browser'
import plotly.offline
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
from textwrap import wrap
from pandas.plotting import parallel_coordinates
import random
import networkx as nx
import plotly.express as px
#%% data path

work_path = "/Users/mulugetasemework/Dropbox/Excercises"

#%% data importing, from command line

# curl -s  'https://wwwdasis.samhsa.gov/dasis2/teds.htm' | head -n 10
#   https://simplemaps.com/data/us-cities
#   http://goodcsv.com/wp-content/uploads/2020/08/us-states-territories.csv
#   https://developers.google.com/public-data/docs/canonical/states_csv

#%% parameters and direcotries

# get data from here:
data_path = os.path.join(work_path,'tedsa_puf_2017.csv')

# results and figures saving paths
results_path = os.path.join(work_path,'processed_data')
figures_path = os.path.join(work_path,'generated_figures')
# create folders if they don't exist
if not os.path.exists(results_path):
    os.makedirs(results_path)

if not os.path.exists(figures_path):
    os.makedirs(figures_path)

    #%% upload state codes, format and save

state_codes = pd.read_csv(os.path.join(work_path,'state_codes.txt') , sep="|")
state_codes.to_csv(os.path.join(results_path,'state_codes.csv'), index=False)
state_codes = pd.read_csv(os.path.join(results_path,'state_codes.csv'))

# %% plots house keeping

plt.rcdefaults()
plt.close("all")
np.random.seed(0)
# %% important parameters to play with
sampling_percentage = 1 # how much of the data we would like to use,
# in fractions, upto "1", meaning use all
corr_thresh = 0.99# drop one correlated column
arrest_thresh = 0.6 # we somehow decided here 60 %is good outcome
sparsity_thresh = .5 # any column with more than this percentage of redundancy is discarded
# i.e. it has too many redundant values
NaN_thresh = 0.95 # any column with  this percentage of Nans is discarded
# intentionally kept high for this exercise since the few rows
# with data in sparse columns are still important

# %% import data

All_data = pd.read_csv(data_path, low_memory=False, error_bad_lines=False).reset_index(drop=True)

#peek at data size
print('\n your data size is:', All_data.shape)

#keep the original data for possible future use, instead of reuploading it
All_data_orig = All_data.copy()

#%%
def clean_up_data(All_data, sparsity_thresh, NaN_thresh, corr_thresh, bad_value):
    # cleanup: NANs and zero standard deviation columns and sparse rows
    indexer_errors = ['level_0','index']
    for i in indexer_errors:
        if i in All_data.columns:
            All_data = All_data.drop(i, axis=1)
    All_data = All_data.loc[:,~All_data.columns.duplicated()]
    All_data = All_data.dropna(axis=1, how='all')
    All_data = All_data.dropna(axis=0, how='all')
    All_data = All_data.loc[:, (All_data != All_data.iloc[0]).any()]
    All_data = All_data.reset_index(drop=True)


    All_data =  All_data[( All_data != bad_value).all(1)]

    # first  clean up data with sparsity analysis
    # remove columns which are very sparsly populated as they might cause false results
    # such as becoming very important in predictions despite having few real data points
    # column 1 for this data is ID, so it can be repeated
    sparse_cols = [(len(np.unique(All_data.iloc[:,i]))/len(All_data))*100  <
                   int((1-sparsity_thresh)*len(All_data)) for i in range(0, All_data.shape[1])]

    #remove sparse columns (i.e. with too many redundant values)
    All_data = All_data.iloc[:, sparse_cols]

    #remove too-many NaN columns
    non_NaN_cols = [All_data.iloc[:,i].isna().sum() < int(NaN_thresh*len(All_data)) for i in range(All_data.shape[1])]
    All_data = All_data.iloc[:, non_NaN_cols]

    # drop the pesky "Unnamed: 0" column, if exists
    # this happens sometimes depending on which packes are used or the quality of the .CSV file

    unNamedCols =   All_data.filter(regex='Unnamed').columns

    if not unNamedCols.empty:
        for i in unNamedCols:
            if i in All_data.columns:
                All_data = All_data.drop(i, axis=1)
        # All_data4 = All_data.drop(unNamedCols), axis=1, inplace=True)

    # drop highly correlated columns
    cols = All_data.select_dtypes([np.number]).columns
        # to dot that, first Create correlation matrix
    corr_matrix = All_data.reindex(columns=cols).corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find index of feature columns with correlation greater than 0.95
    to_drop = [column  for column in upper.columns if any(upper[column] > corr_thresh)]
    # Drop Marked Features
    All_data.drop(All_data[to_drop], axis=1)

    return All_data
#%%
def chunkIt(seq, num):
    """
    source: https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length
    """
    out = []
    last = seq[0]#0.0
    # if num != 0:
    avg = len(seq) / float(num)

    while last < len(seq):
        # if (int(last) != int(last + avg)):
        this =[[int(last), int(last + avg)]]
        # print(this)
        # if this[0][0] != this[0][1]:
        out.append(this[0])
        last += avg
    return out

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
    demonstrate that our data cleanup function works fine (i.e. it will remove it
    because it is redundant)

"""
All_data = All_data.reset_index(drop=True)
# % clean up data, such as removing 0 STD and sparse columns
# for this data set, cleaning up will get rid off 'ADMYR', since it is all 2017
All_data = All_data.drop(['ADMYR'], axis=1)

#%%
def clean_and_populate(data_to_populate,  data_columns):
    All_data_populated =  clean_up_data(data_to_populate.reindex(columns =  data_columns),
                                        sparsity_thresh, NaN_thresh, corr_thresh,-9)

    l1 = np.array(All_data_populated.STFIPS)
    l2 = np.array(state_codes.STATE)
    ind = abs(l1 - l2[:,None]) <= 3
    inds = np.where((ind.max(0)))[0]
    All_data_populated = All_data_populated.iloc[inds,:].reset_index(drop=True)
    indss = [np.where(state_codes.STATE==x)[0][0] for x in All_data_populated.STFIPS]

    newD = pd.DataFrame(state_codes.loc[indss,['STUSAB', 'STATE_NAME']].values).reset_index(drop=True)
    newD.columns =['STUSAB', 'STATE_NAME']
    # All_data_populated[['STUSAB', 'STATE_NAME']] = newD
    All_data_populated = pd.concat([All_data_populated.reset_index(),newD.reset_index()], axis=1)
    All_data_populated = All_data_populated.loc[:,~All_data_populated.columns.duplicated()]
    colNames = list(All_data_populated)
    #% Move the most important column to the last column, to make predictions easy
    colNames.insert(len(colNames), colNames.pop(colNames.index('ARRESTS')))
    All_data_populated = All_data_populated.reindex(columns=colNames)

    relvCols = ['CBSA',   'STCOU', 'NAME', 'LSAD','POPESTIMATE2017',
    'NPOPCHG2017', 'BIRTHS2017',  'DEATHS2017',  'NATURALINC2017',
           'INTERNATIONALMIG2017',  'DOMESTICMIG2017', 'NETMIG2017','RESIDUAL2017']
    """
            CBSA    Core Based Statistical Area code
            MDIV    Metropolitan Division code
            STCOU   State and county code
            NAME    Name/title of area
            LSAD    Legal/Statistical Area Description
            POPESTIMATE2017     7/1/2017 resident total population estimate
            NPOPCHG2017     Numeric change in resident total population 7/1/2016 to 7/1/2017
            BIRTHS2017      Births in period 7/1/2016 to 6/30/2017
            DEATHS2017      Deaths in period 7/1/2016 to 6/30/2017
            NATURALINC2017  Natural increase in period 7/1/2016 to 6/30/2017
            INTERNATIONALMIG2017    Net international migration in period 7/1/2016 to 6/30/2017
            DOMESTICMIG2017     Net domestic migration in period 7/1/2016 to 6/30/2017
            NETMIG2017          Net migration in period 7/1/2016 to 6/30/2017
            RESIDUAL2017        Residual for period 7/1/2016 to 6/30/2017
    """
    US_CBSA = pd.read_csv(os.path.join(work_path,'cbsa-est2019-alldata.csv'), engine='python')
    US_CBSA = US_CBSA.reindex(columns = relvCols)
    US_CBSA[['city','state']]= pd.DataFrame(US_CBSA.NAME.str.split(',',1).tolist(),
                                      columns = ['city','state'])
    US_CBSA = US_CBSA.drop(['STCOU'], axis=1).reset_index(drop=True)


    not_in_list = list(set(All_data_populated.CBSA2010)-set(US_CBSA.CBSA))
    All_data_populated = All_data_populated[~All_data_populated.CBSA2010.isin(not_in_list)]

    cbsa =   US_CBSA.CBSA.reset_index(drop=True)
    cbsa_pop =  pd.Series(All_data_populated.CBSA2010).reset_index(drop=True)
    ind = [np.where(cbsa==x)[0][0] for x in cbsa_pop]
    newD = US_CBSA.iloc[ind,:].reset_index(drop=True)
    All_data_populated = pd.concat([All_data_populated.reset_index(),newD.reset_index()], axis=1)
    All_data_populated = All_data_populated.loc[:,~All_data_populated.columns.duplicated()]

    #% add lat long
    US_cities_data = pd.read_csv(os.path.join(work_path,'uscities.csv'), low_memory=False, error_bad_lines=False).reset_index(drop=True)

    """
    https://simplemaps.com/data/us-cities
            county_name	   The name of the primary county (or equivalent) that contains the city/town.
            population	   An estimate of the city's urban population. (2018).
            density	       The estimated population per square kilometer.
            incorporated	   TRUE if the place is a city/town. FALSE if the place is
                            just a commonly known name for a populated area.
    """

    US_cities_data = US_cities_data[['city','county_name','lat', 'lng','population','density','incorporated']]
    not_in_list = list(set(All_data_populated.city)-set(US_cities_data.city))
    All_data_populated = All_data_populated[~All_data_populated.city.isin(not_in_list)]
    usCityVals_pop = All_data_populated.city.reset_index(drop=True).str.strip()
    usCityVals = US_cities_data.city.reset_index(drop=True).str.strip()
    ind = [np.where(usCityVals==x)[0][0] for x in usCityVals_pop]

    newD = US_cities_data.loc[ind,[ 'county_name','lat', 'lng','population','density','incorporated']].reset_index(drop=True)

    All_data_populated = clean_up_data(All_data_populated, sparsity_thresh, NaN_thresh, corr_thresh,-9)
    All_data_populated = pd.concat([All_data_populated.reset_index(),newD.reset_index()], axis=1)

    file_n = ('ALL_TEDS_DATA_') + str(All_data_populated.shape[1]) + '_columns'
    All_data_populated.to_csv(os.path.join(results_path, ( file_n+ 'ALL_TEDS_DATA.csv')))

    return All_data_populated

#%% corr to rank important vars
def do_corr(df):
    # corr matrix
    corr = df.corr()
    corr = corr[corr.columns[::-1]]
    corr = corr.sort_values(by=['ARRESTS'], ascending=[False ])
    colls = corr.index

    return corr, colls

# %%
def plot_demographics():
    plt.close("all")
    plt.figure(figsize=(10,8), dpi= 80)
    All_data.groupby(['AGE', 'GENDER']).size().unstack().plot(kind='bar',stacked=True)
            # Decorations
    plt.title(('AGE Group (x-axis) and Gender (color) \n demographic of the 2017 Treatment Episode Data Set (TEDS) '),
              fontsize=14)
    plt.xlabel('AGE group      (1 is 12-14 years,  5 (max) is 25–29 years,  12 is 65 years and older)')
    plt.ylabel('Count')
    plt.legend(['Unkown/unvailable', 'Male','Female'],loc="best",fontsize='small')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, str('Demographics.png')), bbox_inches = "tight")
    plt.show()

#%%
data_dict = [{'MARSTAT': {1: 'Never married',
   2: 'Now married',
   3: 'Separated',
   4: 'Divorced_or_widowed'}},
 {'HERFLG': {0: 'Heroine_not_reported', 1: 'Heroine_rreported'}},
 {'FRSTUSE1': {1: 'FRSTUSE1_11_years_and_under',
   2: 'FRSTUSE1_12_14_years',
   3: 'FRSTUSE1_15_17_years',
   4: 'FRSTUSE1_18_20_years',
   5: 'FRSTUSE1_21_24_years',
   6: 'FRSTUSE1_25_29_years',
   7: 'FRSTUSE1_30_years_and_older'}},
 {'HLTHINS': {1: 'Private insurance_Blue_Cross_Blue_Shield_HMO',
   2: 'Medicaid',
   3: 'Medicare_other_e_g_TRICARE_CHAMPUS',
   4: 'No_HLTHINS'}},
 {'ALCDRUG': {0: 'No_alcohol_or_drugs',
   1: 'Alcohol_only',
   2: 'Other_drugs_only',
   3: 'Alcohol_and_other_drugs'}},
 {'METHUSE': {1: 'METHUSE_Yes', 2: 'No_METHUSE'}},
 {'EDUC': {1: 'EDUC_8_years_or_less',
   2: 'EDUC_9_11_years',
   3: 'EDUC_12_years_or_GED',
   4: 'EDUC_13–_15_years',
   5: 'EDUC_16_years_or_more'}},
 {'ARRESTS': {0: 'No_arrests', 1: 'One_arrest', 2: 'Two_or_more_arrests'}}]
#%%

def remap(data, dict_labels):
    """
    Source: https://www.xspdf.com/help/50571437.html
    """
    for field,values in dict_labels.items():
        data.replace({field:values},inplace=True)
    return data

#%%  basket analyisis data

def create_basket_data(data, basket_data_columns):
    data = data.loc[:,~data.columns.duplicated()]
    data = clean_up_data(data.reindex(columns=colNames), sparsity_thresh,
                         NaN_thresh, corr_thresh, -9)
    basket_data_raw =  data.reindex(columns = basket_data_columns)
    basket_data_raw = clean_up_data(basket_data_raw, sparsity_thresh, NaN_thresh, corr_thresh,-9)
    basket_data_raw.CASEID = basket_data_raw.CASEID.astype('str')

    basket_data = basket_data_raw.copy()
    for i in range(len(data_dict)):
        di = dict(list(data_dict[i].values())[0])
        remap(basket_data[list(data_dict[i].keys())[0]], di)
    basket_data = basket_data.reset_index(drop=True)
    basket_data.to_csv(os.path.join(results_path,'ALL_TEDS_DATA_basket_data.csv'), index=False)

    TEDS_attributes =  pd.DataFrame([[basket_data.CASEID[i],  basket_data.iloc[i,1:].values.tolist()]
                                     for i in range(len(basket_data))])
    TEDS_attributes.columns =['CASEID', 'Attributes']
    TEDS_attributes.to_csv(os.path.join(results_path,'ALL_TEDS_DATA_TEDS_attributes.csv'), index=False)

    return TEDS_attributes, basket_data_raw

#%% association rules
def assoc_rules(TEDS_attributes):
    te = TransactionEncoder()
    te_as_array = te.fit(TEDS_attributes.Attributes).transform(TEDS_attributes.Attributes)
    te_as_array.astype("int")

    df = pd.DataFrame(te_as_array, columns=te.columns_)
    print(df.head())

    frequent_itemsets = apriori(df, min_support = 0.3, max_len = 3, use_colnames=True)
    print("Found frequebt item sets of size: ", frequent_itemsets.shape)

    # frequent_itemsets = apriori(onehot, min_support = 0.001, max_len = 3, use_colnames=True)
    # compute all association rules for frequent_itemsets
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

    rules['lhs items'] = rules['antecedents'].apply(lambda x:len(x) )
    rules[rules['lhs items']>1].sort_values('lift', ascending=False).head()
    # Replace frozen sets with strings
    rules['antecedents_'] = rules['antecedents'].apply(lambda a: ','.join(list(a)))
    rules['consequents_'] = rules['consequents'].apply(lambda a: ','.join(list(a)))
    rules['rule'] = rules.index
    coords = rules[['antecedents_','consequents_','rule']]
    # Transform the DataFrame of rules into a matrix using the lift metric
    pivot = rules[rules['lhs items']>1].pivot(index = 'antecedents_',
                        columns = 'consequents_', values= 'lift')

    all_rules_titles = (rules['antecedents_']  + '   -  predicts -    ' + rules['consequents_'] )
    nx_title_text = (('TEDS data (attributes) basket rule relationships.\n\n Top three rules are : \n') +
        str(all_rules_titles[:3].values))

    confidence=rules[['confidence']].to_numpy()
    support=rules[['support']].to_numpy()
    return rules,coords,pivot,nx_title_text,confidence,support

    for i in range(len(support)):
       support[i] = support[i] + 0.0025 * (random.randint(1,10) - 5)
       confidence[i] = confidence[i] + 0.0025 * (random.randint(1,10) - 5)

def plot_corrs(corr):
    # lower triangle mask
    mask = np.triu(np.ones_like(corr, dtype=bool))

    plt.close("all")
    plt.figure(figsize=(12,10), dpi= 80)
    # colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

    plt.title('Correlogram (correlation statistics) of Treatment Episode Data Set (TEDS, 2017) parameters. \n \
              (coefficients sorted by ARRESTS (left column)). \n Order in left vertical axis shows correlation with ARRESTS \n (i.e. Next to ARRESTS themselves, top parameters are highly correlated with arrests) \n example: HERFLG: Heroin reported at admission',
              fontsize=14)

    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.margins(0.01,0.01)
    plt.savefig(os.path.join(figures_path,'TEDS_correlogram.png'))
    plt.show()

def draw_support_confidence(rules):
    #support and confidence
    plt.close("all")
    plt.figure(figsize=(12,10), dpi= 80)
    sns.scatterplot(x = "support", y = "confidence",
                    size = "lift", data = rules)
    plt.title('Confidence vs support', fontsize = 14)
    plt.margins(0.01,0.01)
    plt.savefig(os.path.join(figures_path,'TEDS_support_confidence.png'))
    plt.show()

def draw_coord(coords):
    # coordinates plot
    plt.close("all")
    plt.figure(figsize=(12,10), dpi= 80)
    parallel_coordinates(coords, 'rule')
    plt.legend([])
    plt.grid(True)
    plt.title('TEDS attributes parallel coordinates plot', fontsize = 14)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path,'TEDS_basket_coordinate_plot.png'))
    plt.show()

def draw_top(TEDS_attributes, top_how_many):
    plt.close("all")
    plt.figure(figsize=(16,16), dpi= 80)
    color = plt.cm.inferno(np.linspace(0,1,top_how_many))
    Items = TEDS_attributes.Attributes.value_counts().head(top_how_many)
    Items.plot.bar(color = color)
    plt.title('Top '+ str(top_how_many) +' Most Frequent Items')
    plt.ylabel('Counts')
    plt.xlabel('Items')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path,'Top_freq_item_sets.png'))
    plt.show()


def draw_heatm(pivot):
    plt.close("all")
    plt.figure(figsize=(12,10), dpi= 80)
    sns.heatmap(pivot, annot = True )
    plt.title('TEDS data (attributes) relationships. \n (Lift >1 indicates prediction is strong)\n', fontsize = 16)
    plt.yticks(rotation=0)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.margins(0.01,0.01)
    plt.savefig(os.path.join(figures_path,'TEDS_basket_heatmap.png'))
    plt.show()


def draw_graph(rules, rules_to_show, title_text):

  """
  source: https://intelligentonlinetools.com/blog/2018/02/10/how-to-create-data-visualization-for-association-rules-in-data-mining/
  """
  G1 = nx.DiGraph()
  color_map=[]
  N = 50
  colors = np.random.rand(N)
  strs=['R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11']

  for i in range (rules_to_show):
    G1.add_nodes_from(["R"+str(i)])
    for a in rules.iloc[i]['antecedents']:
        G1.add_nodes_from([a])
        G1.add_edge(a, "R"+str(i), color=colors[i] , weight = 2)
    for c in rules.iloc[i]['consequents']:
            G1.add_nodes_from([c])
            G1.add_edge("R"+str(i), c, color=colors[i],  weight=2)

  for node in G1:
       found_a_string = False
       for item in strs:
           if node==item:
                found_a_string = True
       if found_a_string:
            color_map.append('yellow')
       else:
            color_map.append('green')

  edges = G1.edges()
  colors = [G1[u][v]['color'] for u,v in edges]
  weights = [G1[u][v]['weight'] for u,v in edges]
  plt.close("all")
  plt.figure(figsize=(10,10), dpi= 80)
  pos = nx.spring_layout(G1, k=16, scale=1)
  nx.draw(G1, pos, node_color = color_map, edge_color=colors, width=weights, font_size=16, with_labels=False)

  for p in pos:  # raise text positions
           pos[p][1] += 0.07

  nx.draw_networkx_labels(G1, pos)
  plt.title("TEDS data (attributes) basket rule relationships.\n%s" % "\n".join(wrap(title_text, width=60)))
  plt.title(title_text, fontsize = 12)
  plt.savefig(os.path.join(figures_path,'TEDS_basket_rules_nx.png'), bbox_inches='tight', pad_inches=1)
  plt.show()

#%%
def px_plot(data, colls, animate_or_not):

    for cl in colls:
        ind = np.where(data[cl].notna())[0]
        data = data.iloc[ind,:]
        print(cl)
        if animate_or_not:
            fig = px.density_mapbox(data, lat='lat',
                                    lon='lng', z=cl, radius=5,
                                    center=dict(lat=0, lon=180),
                                    zoom=0,
                                    animation_frame = cl,
                                    mapbox_style="stamen-terrain")
            filen = (os.path.join(figures_path,('html_files/htmls_animated/TEDS_nx_map_' +
                 str(cl) + '_animated.html')))
        else:
            fig = px.density_mapbox(data, lat='lat',
                                    lon='lng', z=cl, radius=10,
                                    center=dict(lat=0, lon=180),
                                    zoom=0, mapbox_style="stamen-terrain")
            filen = (os.path.join(figures_path,('html_files/TEDS_nx_map_' +
                 str(cl) + '.html')))

        fig.update_layout(
                title_text = ('2017 US state ' + str(cl) + ' <br>(Click legend to toggle traces)'),
                showlegend = True,
                paper_bgcolor ='white',
                # margin={"r":0,"t":0,"l":0,"b":0},
                geo = dict(
                    scope = 'usa',
                    landcolor = 'rgb(217, 217, 217)'),
                mapbox_style="carto-positron",#"carto-positron",
                mapbox_zoom=3, mapbox_center = {"lat": 37.0902, "lon": -95.7129},
            )
# htmkl file
        plotly.offline.plot(fig, filename=filen)
        fig.show()

#%%
##############################################################################
########################   analysis and plots   ##############################
##############################################################################

#%%
plot_demographics()

#%%
colNames = ['EDUC', 'MARSTAT', 'SERVICES', 'DETCRIM',
           'NOPRIOR', 'PSOURCE', 'EMPLOY', 'METHUSE', 'PSYPROB', 'PREG', 'GENDER',
           'VET', 'LIVARAG', 'DAYWAIT', 'DSMCRIT', 'AGE', 'RACE', 'ETHNIC',
           'DETNLF', 'PRIMINC', 'SUB1', 'SUB2', 'SUB3', 'ROUTE1', 'ROUTE2',
           'ROUTE3', 'FREQ1', 'FREQ2', 'FREQ3', 'FRSTUSE1', 'FRSTUSE2', 'FRSTUSE3',
           'HLTHINS', 'PRIMPAY', 'FREQ_ATND_SELF_HELP', 'ALCFLG', 'COKEFLG',
           'MARFLG', 'HERFLG', 'METHFLG', 'OPSYNFLG', 'PCPFLG', 'HALLFLG',
           'MTHAMFLG', 'AMPHFLG', 'STIMFLG', 'BENZFLG', 'TRNQFLG', 'BARBFLG',
           'SEDHPFLG', 'INHFLG', 'OTCFLG', 'OTHERFLG', 'DIVISION', 'REGION', 'IDU',
           'ALCDRUG',  'ARRESTS',
           'POPESTIMATE2017', 'NPOPCHG2017', 'BIRTHS2017', 'DEATHS2017',
           'NATURALINC2017', 'INTERNATIONALMIG2017', 'DOMESTICMIG2017',
           'NETMIG2017', 'density', 'incorporated']

df = clean_up_data(All_data.reindex(columns=colNames), sparsity_thresh, NaN_thresh, corr_thresh, -9)
corr, colls = do_corr(df)

#%%


All_data_populated  = clean_and_populate(All_data, All_data.columns)

basket_data_columns = ["CBSA2010", "STFIPS","CASEID","MARSTAT","HERFLG", "FRSTUSE1","HLTHINS",
                                                         "ALCDRUG", "METHUSE", "EDUC", "ARRESTS"]

All_data_populated_basket = clean_and_populate(All_data, basket_data_columns)

TEDS_attributes, basket_data_raw = create_basket_data(All_data_populated_basket, basket_data_columns[2:])

rules, coords, pivot, nx_title_text, confidence, support = assoc_rules(TEDS_attributes)

########################            plots       ##############################
##############################################################################
plot_corrs(corr)
draw_support_confidence(rules)
draw_coord(coords)
top_how_many = 10
draw_top(TEDS_attributes, top_how_many)
draw_heatm(pivot)
draw_graph(rules, top_how_many, nx_title_text)

#%%%  px plot
colors = ["royalblue","crimson","lightseagreen","orange","lightgrey",
          'darkorchid','darkred','darksalmon','darkseagreen',
            'darkslateblue','darkslategray','darkslategrey',
            'darkturquoise','darkviolet','deeppink','deepskyblue',
            'dimgray','dimgrey','dodgerblue','firebrick',
            'floralwhite','forestgreen','fuchsia','gainsboro',
            'ghostwhite','gold','goldenrod','gray','grey','green',
            'greenyellow','honeydew','hotpink','indianred, indigo']
#px plots
colls = ['MARSTAT', 'HERFLG','FRSTUSE1', 'HLTHINS', 'ALCDRUG', 'METHUSE', 'EDUC',
       'ARRESTS', 'POPESTIMATE2017',
       'NPOPCHG2017', 'BIRTHS2017', 'DEATHS2017', 'NATURALINC2017',
       'INTERNATIONALMIG2017', 'DOMESTICMIG2017', 'NETMIG2017', 'RESIDUAL2017',
       'population', 'density']

colls_for_animation = ['MARSTAT', 'HERFLG','FRSTUSE1', 'HLTHINS', 'ALCDRUG', 'METHUSE', 'EDUC',
       'ARRESTS', 'population', 'density']

#%%
animate_or_not = 0
px_plot(All_data_populated_basket, colls, animate_or_not)

animate_or_not = 1
px_plot(All_data_populated_basket, colls_for_animation, animate_or_not)
