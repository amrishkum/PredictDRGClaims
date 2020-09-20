#Add packages
#These are my standard packages I load for almost every project
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.utils import  shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
import pickle
from sklearn.manifold import TSNE
import csv
#From Scikit Learn
from sklearn import preprocessing
from sklearn.model_selection  import train_test_split, cross_val_score, KFold
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import time
import re
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction import text 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.preprocessing import LabelEncoder
from scipy import sparse
from functools import partial


claims1 = pd.read_csv("DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv", sep=",", encoding = 'utf8')
claims2 = pd.read_csv("DE1_0_2008_to_2010_Inpatient_Claims_Sample_2.csv", sep=",", encoding = 'utf8')
claims = pd.concat([claims1])
print(claims.head())

dgnColumns = ['ICD9_DGNS_CD_1', 'ICD9_DGNS_CD_2',
       'ICD9_DGNS_CD_3', 'ICD9_DGNS_CD_4', 'ICD9_DGNS_CD_5', 'ICD9_DGNS_CD_6',
       'ICD9_DGNS_CD_7', 'ICD9_DGNS_CD_8', 'ICD9_DGNS_CD_9', 'ICD9_DGNS_CD_10']


# drg = pd.read_csv("Drg.csv", sep=",", encoding = 'utf8')
# drg['CLM_DRG_CD'] = drg['CLM_DRG_CD'].astype(str)
# print(drg.head())


# joinClaims = pd.merge(claims, drg, how='left', on='CLM_DRG_CD')
# print(joinClaims.head(5))

# joinClaims = joinClaims[joinClaims['MDC'].notnull()]
# print(joinClaims.shape)

for col in dgnColumns:
    print('Column {} data type {} has {} unique values'.format(col, claims[col].dtype, len(claims[col].unique())))

claims_subset_df = claims[dgnColumns]
claims_subset_df = claims_subset_df.reset_index(drop=True)
print(claims_subset_df.shape)

claims_subset_df = claims_subset_df.fillna('0')

baskets = claims_subset_df.values.tolist()
print(len(baskets))
baskets_cleaned = []
for i in range(len(baskets)):
    bas = []
    for j in range(len(baskets[i])):
        if baskets[i][j] != '0':
            bas.append(baskets[i][j])
    if bas:
        baskets_cleaned.append(bas)
print(len(baskets_cleaned))
numBaskets = len(baskets_cleaned)
print(numBaskets)

def support(itemset):
    basketSubset = baskets_cleaned
    for item in itemset:
        basketSubset = [basket for basket in basketSubset if item in basket]    
    return float(len(basketSubset))/float(numBaskets)

print(support(['4580','7802']))

df = pd.read_csv("./diag_codes.csv")
diag_codes = df['CODE'].values.tolist()
print(len(diag_codes))

supportItems1 = []
minsupport=0.0001

for index, item in enumerate(diag_codes): 
    itemset=[item]
    if support(itemset)>=minsupport:
        supportItems1.append(item)

missing_items =[]
for i in range(len(baskets_cleaned)):
    for j in range(len(baskets_cleaned[i])):
        if baskets_cleaned[i][j] not in diag_codes:
            if baskets_cleaned[i][j] not in missing_items:
                missing_items.append(baskets_cleaned[i][j])

print(missing_items)
for index, item in enumerate(missing_items):
    itemset=[item]
    if support(itemset)>=minsupport:
        supportItems1.append(item)

print(supportItems1)

import itertools

def aprioriIteration(i,supportItems,assocRules,newSupportItems,minsupport,minconfidence):
    
    for itemset in itertools.combinations(supportItems,i): 
        itemset = list(itemset)
        if support(itemset)>minsupport:
             for j in range(i):
                rule_to = itemset[j]
                rule_from = [x for x in itemset if x!=itemset[j]]
                confidence=support(itemset)/support(rule_from) 
                if confidence>minconfidence:
                    assocRules.append((rule_from,rule_to))
                    for x in itemset:
                        if x not in newSupportItems:
                            newSupportItems.append(x)
    return assocRules, newSupportItems


minsupport=0.0001
minconfidence=0.7
assocRules=[]
newSupportItems=[]

assocRules, supportItems2 =  aprioriIteration(2,supportItems1,assocRules,newSupportItems,minsupport,minconfidence)

print(assocRules)
print(supportItems2)

assocRules, supportItems3 =  aprioriIteration(3,supportItems2,assocRules,newSupportItems,minsupport,minconfidence)
print(assocRules)
print(supportItems3)





