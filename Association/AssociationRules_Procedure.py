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


claims = pd.read_csv("DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv", sep=",", encoding = 'utf8')
print(claims.head())

prcdrColumns = [ 'ICD9_PRCDR_CD_1', 'ICD9_PRCDR_CD_2', 'ICD9_PRCDR_CD_3','ICD9_PRCDR_CD_4', 'ICD9_PRCDR_CD_5', 'ICD9_PRCDR_CD_6']


drg = pd.read_csv("Drg.csv", sep=",", encoding = 'utf8')
drg['CLM_DRG_CD'] = drg['CLM_DRG_CD'].astype(str)
print(drg.head())


joinClaims = pd.merge(claims, drg, how='left', on='CLM_DRG_CD')
print(joinClaims.head(5))

joinClaims = joinClaims[joinClaims['MDC'].notnull()]
print(joinClaims.shape)

for col in prcdrColumns:
    print('Column {} data type {} has {} unique values'.format(col, joinClaims[col].dtype, len(joinClaims[col].unique())))

claims_subset_df = joinClaims[prcdrColumns]
claims_subset_df = claims_subset_df.reset_index(drop=True)
print(claims_subset_df.shape)


for j in range(1,7):
    claims_subset_df['ICD9_PRCDR_CD_' + str(j)] = claims_subset_df['ICD9_PRCDR_CD_' + str(j)].astype(str)
    claims_subset_df['ICD9_PRCDR_CD_' + str(j)] = claims_subset_df['ICD9_PRCDR_CD_' + str(j)].str.strip().str.replace('.', '')
    claims_subset_df['ICD9_PRCDR_CD_' + str(j)] = claims_subset_df['ICD9_PRCDR_CD_' + str(j)].str.strip().str.replace('NaN', '0')
    claims_subset_df['ICD9_PRCDR_CD_' + str(j)] = claims_subset_df['ICD9_PRCDR_CD_' + str(j)].str.strip().str.replace('nan', '0')

baskets = claims_subset_df.values
print(baskets)
numBaskets = len(baskets)
print(numBaskets)

def support(itemset):
    basketSubset = baskets
    for item in itemset:
        basketSubset = [basket for basket in basketSubset if item in basket]    
    return float(len(basketSubset))/float(numBaskets)

# print(support(['4580','7802']))

df = pd.read_csv("./proc_codes.csv")
proc_codes = df['CODE'].values.tolist()
print(len(proc_codes))

supportItems1 = []
minsupport=0.01
for item in proc_codes:
    itemset=[str(item)]
    if support(itemset)>=minsupport:
        supportItems1.append(str(item))

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


minsupport=0.01
minconfidence=0.5
assocRules=[]
newSupportItems=[]

assocRules, supportItems2 =  aprioriIteration(2,supportItems1,assocRules,newSupportItems,minsupport,minconfidence)

print(assocRules)
print(supportItems2)

assocRules, supportItems3 =  aprioriIteration(3,supportItems2,assocRules,newSupportItems,minsupport,minconfidence)
print(assocRules)
print(supportItems3)







