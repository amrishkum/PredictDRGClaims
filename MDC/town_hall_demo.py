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

logPath = "./tb_logs"
features = 2500


def read_claims_data():
    claims1 = pd.read_csv("DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv", sep=",", encoding = 'utf8')
    # claims2 = pd.read_csv("DE1_0_2008_to_2010_Inpatient_Claims_Sample_2.csv", sep=",", encoding = 'utf8')
    claims = pd.concat([claims1])
    print(claims.shape)
    summary1 = pd.read_csv("DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv", sep=",", encoding = 'utf8')
    summary2 = pd.read_csv("DE1_0_2009_Beneficiary_Summary_File_Sample_1.csv", sep=",", encoding = 'utf8')
    summary3 = pd.read_csv("DE1_0_2010_Beneficiary_Summary_File_Sample_1.csv", sep=",", encoding = 'utf8')
    # summary4 = pd.read_csv("DE1_0_2008_Beneficiary_Summary_File_Sample_2.csv", sep=",", encoding = 'utf8')
    # summary5 = pd.read_csv("DE1_0_2009_Beneficiary_Summary_File_Sample_2.csv", sep=",", encoding = 'utf8')
    # summary6 = pd.read_csv("DE1_0_2010_Beneficiary_Summary_File_Sample_2.csv", sep=",", encoding = 'utf8')
    summary = pd.concat([summary1, summary2, summary3])
    print(summary.shape)
    claimsSummary = pd.merge(claims, summary, how='inner', on='DESYNPUF_ID')
    print(claimsSummary.shape)
    drg = pd.read_csv("Drg.csv", sep=",", encoding = 'utf8')
    drg['CLM_DRG_CD'] = drg['CLM_DRG_CD'].astype(str)
    print(drg.shape)
    joinClaims = pd.merge(claimsSummary, drg, how='left', on='CLM_DRG_CD')
    print(joinClaims.shape)
    joinClaims = joinClaims[joinClaims['MDC'].notnull()]
    print(joinClaims.shape)
    joinClaims1 = joinClaims[joinClaims['MDC'].str.contains('MDC-03|MDC-05|MDC-04|MDC-01') == True]
    print(joinClaims1.shape)
    joinClaims1 =  joinClaims1[joinClaims1.CLM_DRG_CD.str.contains("OTH") == False]
    joinClaims1 =  joinClaims1[joinClaims1.CLM_DRG_CD.str.match("0") == False] 
    targetName = 'MDC'
    targetSeries = joinClaims[targetName]
    #remove target from current location and insert in collum 0
    del joinClaims[targetName]
    joinClaims.insert(0, targetName, targetSeries)
    print(joinClaims.shape)
    targetSeries = joinClaims1[targetName]
    del joinClaims1[targetName]
    joinClaims1.insert(0, targetName, targetSeries)
    print(joinClaims1.shape)

    interestingColumns = ['MDC', 'ADMTNG_ICD9_DGNS_CD','ICD9_DGNS_CD_1', 'ICD9_DGNS_CD_2',
        'ICD9_DGNS_CD_3', 'ICD9_DGNS_CD_4', 'ICD9_DGNS_CD_5', 'ICD9_DGNS_CD_6',
        'ICD9_DGNS_CD_7', 'ICD9_DGNS_CD_8', 'ICD9_DGNS_CD_9', 'ICD9_DGNS_CD_10',
        'ICD9_PRCDR_CD_1', 'ICD9_PRCDR_CD_2', 'ICD9_PRCDR_CD_3',
        'ICD9_PRCDR_CD_4', 'ICD9_PRCDR_CD_5', 'ICD9_PRCDR_CD_6', 'BENE_SEX_IDENT_CD', 'BENE_RACE_CD', 'BENE_BIRTH_DT','SP_STATE_CODE']
    interestingColumns1 = ['MDC', 'CLM_DRG_CD','ADMTNG_ICD9_DGNS_CD','ICD9_DGNS_CD_1', 'ICD9_DGNS_CD_2',
        'ICD9_DGNS_CD_3', 'ICD9_DGNS_CD_4', 'ICD9_DGNS_CD_5', 'ICD9_DGNS_CD_6',
        'ICD9_DGNS_CD_7', 'ICD9_DGNS_CD_8', 'ICD9_DGNS_CD_9', 'ICD9_DGNS_CD_10',
        'ICD9_PRCDR_CD_1', 'ICD9_PRCDR_CD_2', 'ICD9_PRCDR_CD_3',
        'ICD9_PRCDR_CD_4', 'ICD9_PRCDR_CD_5', 'ICD9_PRCDR_CD_6', 'BENE_SEX_IDENT_CD', 'BENE_RACE_CD', 'BENE_BIRTH_DT','SP_STATE_CODE']
    dgnColumns = ['ADMTNG_ICD9_DGNS_CD', 'ICD9_DGNS_CD_1', 'ICD9_DGNS_CD_2',
        'ICD9_DGNS_CD_3', 'ICD9_DGNS_CD_4', 'ICD9_DGNS_CD_5', 'ICD9_DGNS_CD_6',
        'ICD9_DGNS_CD_7', 'ICD9_DGNS_CD_8', 'ICD9_DGNS_CD_9', 'ICD9_DGNS_CD_10']
    prcdrColumns = [ 'ICD9_PRCDR_CD_1', 'ICD9_PRCDR_CD_2', 'ICD9_PRCDR_CD_3',
        'ICD9_PRCDR_CD_4', 'ICD9_PRCDR_CD_5', 'ICD9_PRCDR_CD_6']
    demoColumns = ['BENE_SEX_IDENT_CD', 'BENE_RACE_CD', 'BENE_BIRTH_DT','SP_STATE_CODE']

    for col in interestingColumns:
        print('Column {} data type {} has {} unique values'.format(col, joinClaims[col].dtype, len(joinClaims[col].unique())))

    claims_subset_df = joinClaims[dgnColumns + prcdrColumns]
    claims_subset_df = claims_subset_df.reset_index(drop=True)
    print(claims_subset_df.shape)
    demo_subset_df = joinClaims[demoColumns]
    print(demo_subset_df.shape)
    demo_subset_df = demo_subset_df.reset_index(drop=True)
    demo_subset_df.fillna(0)
    combineClaims_subset = pd.concat([claims_subset_df, demo_subset_df], axis=1)
    print(combineClaims_subset.shape)
    claims_subset_df.fillna(0)

    claims_sets = joinClaims1[interestingColumns1]
    claims_sets.fillna(0)
    claims_sets['ADMTNG_ICD9_DGNS_CD'] = claims_sets['ADMTNG_ICD9_DGNS_CD'].astype(str)
    claims_sets['ADMTNG_ICD9_DGNS_CD'] = claims_sets['ADMTNG_ICD9_DGNS_CD'].str.strip().str.replace('.', '')
    claims_sets['ADMTNG_ICD9_DGNS_CD'] = claims_sets['ADMTNG_ICD9_DGNS_CD'].str.strip().str.replace('NaN', '0')
    claims_sets['ADMTNG_ICD9_DGNS_CD'] = claims_sets['ADMTNG_ICD9_DGNS_CD'].str.strip().str.replace('nan', '0')
    for j in range(1,11):
        claims_sets['ICD9_DGNS_CD_' + str(j)] = claims_sets['ICD9_DGNS_CD_' + str(j)].astype(str)
        claims_sets['ICD9_DGNS_CD_' + str(j)] = claims_sets['ICD9_DGNS_CD_' + str(j)].str.strip().str.replace('.', '')
        claims_sets['ICD9_DGNS_CD_' + str(j)] = claims_sets['ICD9_DGNS_CD_' + str(j)].str.strip().str.replace('NaN', '0')
        claims_sets['ICD9_DGNS_CD_' + str(j)] = claims_sets['ICD9_DGNS_CD_' + str(j)].str.strip().str.replace('nan', '0')
    for j in range(1,7):
        claims_sets['ICD9_PRCDR_CD_' + str(j)] = claims_sets['ICD9_PRCDR_CD_' + str(j)].astype(str)
        claims_sets['ICD9_PRCDR_CD_' + str(j)] = claims_sets['ICD9_PRCDR_CD_' + str(j)].str.strip().str.replace('.', '')
        claims_sets['ICD9_PRCDR_CD_' + str(j)] = claims_sets['ICD9_PRCDR_CD_' + str(j)].str.strip().str.replace('NaN', '0')
        claims_sets['ICD9_PRCDR_CD_' + str(j)] = claims_sets['ICD9_PRCDR_CD_' + str(j)].str.strip().str.replace('nan', '0')
    claims_sets = claims_sets.reset_index(drop=True)
    claims_sets.fillna(0)
    interestingColumns2 = ['MDC', 'ACTUAL_DRG_GROUPER','ADMIT_DIAGNOSIS_CODE','ICD9_DIAGNOSIS_CODE_1', 'ICD9_DIAGNOSIS_CODE_2',
        'ICD9_DIAGNOSIS_CODE_3', 'ICD9_DIAGNOSIS_CODE_4', 'ICD9_DIAGNOSIS_CODE_5', 'ICD9_DIAGNOSIS_CODE_6',
        'ICD9_DIAGNOSIS_CODE_7', 'ICD9_DIAGNOSIS_CODE_8', 'ICD9_DIAGNOSIS_CODE_9', 'ICD9_DIAGNOSIS_CODE_10',
        'ICD9_PROCEDURE_CODE_1', 'ICD9_PROCEDURE_CODE_2', 'ICD9_PROCEDURE_CODE_3', 'ICD9_PROCEDURE_CODE_4', 'ICD9_PROCEDURE_CODE_5', 'ICD9_PROCEDURE_CODE_6', 
        'SEX', 'RACE', 'BIRTH_DATE','STATE_CODE']
    claims_sets.columns = interestingColumns2
    claims_sets.to_csv("test.csv",index=False)

    # print(claims_subset_df.dtypes)
    demo_subset_df = demo_subset_df.astype('int64', copy=False)

    # valid_subset_df = joinClaims[interestingColumns]
    # print(valid_subset_df.head())

    train_df_c = pd.read_pickle("train_mdc.pkl.compress", compression="gzip")
    print(train_df_c.shape)
    train_df_new = pd.concat([train_df_c,demo_subset_df ], axis=1)
    print(train_df_new.shape) 
    uniqueDrgs = targetSeries.unique()
    print(len(uniqueDrgs))
    n_class = len(uniqueDrgs)
    # train_df_new.replace(np.Inf, np.nan)
    # train_df_new.replace(-np.Inf, np.nan)
    #train_df_new.fillna(0)
    print(train_df_new.shape)
    pca_file = open("pca_mdc_1.pkl", "rb")
    pca = pickle.load(pca_file)    
    x = train_df_new.values.tolist()
    Y = targetSeries.values.tolist()
    X = pca.transform(x)
    pca_file.close()
    train_x,test_x,train_y,test_y = train_test_split(X,Y,test_size=0.30, random_state=42)
    print(test_x.shape)
    return test_x,test_y,n_class

def getPredictedValue(testX,n_class):
    feature_size = testX.shape[1]
    print(feature_size)

    with tf.name_scope("input_parameters"):
        feature_size=2500
        n_class = 25

    with tf.name_scope("Claims_Data_Input"):
        x = tf.placeholder(tf.float32,[None,feature_size], name ="diag_proc_demo_data")

    with tf.name_scope('weights'):
        def weight_variable(feature_size, n_class, name=None):
            initial = tf.truncated_normal([feature_size, n_class])
            weights = tf.Variable(initial, name=name)
            return weights

    with tf.name_scope('biases'):
        def bias_variable(n_class, name=None):
            initial = tf.truncated_normal([n_class])
            biases = tf.Variable(initial, name=name)
            return biases

    with tf.name_scope('mdc_neural_network'):
        weights = weight_variable(feature_size, n_class, name='Weight')
        biases = bias_variable(n_class, name='biases')
        logits = tf.matmul(x, weights) + biases

    with tf.name_scope("predicted_mdc"):
        y_conv = tf.nn.softmax(logits=logits)

    saver = tf.train.Saver()
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, "/tmp/model_mdc.ckpt")
        prediction = tf.argmax(y_conv, axis = 1)
        return prediction.eval(feed_dict={x: testX}, session=sess)
    

testX,testY,n_class = read_claims_data()
print(testX.shape)
predY_encoded = getPredictedValue(testX,n_class)
label_file = open("label_1.pkl", "rb")
encoder = pickle.load(label_file)
predY = encoder.inverse_transform(predY_encoded)
label_file.close()
# print(testY)
# print(predY)
testY_encoded = encoder.transform(testY)
score = accuracy_score(testY_encoded, predY_encoded)
print('test accuracy: {:.4f}'.format(score))
score = accuracy_score(testY, predY)
print('test accuracy: {:.4f}'.format(score))


