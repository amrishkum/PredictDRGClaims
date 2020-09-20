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
import warnings
warnings.filterwarnings('ignore')

logPath = "./tb_logs"
features = 2500
n_class = 25

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical features as a numeric array.
    The input to this transformer should be a matrix of integers or strings,
    denoting the values taken on by categorical (discrete) features.
    The features can be encoded using a one-hot aka one-of-K scheme
    (``encoding='onehot'``, the default) or converted to ordinal integers
    (``encoding='ordinal'``).
    This encoding is needed for feeding categorical data to many scikit-learn
    estimators, notably linear models and SVMs with the standard kernels.
    Read more in the :ref:`User Guide <preprocessing_categorical_features>`.
    Parameters
    ----------
    encoding : str, 'onehot', 'onehot-dense' or 'ordinal'
        The type of encoding to use (default is 'onehot'):
        - 'onehot': encode the features using a one-hot aka one-of-K scheme
          (or also called 'dummy' encoding). This creates a binary column for
          each category and returns a sparse matrix.
        - 'onehot-dense': the same as 'onehot' but returns a dense array
          instead of a sparse matrix.
        - 'ordinal': encode the features as ordinal integers. This results in
          a single column of integers (0 to n_categories - 1) per feature.
    categories : 'auto' or a list of lists/arrays of values.
        Categories (unique values) per feature:
        - 'auto' : Determine categories automatically from the training data.
        - list : ``categories[i]`` holds the categories expected in the ith
          column. The passed categories are sorted before encoding the data
          (used categories can be found in the ``categories_`` attribute).
    dtype : number type, default np.float64
        Desired dtype of output.
    handle_unknown : 'error' (default) or 'ignore'
        Whether to raise an error or ignore if a unknown categorical feature is
        present during transform (default is to raise). When this is parameter
        is set to 'ignore' and an unknown category is encountered during
        transform, the resulting one-hot encoded columns for this feature
        will be all zeros.
        Ignoring unknown categories is not supported for
        ``encoding='ordinal'``.
    Attributes
    ----------
    categories_ : list of arrays
        The categories of each feature determined during fitting. When
        categories were specified manually, this holds the sorted categories
        (in order corresponding with output of `transform`).
    Examples
    --------
    Given a dataset with three features and two samples, we let the encoder
    find the maximum value per feature and transform the data to a binary
    one-hot encoding.
    >>> from sklearn.preprocessing import CategoricalEncoder
    >>> enc = CategoricalEncoder(handle_unknown='ignore')
    >>> enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])
    ... # doctest: +ELLIPSIS
    CategoricalEncoder(categories='auto', dtype=<... 'numpy.float64'>,
              encoding='onehot', handle_unknown='ignore')
    >>> enc.transform([[0, 1, 1], [1, 0, 4]]).toarray()
    array([[ 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.],
           [ 0.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.]])
    See also
    --------
    sklearn.preprocessing.OneHotEncoder : performs a one-hot encoding of
      integer ordinal features. The ``OneHotEncoder assumes`` that input
      features take on values in the range ``[0, max(feature)]`` instead of
      using the unique values.
    sklearn.feature_extraction.DictVectorizer : performs a one-hot encoding of
      dictionary items (also handles string-valued features).
    sklearn.feature_extraction.FeatureHasher : performs an approximate one-hot
      encoding of dictionary items or strings.
    """

    def __init__(self, encoding='onehot', categories='auto', dtype=np.float64,
                 handle_unknown='error'):
        self.encoding = encoding
        self.categories = categories
        self.dtype = dtype
        self.handle_unknown = handle_unknown

    def fit(self, X, y=None):
        """Fit the CategoricalEncoder to X.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_feature]
            The data to determine the categories of each feature.
        Returns
        -------
        self
        """

        if self.encoding not in ['onehot', 'onehot-dense', 'ordinal']:
            template = ("encoding should be either 'onehot', 'onehot-dense' "
                        "or 'ordinal', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.handle_unknown not in ['error', 'ignore']:
            template = ("handle_unknown should be either 'error' or "
                        "'ignore', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.encoding == 'ordinal' and self.handle_unknown == 'ignore':
            raise ValueError("handle_unknown='ignore' is not supported for"
                             " encoding='ordinal'")

        X = check_array(X, dtype=np.object, accept_sparse='csc', copy=True)
        n_samples, n_features = X.shape

        self._label_encoders_ = [LabelEncoder() for _ in range(n_features)]

        for i in range(n_features):
            le = self._label_encoders_[i]
            Xi = X[:, i]
            if self.categories == 'auto':
                le.fit(Xi)
            else:
                valid_mask = np.in1d(Xi, self.categories[i])
                if not np.all(valid_mask):
                    if self.handle_unknown == 'error':
                        diff = np.unique(Xi[~valid_mask])
                        msg = ("Found unknown categories {0} in column {1}"
                               " during fit".format(diff, i))
                        raise ValueError(msg)
                le.classes_ = np.array(np.sort(self.categories[i]))

        self.categories_ = [le.classes_ for le in self._label_encoders_]

        return self

    def transform(self, X):
        """Transform X using one-hot encoding.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to encode.
        Returns
        -------
        X_out : sparse matrix or a 2-d array
            Transformed input.
        """
        X = check_array(X, accept_sparse='csc', dtype=np.object, copy=True)
        n_samples, n_features = X.shape
        X_int = np.zeros_like(X, dtype=np.int)
        X_mask = np.ones_like(X, dtype=np.bool)

        for i in range(n_features):
            valid_mask = np.in1d(X[:, i], self.categories_[i])

            if not np.all(valid_mask):
                if self.handle_unknown == 'error':
                    diff = np.unique(X[~valid_mask, i])
                    msg = ("Found unknown categories {0} in column {1}"
                           " during transform".format(diff, i))
                    raise ValueError(msg)
                else:
                    # Set the problematic rows to an acceptable value and
                    # continue `The rows are marked `X_mask` and will be
                    # removed later.
                    X_mask[:, i] = valid_mask
                    X[:, i][~valid_mask] = self.categories_[i][0]
            X_int[:, i] = self._label_encoders_[i].transform(X[:, i])

        if self.encoding == 'ordinal':
            return X_int.astype(self.dtype, copy=False)

        mask = X_mask.ravel()
        n_values = [cats.shape[0] for cats in self.categories_]
        n_values = np.array([0] + n_values)
        indices = np.cumsum(n_values)

        column_indices = (X_int + indices[:-1]).ravel()[mask]
        row_indices = np.repeat(np.arange(n_samples, dtype=np.int32),
                                n_features)[mask]
        data = np.ones(n_samples * n_features)[mask]

        out = sparse.csc_matrix((data, (row_indices, column_indices)),
                                shape=(n_samples, indices[-1]),
                                dtype=self.dtype).tocsr()
        if self.encoding == 'onehot-dense':
            return out.toarray()
        else:
            return out

interestingColumns = ['ACTUAL_MDC', 'ADMIT_DIAGNOSIS_CODE','ICD9_DIAGNOSIS_CODE_1', 'ICD9_DIAGNOSIS_CODE_2',
        'ICD9_DIAGNOSIS_CODE_3', 'ICD9_DIAGNOSIS_CODE_4', 'ICD9_DIAGNOSIS_CODE_5', 'ICD9_DIAGNOSIS_CODE_6',
        'ICD9_DIAGNOSIS_CODE_7', 'ICD9_DIAGNOSIS_CODE_8', 'ICD9_DIAGNOSIS_CODE_9', 'ICD9_DIAGNOSIS_CODE_10',
        'ICD9_PROCEDURE_CODE_1', 'ICD9_PROCEDURE_CODE_2', 'ICD9_PROCEDURE_CODE_3', 'ICD9_PROCEDURE_CODE_4', 'ICD9_PROCEDURE_CODE_5', 'ICD9_PROCEDURE_CODE_6', 
        'SEX', 'RACE', 'BIRTH_DATE','STATE_CODE']
dgnColumns = ['ADMIT_DIAGNOSIS_CODE','ICD9_DIAGNOSIS_CODE_1', 'ICD9_DIAGNOSIS_CODE_2',
        'ICD9_DIAGNOSIS_CODE_3', 'ICD9_DIAGNOSIS_CODE_4', 'ICD9_DIAGNOSIS_CODE_5', 'ICD9_DIAGNOSIS_CODE_6',
        'ICD9_DIAGNOSIS_CODE_7', 'ICD9_DIAGNOSIS_CODE_8', 'ICD9_DIAGNOSIS_CODE_9', 'ICD9_DIAGNOSIS_CODE_10']
prcdrColumns = [ 'ICD9_PROCEDURE_CODE_1', 'ICD9_PROCEDURE_CODE_2', 'ICD9_PROCEDURE_CODE_3', 'ICD9_PROCEDURE_CODE_4', 'ICD9_PROCEDURE_CODE_5', 'ICD9_PROCEDURE_CODE_6']
demoColumns = ['SEX', 'RACE', 'BIRTH_DATE','STATE_CODE']
interestingColumns2 = ['ACTUAL_MDC', 'ACTUAL_DRG_GROUPER','ADMIT_DIAGNOSIS_CODE','ICD9_DIAGNOSIS_CODE_1', 'ICD9_DIAGNOSIS_CODE_2',
        'ICD9_DIAGNOSIS_CODE_3', 'ICD9_DIAGNOSIS_CODE_4', 'ICD9_DIAGNOSIS_CODE_5', 'ICD9_DIAGNOSIS_CODE_6',
        'ICD9_DIAGNOSIS_CODE_7', 'ICD9_DIAGNOSIS_CODE_8', 'ICD9_DIAGNOSIS_CODE_9', 'ICD9_DIAGNOSIS_CODE_10',
        'ICD9_PROCEDURE_CODE_1', 'ICD9_PROCEDURE_CODE_2', 'ICD9_PROCEDURE_CODE_3', 'ICD9_PROCEDURE_CODE_4', 'ICD9_PROCEDURE_CODE_5', 'ICD9_PROCEDURE_CODE_6', 
        'SEX', 'RACE', 'BIRTH_DATE','STATE_CODE']
interestingColumns3 = ['ACTUAL_MDC', 'ACTUAL_DRG_GROUPER','ADMIT_DIAGNOSIS_CODE','ICD9_DIAGNOSIS_CODE_1', 'ICD9_DIAGNOSIS_CODE_2',
        'ICD9_DIAGNOSIS_CODE_3', 'ICD9_DIAGNOSIS_CODE_4', 'ICD9_DIAGNOSIS_CODE_5', 'ICD9_DIAGNOSIS_CODE_6',
        'ICD9_DIAGNOSIS_CODE_7', 'ICD9_DIAGNOSIS_CODE_8', 'ICD9_DIAGNOSIS_CODE_9', 'ICD9_DIAGNOSIS_CODE_10',
        'ICD9_PROCEDURE_CODE_1', 'ICD9_PROCEDURE_CODE_2', 'ICD9_PROCEDURE_CODE_3', 'ICD9_PROCEDURE_CODE_4', 'ICD9_PROCEDURE_CODE_5', 'ICD9_PROCEDURE_CODE_6', 
        'SEX', 'RACE', 'BIRTH_DATE','STATE_CODE', 'PREDICTED_MDC']
interestingColumns4 = ['ACTUAL_MDC', 'ACTUAL_DRG_GROUPER','ADMIT_DIAGNOSIS_CODE','ICD9_DIAGNOSIS_CODE_1', 'ICD9_DIAGNOSIS_CODE_2',
        'ICD9_DIAGNOSIS_CODE_3', 'ICD9_DIAGNOSIS_CODE_4', 'ICD9_DIAGNOSIS_CODE_5', 'ICD9_DIAGNOSIS_CODE_6',
        'ICD9_DIAGNOSIS_CODE_7', 'ICD9_DIAGNOSIS_CODE_8', 'ICD9_DIAGNOSISMD_CODE_9', 'ICD9_DIAGNOSIS_CODE_10',
        'ICD9_PROCEDURE_CODE_1', 'ICD9_PROCEDURE_CODE_2', 'ICD9_PROCEDURE_CODE_3', 'ICD9_PROCEDURE_CODE_4', 'ICD9_PROCEDURE_CODE_5', 'ICD9_PROCEDURE_CODE_6', 
        'SEX', 'RACE', 'BIRTH_DATE','STATE_CODE','PREDICTED_MDC', 'PREDICTED_DRG_GROUPER']

def testSample(data):

    finalcolumns = dgnColumns + prcdrColumns + demoColumns
    # joinClaims = pd.DataFrame(data, columns = finalcolumns)
    joinClaims = data[finalcolumns]

    claims_subset_df = joinClaims[dgnColumns + prcdrColumns]
    claims_subset_df = claims_subset_df.reset_index(drop=True)
    # print(claims_subset_df.shape)
    demo_subset_df = joinClaims[demoColumns]
    # print(demo_subset_df.shape)
    demo_subset_df = demo_subset_df.reset_index(drop=True)
    # print(demo_subset_df.shape)
    demo_subset_df.fillna(0)
    combineClaims_subset = pd.concat([claims_subset_df, demo_subset_df], axis=1)
    # print(combineClaims_subset.shape)
    claims_subset_df.fillna(0)
    # print(claims_subset_df.dtypes)
    demo_subset_df = demo_subset_df.astype('int64', copy=False)

    claims_subset_df['ADMIT_DIAGNOSIS_CODE'] = claims_subset_df['ADMIT_DIAGNOSIS_CODE'].astype(str)
    claims_subset_df['ADMIT_DIAGNOSIS_CODE'] = claims_subset_df['ADMIT_DIAGNOSIS_CODE'].str.strip().str.replace('.', '')
    claims_subset_df['ADMIT_DIAGNOSIS_CODE'] = claims_subset_df['ADMIT_DIAGNOSIS_CODE'].str.strip().str.replace('NaN', '0')
    claims_subset_df['ADMIT_DIAGNOSIS_CODE'] = claims_subset_df['ADMIT_DIAGNOSIS_CODE'].str.strip().str.replace('nan', '0')
    for j in range(1,11):
        claims_subset_df['ICD9_DIAGNOSIS_CODE_' + str(j)] = claims_subset_df['ICD9_DIAGNOSIS_CODE_' + str(j)].astype(str)
        claims_subset_df['ICD9_DIAGNOSIS_CODE_' + str(j)] = claims_subset_df['ICD9_DIAGNOSIS_CODE_' + str(j)].str.strip().str.replace('.', '')
        claims_subset_df['ICD9_DIAGNOSIS_CODE_' + str(j)] = claims_subset_df['ICD9_DIAGNOSIS_CODE_' + str(j)].str.strip().str.replace('NaN', '0')
        claims_subset_df['ICD9_DIAGNOSIS_CODE_' + str(j)] = claims_subset_df['ICD9_DIAGNOSIS_CODE_' + str(j)].str.strip().str.replace('nan', '0')
    for j in range(1,7):
        claims_subset_df['ICD9_PROCEDURE_CODE_' + str(j)] = claims_subset_df['ICD9_PROCEDURE_CODE_' + str(j)].astype(str)
        claims_subset_df['ICD9_PROCEDURE_CODE_' + str(j)] = claims_subset_df['ICD9_PROCEDURE_CODE_' + str(j)].str.strip().str.replace('.', '')
        claims_subset_df['ICD9_PROCEDURE_CODE_' + str(j)] = claims_subset_df['ICD9_PROCEDURE_CODE_' + str(j)].str.strip().str.replace('NaN', '0')
        claims_subset_df['ICD9_PROCEDURE_CODE_' + str(j)] = claims_subset_df['ICD9_PROCEDURE_CODE_' + str(j)].str.strip().str.replace('nan', '0')

    start_time = time.clock()
    # skip first column the target variablee
    j=1
    encoder_file = open("cat_encoder_mdc_" + str(j) + ".pkl", "rb")
    cat = pickle.load(encoder_file)
    train_df = cat.transform(claims_subset_df['ADMIT_DIAGNOSIS_CODE'].values.reshape(-1,1))
    encoder_file.close()
    train_df_c = pd.DataFrame(train_df.toarray(), columns=cat.categories_)

    for col in interestingColumns[2:18]:   
        j=j+1 
        encoder_file = open("cat_encoder_mdc_" + str(j) + ".pkl", "rb")
        cat = pickle.load(encoder_file)
        train_df = cat.transform(claims_subset_df[col].values.reshape(-1,1))
        encoder_file.close()
        train_df_c = train_df_c.append(pd.DataFrame(train_df.toarray(), columns=cat.categories_))
        train_df_c = train_df_c.fillna(0)
        train_df_c = train_df_c.groupby(train_df_c.index).sum()
        # print(train_df_c.shape)
        # print("Time to run", time.clock() - start_time, "seconds")

    # print(train_df_c.shape)
    train_df_new = pd.concat([train_df_c,demo_subset_df ], axis=1)
    # print(train_df_new.shape) 
    # print(train_df_new.dtypes)
    pca_file = open("pca_mdc_1.pkl", "rb")
    pca = pickle.load(pca_file)    
    x = train_df_new.values.tolist()
    X = pca.transform(x)
    pca_file.close()
    return X

def getModel(sess):
    feature_size=2500

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
    saver.restore(sess, "C:/Users/ak055384/Documents/HIMClaims/MDC/tmp/model_mdc.ckpt")
    y_res = [y_conv, x]
    return y_res


def getPredictedMDC(inputs,sess, y_res):
    x = y_res[1]
    # dataReshaped = np.array(inputs).reshape(1,21)
    dataReshaped = inputs
    testX = testSample(dataReshaped)
    feature_size = testX.shape[1]
    #print(feature_size)
    prediction = tf.argmax(y_res[0], axis = 1)
    predictedMDC = prediction.eval(feed_dict={x: testX}, session=sess)
    label_file = open("label_1.pkl", "rb")
    encoder = pickle.load(label_file)
    predY = encoder.inverse_transform(predictedMDC)
    label_file.close()
   # print('Predicted MDC ', predY)
    predictedMDC = np.array(predY).reshape(inputs.shape[0],1)
    inputs.loc[:,'PREDICTED_MDC'] = predY
    return inputs

def computeDRG(inputs):
    # dataReshaped = np.array(inputs).reshape(1,21)
    dataReshaped = inputs
    finalcolumns = interestingColumns3
    finalClaims = pd.DataFrame(dataReshaped, columns = finalcolumns)
    # print(finalClaims.shape)
    finalClaims  = finalClaims[finalClaims['ACTUAL_MDC'].notnull()]  
    data1 = finalClaims[finalClaims['ACTUAL_MDC'].str.match('MDC-01') == True]
    # print(data1.shape)  
    data2 = finalClaims[finalClaims['ACTUAL_MDC'].str.match('MDC-03') == True]
    # print(data2.shape)  
    data3 = finalClaims[finalClaims['ACTUAL_MDC'].str.match('MDC-04') == True]
    # print(data3.shape)  
    data4 = finalClaims[finalClaims['ACTUAL_MDC'].str.match('MDC-05') == True]
    # print(data4.shape) 
    data_array = [] 
    data = pd.DataFrame(data_array, columns = interestingColumns4)
    # data4.to_csv("mdc_5.csv", index=False)
    if not data4.empty and data4['ACTUAL_MDC'].iloc[0] == 'MDC-05':
        finalcolumns = dgnColumns + prcdrColumns + demoColumns
        # joinClaims = pd.DataFrame(data1, columns = finalcolumns)
        joinClaims = data4[finalcolumns]
        claims_subset_df = joinClaims[dgnColumns + prcdrColumns]
        claims_subset_df = claims_subset_df.reset_index(drop=True)
        # print(claims_subset_df.shape)
        demo_subset_df = joinClaims[demoColumns]
        # print(demo_subset_df.shape)
        demo_subset_df = demo_subset_df.reset_index(drop=True)
        # print(demo_subset_df.shape)
        demo_subset_df.fillna(0)
        combineClaims_subset = pd.concat([claims_subset_df, demo_subset_df], axis=1)
        # print(combineClaims_subset.shape)
        claims_subset_df.fillna(0.0)
        # print(claims_subset_df.dtypes)
        demo_subset_df = demo_subset_df.astype('int64', copy=False)

        claims_subset_df['ADMIT_DIAGNOSIS_CODE'] = claims_subset_df['ADMIT_DIAGNOSIS_CODE'].astype(str)
        claims_subset_df['ADMIT_DIAGNOSIS_CODE'] = claims_subset_df['ADMIT_DIAGNOSIS_CODE'].str.strip().str.replace('.', '')
        claims_subset_df['ADMIT_DIAGNOSIS_CODE'] = claims_subset_df['ADMIT_DIAGNOSIS_CODE'].str.strip().str.replace('NaN', '0')
        claims_subset_df['ADMIT_DIAGNOSIS_CODE'] = claims_subset_df['ADMIT_DIAGNOSIS_CODE'].str.strip().str.replace('nan', '0')
        for j in range(1,11):
            claims_subset_df['ICD9_DIAGNOSIS_CODE_' + str(j)] = claims_subset_df['ICD9_DIAGNOSIS_CODE_' + str(j)].astype(str)
            claims_subset_df['ICD9_DIAGNOSIS_CODE_' + str(j)] = claims_subset_df['ICD9_DIAGNOSIS_CODE_' + str(j)].str.strip().str.replace('.', '')
            claims_subset_df['ICD9_DIAGNOSIS_CODE_' + str(j)] = claims_subset_df['ICD9_DIAGNOSIS_CODE_' + str(j)].str.strip().str.replace('NaN', '0')
            claims_subset_df['ICD9_DIAGNOSIS_CODE_' + str(j)] = claims_subset_df['ICD9_DIAGNOSIS_CODE_' + str(j)].str.strip().str.replace('nan', '0')
        for j in range(1,7):
            claims_subset_df['ICD9_PROCEDURE_CODE_' + str(j)] = claims_subset_df['ICD9_PROCEDURE_CODE_' + str(j)].astype(str)
            claims_subset_df['ICD9_PROCEDURE_CODE_' + str(j)] = claims_subset_df['ICD9_PROCEDURE_CODE_' + str(j)].str.strip().str.replace('.', '')
            claims_subset_df['ICD9_PROCEDURE_CODE_' + str(j)] = claims_subset_df['ICD9_PROCEDURE_CODE_' + str(j)].str.strip().str.replace('NaN', '0')
            claims_subset_df['ICD9_PROCEDURE_CODE_' + str(j)] = claims_subset_df['ICD9_PROCEDURE_CODE_' + str(j)].str.strip().str.replace('nan', '0')

        start_time = time.clock()
        # skip first column the target variablee
        j=1
        encoder_file = open("cat_encoder_mdc5_" + str(j) + ".pkl", "rb")
        cat = pickle.load(encoder_file)
        train_df = cat.transform(claims_subset_df['ADMIT_DIAGNOSIS_CODE'].values.reshape(-1,1))

        encoder_file.close()
        train_df_c = pd.DataFrame(train_df.toarray(), columns=cat.categories_)

        for col in interestingColumns[2:18]:   
            j=j+1 
            encoder_file = open("cat_encoder_mdc5_" + str(j) + ".pkl", "rb")
            cat = pickle.load(encoder_file)
            train_df = cat.transform(claims_subset_df[col].values.reshape(-1,1))
            encoder_file.close()
            train_df_c = train_df_c.append(pd.DataFrame(train_df.toarray(), columns=cat.categories_))
            train_df_c = train_df_c.fillna(0)
            train_df_c = train_df_c.groupby(train_df_c.index).sum()
            # print(train_df_c.shape)
            # print("Time to run", time.clock() - start_time, "seconds")
        # print(train_df_c.shape)
        train_df_new = pd.concat([train_df_c,demo_subset_df ], axis=1)
        # print(train_df_new.shape)
        X_test = train_df_new
        clf_dt_file = open("clf_dt_decision_mdc5.pkl", "rb")
        clf_dt = pickle.load(clf_dt_file)
        dt_predicted1 = clf_dt.predict(X_test)
        # print(dt_predicted1)
        data4.loc[:, 'PREDICTED_DRG_GROUPER'] = np.array(dt_predicted1).reshape(data4.shape[0],1)
        data = data.append(data4)
    if not data3.empty and data3['ACTUAL_MDC'].iloc[0] == 'MDC-04':
        finalcolumns = dgnColumns + prcdrColumns + demoColumns
        # joinClaims = pd.DataFrame(data1, columns = finalcolumns)
        joinClaims = data3[finalcolumns]
        claims_subset_df = joinClaims[dgnColumns + prcdrColumns]
        claims_subset_df = claims_subset_df.reset_index(drop=True)
        #print(claims_subset_df.shape)
        demo_subset_df = joinClaims[demoColumns]
        #print(demo_subset_df.shape)
        demo_subset_df = demo_subset_df.reset_index(drop=True)
        #print(demo_subset_df.shape)
        demo_subset_df.fillna(0)
        combineClaims_subset = pd.concat([claims_subset_df, demo_subset_df], axis=1)
        #print(combineClaims_subset.shape)
        claims_subset_df.fillna(0.0)
        # print(claims_subset_df.dtypes)
        demo_subset_df = demo_subset_df.astype('int64', copy=False)

        claims_subset_df['ADMIT_DIAGNOSIS_CODE'] = claims_subset_df['ADMIT_DIAGNOSIS_CODE'].astype(str)
        claims_subset_df['ADMIT_DIAGNOSIS_CODE'] = claims_subset_df['ADMIT_DIAGNOSIS_CODE'].str.strip().str.replace('.', '')
        claims_subset_df['ADMIT_DIAGNOSIS_CODE'] = claims_subset_df['ADMIT_DIAGNOSIS_CODE'].str.strip().str.replace('NaN', '0')
        claims_subset_df['ADMIT_DIAGNOSIS_CODE'] = claims_subset_df['ADMIT_DIAGNOSIS_CODE'].str.strip().str.replace('nan', '0')
        for j in range(1,11):
            claims_subset_df['ICD9_DIAGNOSIS_CODE_' + str(j)] = claims_subset_df['ICD9_DIAGNOSIS_CODE_' + str(j)].astype(str)
            claims_subset_df['ICD9_DIAGNOSIS_CODE_' + str(j)] = claims_subset_df['ICD9_DIAGNOSIS_CODE_' + str(j)].str.strip().str.replace('.', '')
            claims_subset_df['ICD9_DIAGNOSIS_CODE_' + str(j)] = claims_subset_df['ICD9_DIAGNOSIS_CODE_' + str(j)].str.strip().str.replace('NaN', '0')
            claims_subset_df['ICD9_DIAGNOSIS_CODE_' + str(j)] = claims_subset_df['ICD9_DIAGNOSIS_CODE_' + str(j)].str.strip().str.replace('nan', '0')
        for j in range(1,7):
            claims_subset_df['ICD9_PROCEDURE_CODE_' + str(j)] = claims_subset_df['ICD9_PROCEDURE_CODE_' + str(j)].astype(str)
            claims_subset_df['ICD9_PROCEDURE_CODE_' + str(j)] = claims_subset_df['ICD9_PROCEDURE_CODE_' + str(j)].str.strip().str.replace('.', '')
            claims_subset_df['ICD9_PROCEDURE_CODE_' + str(j)] = claims_subset_df['ICD9_PROCEDURE_CODE_' + str(j)].str.strip().str.replace('NaN', '0')
            claims_subset_df['ICD9_PROCEDURE_CODE_' + str(j)] = claims_subset_df['ICD9_PROCEDURE_CODE_' + str(j)].str.strip().str.replace('nan', '0')

        start_time = time.clock()
        # skip first column the target variablee
        j=1
        encoder_file = open("cat_encoder_mdc4_" + str(j) + ".pkl", "rb")
        cat = pickle.load(encoder_file)
        train_df = cat.transform(claims_subset_df['ADMIT_DIAGNOSIS_CODE'].values.reshape(-1,1))

        encoder_file.close()
        train_df_c = pd.DataFrame(train_df.toarray(), columns=cat.categories_)

        for col in interestingColumns[2:18]:   
            j=j+1 
            encoder_file = open("cat_encoder_mdc4_" + str(j) + ".pkl", "rb")
            cat = pickle.load(encoder_file)
            train_df = cat.transform(claims_subset_df[col].values.reshape(-1,1))
            encoder_file.close()
            train_df_c = train_df_c.append(pd.DataFrame(train_df.toarray(), columns=cat.categories_))
            train_df_c = train_df_c.fillna(0)
            train_df_c = train_df_c.groupby(train_df_c.index).sum()
            # print(train_df_c.shape)
            # print("Time to run", time.clock() - start_time, "seconds")
        # print(train_df_c.shape)
        train_df_new = pd.concat([train_df_c,demo_subset_df ], axis=1)
        # print(train_df_new.shape)
        X_test = train_df_new
        clf_dt_file = open("clf_dt_decision_mdc4.pkl", "rb")
        clf_dt = pickle.load(clf_dt_file)
        dt_predicted2 = clf_dt.predict(X_test)
        # print(dt_predicted2)
        data3.loc[:,'PREDICTED_DRG_GROUPER'] = np.array(dt_predicted2).reshape(data3.shape[0],1)
        data = data.append(data3)
    if not data2.empty and data2['ACTUAL_MDC'].iloc[0] == 'MDC-03':
        finalcolumns = dgnColumns + prcdrColumns + demoColumns
        # joinClaims = pd.DataFrame(data1, columns = finalcolumns)
        joinClaims = data2[finalcolumns]
        claims_subset_df = joinClaims[dgnColumns + prcdrColumns]
        claims_subset_df = claims_subset_df.reset_index(drop=True)
        # print(claims_subset_df.shape)
        demo_subset_df = joinClaims[demoColumns]
        # print(demo_subset_df.shape)
        demo_subset_df = demo_subset_df.reset_index(drop=True)
        # print(demo_subset_df.shape)
        demo_subset_df.fillna(0)
        combineClaims_subset = pd.concat([claims_subset_df, demo_subset_df], axis=1)
        # print(combineClaims_subset.shape)
        claims_subset_df.fillna(0.0)
        # print(claims_subset_df.dtypes)
        demo_subset_df = demo_subset_df.astype('int64', copy=False)

        claims_subset_df['ADMIT_DIAGNOSIS_CODE'] = claims_subset_df['ADMIT_DIAGNOSIS_CODE'].astype(str)
        claims_subset_df['ADMIT_DIAGNOSIS_CODE'] = claims_subset_df['ADMIT_DIAGNOSIS_CODE'].str.strip().str.replace('.', '')
        claims_subset_df['ADMIT_DIAGNOSIS_CODE'] = claims_subset_df['ADMIT_DIAGNOSIS_CODE'].str.strip().str.replace('NaN', '0')
        claims_subset_df['ADMIT_DIAGNOSIS_CODE'] = claims_subset_df['ADMIT_DIAGNOSIS_CODE'].str.strip().str.replace('nan', '0')
        for j in range(1,11):
            claims_subset_df['ICD9_DIAGNOSIS_CODE_' + str(j)] = claims_subset_df['ICD9_DIAGNOSIS_CODE_' + str(j)].astype(str)
            claims_subset_df['ICD9_DIAGNOSIS_CODE_' + str(j)] = claims_subset_df['ICD9_DIAGNOSIS_CODE_' + str(j)].str.strip().str.replace('.', '')
            claims_subset_df['ICD9_DIAGNOSIS_CODE_' + str(j)] = claims_subset_df['ICD9_DIAGNOSIS_CODE_' + str(j)].str.strip().str.replace('NaN', '0')
            claims_subset_df['ICD9_DIAGNOSIS_CODE_' + str(j)] = claims_subset_df['ICD9_DIAGNOSIS_CODE_' + str(j)].str.strip().str.replace('nan', '0')
        for j in range(1,7):
            claims_subset_df['ICD9_PROCEDURE_CODE_' + str(j)] = claims_subset_df['ICD9_PROCEDURE_CODE_' + str(j)].astype(str)
            claims_subset_df['ICD9_PROCEDURE_CODE_' + str(j)] = claims_subset_df['ICD9_PROCEDURE_CODE_' + str(j)].str.strip().str.replace('.', '')
            claims_subset_df['ICD9_PROCEDURE_CODE_' + str(j)] = claims_subset_df['ICD9_PROCEDURE_CODE_' + str(j)].str.strip().str.replace('NaN', '0')
            claims_subset_df['ICD9_PROCEDURE_CODE_' + str(j)] = claims_subset_df['ICD9_PROCEDURE_CODE_' + str(j)].str.strip().str.replace('nan', '0')

        start_time = time.clock()
        # skip first column the target variablee
        j=1
        encoder_file = open("cat_encoder_mdc3_" + str(j) + ".pkl", "rb")
        cat = pickle.load(encoder_file)
        train_df = cat.transform(claims_subset_df['ADMIT_DIAGNOSIS_CODE'].values.reshape(-1,1))

        encoder_file.close()
        train_df_c = pd.DataFrame(train_df.toarray(), columns=cat.categories_)

        for col in interestingColumns[2:18]:   
            j=j+1 
            encoder_file = open("cat_encoder_mdc3_" + str(j) + ".pkl", "rb")
            cat = pickle.load(encoder_file)
            train_df = cat.transform(claims_subset_df[col].values.reshape(-1,1))
            encoder_file.close()
            train_df_c = train_df_c.append(pd.DataFrame(train_df.toarray(), columns=cat.categories_))
            train_df_c = train_df_c.fillna(0)
            train_df_c = train_df_c.groupby(train_df_c.index).sum()
            # print(train_df_c.shape)
            # print("Time to run", time.clock() - start_time, "seconds")
        # print(train_df_c.shape)
        train_df_new = pd.concat([train_df_c,demo_subset_df ], axis=1)
        # print(train_df_new.shape)
        X_test = train_df_new
        clf_dt_file = open("clf_dt_decision_mdc3.pkl", "rb")
        clf_dt = pickle.load(clf_dt_file)
        dt_predicted3 = clf_dt.predict(X_test)
        # print(dt_predicted3)
        data2['PREDICTED_DRG_GROUPER'] = np.array(dt_predicted3).reshape(data2.shape[0],1)
        data = data.append(data2)
    if not data1.empty and data1['ACTUAL_MDC'].iloc[0] == 'MDC-01':
        finalcolumns = dgnColumns + prcdrColumns + demoColumns
       # joinClaims = pd.DataFrame(data1, columns = finalcolumns)
        joinClaims = data1[finalcolumns]
        claims_subset_df = joinClaims[dgnColumns + prcdrColumns]
        claims_subset_df = claims_subset_df.reset_index(drop=True)
        # print(claims_subset_df.shape)
        demo_subset_df = joinClaims[demoColumns]
        # print(demo_subset_df.shape)
        demo_subset_df = demo_subset_df.reset_index(drop=True)
        # print(demo_subset_df.shape)
        demo_subset_df.fillna(0)
        combineClaims_subset = pd.concat([claims_subset_df, demo_subset_df], axis=1)
        # print(combineClaims_subset.shape)
        claims_subset_df.fillna(0.0)
        # print(claims_subset_df.dtypes)
        demo_subset_df = demo_subset_df.astype('int64', copy=False)

        claims_subset_df['ADMIT_DIAGNOSIS_CODE'] = claims_subset_df['ADMIT_DIAGNOSIS_CODE'].astype(str)
        claims_subset_df['ADMIT_DIAGNOSIS_CODE'] = claims_subset_df['ADMIT_DIAGNOSIS_CODE'].str.strip().str.replace('.', '')
        claims_subset_df['ADMIT_DIAGNOSIS_CODE'] = claims_subset_df['ADMIT_DIAGNOSIS_CODE'].str.strip().str.replace('NaN', '0')
        claims_subset_df['ADMIT_DIAGNOSIS_CODE'] = claims_subset_df['ADMIT_DIAGNOSIS_CODE'].str.strip().str.replace('nan', '0')
        for j in range(1,11):
            claims_subset_df['ICD9_DIAGNOSIS_CODE_' + str(j)] = claims_subset_df['ICD9_DIAGNOSIS_CODE_' + str(j)].astype(str)
            claims_subset_df['ICD9_DIAGNOSIS_CODE_' + str(j)] = claims_subset_df['ICD9_DIAGNOSIS_CODE_' + str(j)].str.strip().str.replace('.', '')
            claims_subset_df['ICD9_DIAGNOSIS_CODE_' + str(j)] = claims_subset_df['ICD9_DIAGNOSIS_CODE_' + str(j)].str.strip().str.replace('NaN', '0')
            claims_subset_df['ICD9_DIAGNOSIS_CODE_' + str(j)] = claims_subset_df['ICD9_DIAGNOSIS_CODE_' + str(j)].str.strip().str.replace('nan', '0')
        for j in range(1,7):
            claims_subset_df['ICD9_PROCEDURE_CODE_' + str(j)] = claims_subset_df['ICD9_PROCEDURE_CODE_' + str(j)].astype(str)
            claims_subset_df['ICD9_PROCEDURE_CODE_' + str(j)] = claims_subset_df['ICD9_PROCEDURE_CODE_' + str(j)].str.strip().str.replace('.', '')
            claims_subset_df['ICD9_PROCEDURE_CODE_' + str(j)] = claims_subset_df['ICD9_PROCEDURE_CODE_' + str(j)].str.strip().str.replace('NaN', '0')
            claims_subset_df['ICD9_PROCEDURE_CODE_' + str(j)] = claims_subset_df['ICD9_PROCEDURE_CODE_' + str(j)].str.strip().str.replace('nan', '0')

        start_time = time.clock()
        # skip first column the target variablee
        j=1
        encoder_file = open("cat_encoder_mdc1_" + str(j) + ".pkl", "rb")
        cat = pickle.load(encoder_file)
        train_df = cat.transform(claims_subset_df['ADMIT_DIAGNOSIS_CODE'].values.reshape(-1,1))

        encoder_file.close()
        train_df_c = pd.DataFrame(train_df.toarray(), columns=cat.categories_)

        for col in interestingColumns[2:18]:   
            j=j+1 
            encoder_file = open("cat_encoder_mdc1_" + str(j) + ".pkl", "rb")
            cat = pickle.load(encoder_file)
            train_df = cat.transform(claims_subset_df[col].values.reshape(-1,1))
            encoder_file.close()
            train_df_c = train_df_c.append(pd.DataFrame(train_df.toarray(), columns=cat.categories_))
            train_df_c = train_df_c.fillna(0)
            train_df_c = train_df_c.groupby(train_df_c.index).sum()
            # print(train_df_c.shape)
            # print("Time to run", time.clock() - start_time, "seconds")
        # print(train_df_c.shape)
        train_df_new = pd.concat([train_df_c,demo_subset_df ], axis=1)
        # print(train_df_new.shape)
        X_test = train_df_new
        clf_dt_file = open("clf_dt_decision_mdc1.pkl", "rb")
        clf_dt = pickle.load(clf_dt_file)
        dt_predicted4 = clf_dt.predict(X_test)
        # print(dt_predicted4)
        data1['PREDICTED_DRG_GROUPER'] = np.array(dt_predicted4).reshape(data1.shape[0],1)
        data = data.append(data1)
    # data.to_csv("data.csv", index=False)
    return data



sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)
y_res = getModel(sess)

test = pd.read_csv("test.csv", sep=",", encoding = 'utf8')
test.columns = interestingColumns2
print(test.shape)
sampleData = test.sample(frac=0.3)
print(sampleData.head())
sampleSubset = sampleData.sample(n=100)


sampleWithPredictedMDC = getPredictedMDC(sampleSubset,sess,y_res)
testY = sampleWithPredictedMDC['ACTUAL_MDC'].astype(str)
predY = sampleWithPredictedMDC['PREDICTED_MDC'].astype(str)
score = accuracy_score(testY, predY)
print('Test MDC accuracy: {:.4f}'.format(score))
sampleWithPredictedDRG = computeDRG(sampleWithPredictedMDC)
resultantData = sampleWithPredictedDRG[['ACTUAL_MDC', 'ACTUAL_DRG_GROUPER', 'PREDICTED_MDC', 'PREDICTED_DRG_GROUPER']]
testY = resultantData['ACTUAL_DRG_GROUPER'].astype(str)
predY = resultantData['PREDICTED_DRG_GROUPER'].astype(str)
score = accuracy_score(testY, predY)
print('Test Grouper accuracy: {:.4f}'.format(score))
# print('Predicted Grouper', finalData['PREDICTED_DRG_GROUPER'])





