import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.utils import  shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import label_binarize
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
features = 3500

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

def read_claims_data(sess):
    claims1 = pd.read_csv("DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv", sep=",", encoding = 'utf8')
    claims2 = pd.read_csv("DE1_0_2008_to_2010_Inpatient_Claims_Sample_2.csv", sep=",", encoding = 'utf8')
    claims3 = pd.read_csv("DE1_0_2008_to_2010_Inpatient_Claims_Sample_3.csv", sep=",", encoding = 'utf8')
    claims4 = pd.read_csv("DE1_0_2008_to_2010_Inpatient_Claims_Sample_4.csv", sep=",", encoding = 'utf8')
    claims5 = pd.read_csv("DE1_0_2008_to_2010_Inpatient_Claims_Sample_5.csv", sep=",", encoding = 'utf8')
    claims6 = pd.read_csv("DE1_0_2008_to_2010_Inpatient_Claims_Sample_6.csv", sep=",", encoding = 'utf8')
    claims = pd.concat([claims1, claims2, claims3, claims4, claims5, claims6])
    print(claims.shape)
    summary1 = pd.read_csv("DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv", sep=",", encoding = 'utf8')
    summary2 = pd.read_csv("DE1_0_2009_Beneficiary_Summary_File_Sample_1.csv", sep=",", encoding = 'utf8')
    summary3 = pd.read_csv("DE1_0_2010_Beneficiary_Summary_File_Sample_1.csv", sep=",", encoding = 'utf8')
    summary4 = pd.read_csv("DE1_0_2008_Beneficiary_Summary_File_Sample_2.csv", sep=",", encoding = 'utf8')
    summary5 = pd.read_csv("DE1_0_2009_Beneficiary_Summary_File_Sample_2.csv", sep=",", encoding = 'utf8')
    summary6 = pd.read_csv("DE1_0_2010_Beneficiary_Summary_File_Sample_2.csv", sep=",", encoding = 'utf8')
    summary7 = pd.read_csv("DE1_0_2008_Beneficiary_Summary_File_Sample_3.csv", sep=",", encoding = 'utf8')
    summary8 = pd.read_csv("DE1_0_2009_Beneficiary_Summary_File_Sample_3.csv", sep=",", encoding = 'utf8')
    summary9 = pd.read_csv("DE1_0_2010_Beneficiary_Summary_File_Sample_3.csv", sep=",", encoding = 'utf8')
    summary10 = pd.read_csv("DE1_0_2008_Beneficiary_Summary_File_Sample_4.csv", sep=",", encoding = 'utf8')
    summary11 = pd.read_csv("DE1_0_2009_Beneficiary_Summary_File_Sample_4.csv", sep=",", encoding = 'utf8')
    summary12 = pd.read_csv("DE1_0_2010_Beneficiary_Summary_File_Sample_4.csv", sep=",", encoding = 'utf8')
    summary13 = pd.read_csv("DE1_0_2008_Beneficiary_Summary_File_Sample_5.csv", sep=",", encoding = 'utf8')
    summary14 = pd.read_csv("DE1_0_2009_Beneficiary_Summary_File_Sample_5.csv", sep=",", encoding = 'utf8')
    summary15 = pd.read_csv("DE1_0_2010_Beneficiary_Summary_File_Sample_5.csv", sep=",", encoding = 'utf8')
    summary16 = pd.read_csv("DE1_0_2008_Beneficiary_Summary_File_Sample_6.csv", sep=",", encoding = 'utf8')
    summary17 = pd.read_csv("DE1_0_2009_Beneficiary_Summary_File_Sample_6.csv", sep=",", encoding = 'utf8')
    summary18 = pd.read_csv("DE1_0_2010_Beneficiary_Summary_File_Sample_6.csv", sep=",", encoding = 'utf8')
    summary = pd.concat([summary1, summary2,summary3, summary4, summary5,summary6, summary7, summary8,summary9, summary10, 
                            summary11,summary12, summary13, summary14,summary15, summary16, summary17,summary18])
    print(summary.shape)
    claimsSummary = pd.merge(claims, summary, how='inner', on='DESYNPUF_ID')
    print(claimsSummary.shape)
    drg = pd.read_csv("Drg.csv", sep=",", encoding = 'utf8')
    drg['CLM_DRG_CD'] = drg['CLM_DRG_CD'].astype(str)
    print(drg.shape)
    print(len(drg['MDC'].unique()))
    joinClaims = pd.merge(claimsSummary, drg, how='left', on='CLM_DRG_CD')
    print(joinClaims.shape)
    joinClaims = joinClaims[joinClaims['MDC'].notnull()]
    print(joinClaims.shape)    
    joinClaims = joinClaims[joinClaims['MDC'].str.match('MDC-01') == True]
    print(joinClaims.shape)
    joinClaims =  joinClaims[joinClaims.CLM_DRG_CD.str.contains("OTH") == False]
    joinClaims =  joinClaims[joinClaims.CLM_DRG_CD.str.match("0") == False] 
    # joinClaims['CLM_DRG_CD'] = pd.to_numeric(joinClaims['CLM_DRG_CD'], errors='ignore').astype(int)
    targetName = 'CLM_DRG_CD'
    targetSeries = joinClaims[targetName]
    
    #remove target from current location and insert in collum 0
    del joinClaims[targetName]
    joinClaims.insert(0, targetName, targetSeries)
    print(joinClaims.shape)
    uniqueDrgs = targetSeries.unique()
    print(len(uniqueDrgs))

    interestingColumns = ['CLM_DRG_CD', 'ADMTNG_ICD9_DGNS_CD','ICD9_DGNS_CD_1', 'ICD9_DGNS_CD_2',
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
    print(demo_subset_df.shape)
    demo_subset_df.fillna(0)
    combineClaims_subset = pd.concat([claims_subset_df, demo_subset_df], axis=1)
    print(combineClaims_subset.shape)
    claims_subset_df.fillna(0.0)
    print(claims_subset_df.shape)
    print(targetSeries.shape)

    uniqueDrgs = targetSeries.unique()
    print(len(uniqueDrgs))
    n_class = len(uniqueDrgs)

    claims_subset_df['ADMTNG_ICD9_DGNS_CD'] = claims_subset_df['ADMTNG_ICD9_DGNS_CD'].astype(str)
    claims_subset_df['ADMTNG_ICD9_DGNS_CD'] = claims_subset_df['ADMTNG_ICD9_DGNS_CD'].str.strip().str.replace('.', '')
    claims_subset_df['ADMTNG_ICD9_DGNS_CD'] = claims_subset_df['ADMTNG_ICD9_DGNS_CD'].str.strip().str.replace('NaN', '0')
    claims_subset_df['ADMTNG_ICD9_DGNS_CD'] = claims_subset_df['ADMTNG_ICD9_DGNS_CD'].str.strip().str.replace('nan', '0')
    for j in range(1,11):
        claims_subset_df['ICD9_DGNS_CD_' + str(j)] = claims_subset_df['ICD9_DGNS_CD_' + str(j)].astype(str)
        claims_subset_df['ICD9_DGNS_CD_' + str(j)] = claims_subset_df['ICD9_DGNS_CD_' + str(j)].str.strip().str.replace('.', '')
        claims_subset_df['ICD9_DGNS_CD_' + str(j)] = claims_subset_df['ICD9_DGNS_CD_' + str(j)].str.strip().str.replace('NaN', '0')
        claims_subset_df['ICD9_DGNS_CD_' + str(j)] = claims_subset_df['ICD9_DGNS_CD_' + str(j)].str.strip().str.replace('nan', '0')
    for j in range(1,7):
        claims_subset_df['ICD9_PRCDR_CD_' + str(j)] = claims_subset_df['ICD9_PRCDR_CD_' + str(j)].astype(str)
        claims_subset_df['ICD9_PRCDR_CD_' + str(j)] = claims_subset_df['ICD9_PRCDR_CD_' + str(j)].str.strip().str.replace('.', '')
        claims_subset_df['ICD9_PRCDR_CD_' + str(j)] = claims_subset_df['ICD9_PRCDR_CD_' + str(j)].str.strip().str.replace('NaN', '0')
        claims_subset_df['ICD9_PRCDR_CD_' + str(j)] = claims_subset_df['ICD9_PRCDR_CD_' + str(j)].str.strip().str.replace('nan', '0')

    start_time = time.clock()
    # skip first column the target variablee
    # cat = CategoricalEncoder()
    # train_df = cat.fit_transform(claims_subset_df['ADMTNG_ICD9_DGNS_CD'].values.reshape(-1,1))
    # j=1
    # encoder_file = open("cat_encoder_mdc1_" + str(j) + ".pkl", "wb")
    # pickle.dump(cat,encoder_file)
    # encoder_file.close()
    # train_df_c = pd.DataFrame(train_df.toarray(), columns=cat.categories_)

    # for col in interestingColumns[2:18]:   
    #     j=j+1 
    #     cat = CategoricalEncoder()
    #     train_df = cat.fit_transform(claims_subset_df[col].values.reshape(-1,1))
    #     encoder_file = open("cat_encoder_mdc1_" + str(j) + ".pkl", "wb")
    #     pickle.dump(cat,encoder_file)
    #     encoder_file.close()
    #     train_df_c = train_df_c.append(pd.DataFrame(train_df.toarray(), columns=cat.categories_))
    #     train_df_c = train_df_c.fillna(0)
    #     train_df_c = train_df_c.groupby(train_df_c.index).sum()
    #     print(train_df_c.shape)
    #     print("Time to run", time.clock() - start_time, "seconds")
    # print(train_df_c.shape)
    # train_df_c.to_pickle("train_mdc1.pkl.compress", compression="gzip", protocol=4)

    train_df_c = pd.read_pickle("train_mdc1.pkl.compress", compression="gzip")
    print(train_df_c.shape)
    train_df_new = pd.concat([train_df_c,demo_subset_df ], axis=1)
    print(train_df_new.shape)
    X = train_df_new
    Y = targetSeries
    # x = train_df_new.values.tolist()
    # Y = targetSeries.values.tolist()
    # pca = PCA(n_components = features, random_state=42)
    # X = pca.fit_transform(x)
    # pca_file = open("pca_mdc1_1.pkl", "wb")
    # pickle.dump(pca, pca_file)
    # pca_file.close()
    # print(sum(pca.explained_variance_ratio_))
    # encoder = LabelEncoder()
    # y_new = encoder.fit_transform(y)
    # label_file = open("label4_1.pkl", "wb")
    # pickle.dump(encoder, label_file)
    # label_file.close()
    # print(y_new)
    # y_hot = tf.one_hot(indices = y_new, depth = n_class, on_value = 1, off_value=0, axis=1, name='labels')
    # Y = sess.run(y_hot)
    print(X.shape)
    print(Y.shape)
    return X,Y,n_class,uniqueDrgs

sess = tf.Session()
X,Y,n_class,uniqueDrgs = read_claims_data(sess)
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.3, random_state=42)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

def checkAccuracy(clf_dt):
    dt_expected = y_test
    dt_predicted = clf_dt.predict(X_test)
    print(clf_dt.score(X_test, y_test))
    # summarize the fit of the model
    print("Test Accuracy: " + str(metrics.accuracy_score(dt_expected, dt_predicted)))
    print(metrics.classification_report(dt_expected, dt_predicted))
    dt_train_predicted = clf_dt.predict(X_train)
    print("Train Accuracy: " + str(metrics.accuracy_score(y_train, dt_train_predicted)))
    #  Compute ROC curve and ROC area for each class
    dt_expected_new = label_binarize(dt_expected, classes = uniqueDrgs)
    dt_predicted_new = label_binarize(dt_predicted, classes = uniqueDrgs)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_class):
        fpr[i], tpr[i], _ = roc_curve(dt_expected_new[:,i], dt_predicted_new[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Plot of a ROC curve for a specific class
    for i in range(n_class):
        plt.figure()
        plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
    plt.show()
    return accuracy_score(dt_expected, dt_predicted)


#Decision Tree train model. Call up my model and name it clf
from sklearn import tree 
clf_dt = tree.DecisionTreeClassifier()
print(clf_dt)
clf_dt = clf_dt.fit(X_train, y_train)
clf_dt_file = open("clf_dt_decision_mdc1.pkl", "wb")
pickle.dump(clf_dt, clf_dt_file)
clf_dt_file.close()
test_accuracy = checkAccuracy(clf_dt)
print("Test Accuracy for Decision Tree: ",test_accuracy)
#Call up the model to see the parameters you can tune (and their default setting)

from sklearn.naive_bayes import MultinomialNB
clf_dt = MultinomialNB(fit_prior = False)
print(clf_dt)
clf_dt = clf_dt.fit(X_train, y_train)
clf_dt_file = open("clf_dt_bayes_mdc1.pkl", "wb")
pickle.dump(clf_dt, clf_dt_file)
clf_dt_file.close()
test_accuracy = checkAccuracy(clf_dt)
print("Test Accuracy for Naive Bayes: ",test_accuracy)

from sklearn.ensemble import RandomForestClassifier
clf_dt = RandomForestClassifier()
print(clf_dt)
clf_dt = clf_dt.fit(X_train, y_train)
clf_dt_file = open("clf_dt_random_mdc1.pkl", "wb")
pickle.dump(clf_dt, clf_dt_file)
clf_dt_file.close()
test_accuracy = checkAccuracy(clf_dt)
print("Test Accuracy for Random Forest: ",test_accuracy)

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# dec_clf = tree.DecisionTreeClassifier(random_state=42)
# rnd_clf = RandomForestClassifier(random_state=42)
# svm_clf = SVC(probability=True, random_state=42)

# clf_dt = VotingClassifier(
#     estimators=[('lr', dec_clf), ('rf', rnd_clf), ('svc', svm_clf)],
#     voting='soft')
# print(clf_dt)
# test_accuracy = checkAccuracy(clf_dt)
# print("Test Accuracy for Voting Classifier: ",test_accuracy)

# from sklearn.ensemble import BaggingClassifier
# clf_dt = BaggingClassifier(
#     tree.DecisionTreeClassifier(random_state=42), n_estimators=100,
#     bootstrap=True, n_jobs=-1, oob_score=True, random_state=42)
# print(clf_dt)
# test_accuracy = checkAccuracy(clf_dt)
# print("Test Accuracy for Bagging Classifier: ",test_accuracy)

from sklearn.ensemble import AdaBoostClassifier

clf_dt = AdaBoostClassifier(
    tree.DecisionTreeClassifier(), n_estimators=1000,
    algorithm="SAMME.R", learning_rate=0.1)
print(clf_dt)
clf_dt = clf_dt.fit(X_train, y_train)
clf_dt_file = open("clf_dt_adaboost_mdc1.pkl", "wb")
pickle.dump(clf_dt, clf_dt_file)
clf_dt_file.close()
test_accuracy = checkAccuracy(clf_dt)
print("Test Accuracy for Ada boost Classifier: ",test_accuracy)

# from sklearn.ensemble import GradientBoostingRegressor
# clf_dt = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.1)
# print(clf_dt)
# test_accuracy = checkAccuracy(clf_dt)
# print("Test Accuracy for Gradient Boosting Classifier: ",test_accuracy)

from xgboost.sklearn import XGBClassifier 
clf_dt = XGBClassifier(n_estimators=1000)
print(clf_dt)
clf_dt = clf_dt.fit(X_train, y_train)
clf_dt_file = open("clf_dt_xgb_mdc1.pkl", "wb")
pickle.dump(clf_dt, clf_dt_file)
clf_dt_file.close()
#Fit clf to the training data
test_accuracy = checkAccuracy(clf_dt)
print("Test Accuracy for XGB Classifier: ",test_accuracy)


# from hyperopt import fmin, tpe, hp, STATUS_OK,Trials
# space ={
#     'n_estimators':hp.quniform('n_estimators',5,10,1),
#     'learning_rate':hp.quniform('learning_rate',0.025,0.1,0.025),
#     'max_depth':hp.quniform('max_depth',1,13,1),
#     'min_child_weight': hp.quniform('min_child_weight',1,6,1),
#     'subsample': hp.quniform('subsample',0.5,1,0.05),
#     'gamma':hp.quniform('gamma',0.5,1,0.05),
#     'colsample_bytree':hp.quniform('colsample_bytree',0.5,1,0.05),
#     'nthread':6,
#     'silent':1
# }

# def score(params):
#     params['n_estimators']=int(params['n_estimators'])
#     clf_dt=XGBClassifier(**params)
#     return {'loss':1-checkAccuracy(clf_dt),'status':STATUS_OK}

# trials=Trials()
# best=fmin(score,space,algo=tpe.suggest,trials=trials,max_evals=250)
# print(best)
# print(1-score(best)['loss'])


# print(n_class)

# print(train_x.shape)
# print(train_y.shape)
# print(test_x.shape)
# print(test_y.shape)
# print(valid_x.shape)
# print(valid_y.shape)

# feature_size = X.shape[1]
# print(feature_size)

# with tf.name_scope("input_parameters"):
#     feature_size=features
#     n_class = 40

# with tf.name_scope("Claims_Data_Input"):
#     x = tf.placeholder(tf.float32,[None,feature_size], name ="diag_proc_demo_data")
#     y_ = tf.placeholder(tf.float32,[None,n_class], name="mdc_value")

# with tf.name_scope('weights'):
#     def weight_variable(feature_size, n_class, name=None):
#         initial = tf.truncated_normal([feature_size, n_class])
#         weights = tf.Variable(initial, name=name)
#         return weights

# with tf.name_scope('biases'):
#     def bias_variable(n_class, name=None):
#         initial = tf.truncated_normal([n_class])
#         biases = tf.Variable(initial, name=name)
#         return biases

# with tf.name_scope('mdc_neural_network'):
#     weights = weight_variable(feature_size, n_class, name='Weight')
#     biases = bias_variable(n_class, name='biases')
#     logits = tf.matmul(x, weights) + biases

# with tf.name_scope("error_rate"):
#     loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=y_))

# with tf.name_scope("error_optimizer"):
#     optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

# with tf.name_scope("predicted_mdc"):
#     y_conv = tf.nn.softmax(logits=logits)

# with tf.name_scope("training_accuracy"):
#     correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(y_,1))
#     training_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# with tf.name_scope("mini_accuracy"):
#     correct_prediction_mini = tf.equal(tf.argmax(logits,1), tf.argmax(y_,1))
#     mini_accuracy = tf.reduce_mean(tf.cast(correct_prediction_mini, tf.float32))

# with tf.name_scope("validation_accuracy"):
#     correct_prediction_valid = tf.equal(tf.argmax(logits,1), tf.argmax(y_,1))
#     validation_accuracy = tf.reduce_mean(tf.cast(correct_prediction_valid, tf.float32))

# with tf.name_scope("test_accuracy"):
#     correct_prediction_test= tf.equal(tf.argmax(logits,1), tf.argmax(y_,1))
#     test_accuracy = tf.reduce_mean(tf.cast(correct_prediction_test, tf.float32))

# #initialize all variables.
# init = tf.global_variables_initializer()
# sess.run(init)

# #saver

# training_epochs = 10001
# batch_size = 100
# loss_history = np.empty(shape=[1],dtype=float)
# print(train_y.shape[0]//batch_size)
# for step in range(training_epochs):
#     # Pick an offset within the training data, which has been randomized.
#     for iteration in range(train_y.shape[0]//batch_size):
#         offset = (iteration * batch_size) % (train_y.shape[0] - batch_size)
#         batch_data = train_x[offset:(offset + batch_size), :]
#         batch_labels = train_y[offset:(offset + batch_size), :]
#         _, loss_value = sess.run([optimizer, loss], feed_dict={x: batch_data,y_: batch_labels})
#         loss_history = np.append(loss_history,loss_value)
#         if iteration % 100 == 0:
#              print('iteration:{} loss:{:.6f} mini batch accuracy: {:.4f}'.format(iteration, loss_value,  sess.run(mini_accuracy, feed_dict={x:batch_data, y_: batch_labels})))
#     train_accuracy =  sess.run(training_accuracy, feed_dict={x:train_x, y_: train_y})
#     print('step:{} loss:{:.6f} train accuracy: {:.4f}'.format(step, loss_value, train_accuracy))
#     valid_accuracy = sess.run(validation_accuracy, feed_dict={x:valid_x, y_: valid_y})
#     print('step:{} validation accuracy: {:.4f}'.format(step, valid_accuracy))
#     if valid_accuracy >=0.905:
#         break



# plt.plot(range(len(loss_history)),loss_history)
# plt.axis([0,training_epochs,0,np.max(loss_history)])
# plt.show()

# test_y_orig = sess.run(tf.argmax(test_y,axis=1))
# print('test y  ', test_y_orig)

# pred_y = sess.run(tf.argmax(y_conv,axis=1), feed_dict={x:test_x})
# print('predicted y  ', pred_y)

# test_accuracy = sess.run(test_accuracy,feed_dict={x:test_x, y_: test_y})
# print('test accuracy: {:.4f}'.format(test_accuracy))



