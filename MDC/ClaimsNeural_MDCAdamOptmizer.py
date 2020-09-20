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

logPath = "./tb_logs"
features = 2500

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
   """

    def __init__(self, encoding='onehot', categories='auto', dtype=np.float32,
                 handle_unknown='error'):
        self.encoding = encoding
        self.categories = categories
        self.dtype = dtype
        self.handle_unknown = handle_unknown

    def fit(self, X, y=None):
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
    # claims2 = pd.read_csv("DE1_0_2008_to_2010_Inpatient_Claims_Sample_2.csv", sep=",", encoding = 'utf8')
    claims = pd.concat([claims1])
    print(claims.shape)
    summary1 = pd.read_csv("DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv", sep=",", encoding = 'utf8')
    summary2 = pd.read_csv("DE1_0_2009_Beneficiary_Summary_File_Sample_1.csv", sep=",", encoding = 'utf8')
    summary3 = pd.read_csv("DE1_0_2010_Beneficiary_Summary_File_Sample_1.csv", sep=",", encoding = 'utf8')
    # summary4 = pd.read_csv("DE1_0_2008_Beneficiary_Summary_File_Sample_2.csv", sep=",", encoding = 'utf8')
    # summary5 = pd.read_csv("DE1_0_2009_Beneficiary_Summary_File_Sample_2.csv", sep=",", encoding = 'utf8')
    # summary6 = pd.read_csv("DE1_0_2010_Beneficiary_Summary_File_Sample_2.csv", sep=",", encoding = 'utf8')
    summary = pd.concat([summary1, summary2,summary3])
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
    targetName = 'MDC'
    targetSeries = joinClaims[targetName]
    #remove target from current location and insert in collum 0
    del joinClaims[targetName]
    joinClaims.insert(0, targetName, targetSeries)
    print(joinClaims.shape)

    interestingColumns = ['MDC', 'ADMTNG_ICD9_DGNS_CD','ICD9_DGNS_CD_1', 'ICD9_DGNS_CD_2',
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
    print(claims_subset_df.dtypes)

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
    # encoder_file = open("cat_encoder_mdc_" + str(j) + ".pkl", "wb")
    # pickle.dump(cat,encoder_file)
    # encoder_file.close()
    # train_df_c = pd.DataFrame(train_df.toarray(), columns=cat.categories_)

    # for col in interestingColumns[2:18]:   
    #     j=j+1 
    #     cat = CategoricalEncoder()
    #     train_df = cat.fit_transform(claims_subset_df[col].values.reshape(-1,1))
    #     encoder_file = open("cat_encoder_mdc_" + str(j) + ".pkl", "wb")
    #     pickle.dump(cat,encoder_file)
    #     encoder_file.close()
    #     train_df_c = train_df_c.append(pd.DataFrame(train_df.toarray(), columns=cat.categories_))
    #     train_df_c = train_df_c.fillna(0)
    #     train_df_c = train_df_c.groupby(train_df_c.index).sum()
    #     print(train_df_c.shape)
    #     print("Time to run", time.clock() - start_time, "seconds")
    # print(train_df_c.shape)
    # train_df_c.to_pickle("train_mdc.pkl.compress", compression="gzip", protocol=4)

    train_df_c = pd.read_pickle("train_mdc.pkl.compress", compression="gzip")
    print(train_df_c.shape)
    train_df_new = pd.concat([train_df_c,demo_subset_df ], axis=1)
    print(train_df_new.shape)
    uniqueDrgs = targetSeries.unique()
    print(len(uniqueDrgs))
    n_class = len(uniqueDrgs)
    # pca_file = open("pca_mdc_1.pkl", "rb")
    # pca = pickle.load(pca_file)    
    x = train_df_new.values.tolist()
    y = targetSeries.values.tolist()
    pca = PCA(n_components = features, random_state=42)
    X = pca.fit_transform(x)
    print(X.shape)
    pca_file = open("pca_mdc_1.pkl", "wb")
    pickle.dump(pca, pca_file)
    pca_file.close()
    # print(sum(pca.explained_variance_ratio_))
    encoder = LabelEncoder()
    y_new = encoder.fit_transform(y)
    label_file = open("label_1.pkl", "wb")
    pickle.dump(encoder, label_file)
    label_file.close()
    print(y_new)
    y_hot = tf.one_hot(indices = y_new, depth = n_class, on_value = 1, off_value=0, axis=1, name='labels')
    Y = sess.run(y_hot)
    print(Y.shape)
    return X,Y,n_class

sess = tf.Session()
X,Y,n_class = read_claims_data(sess)
train_x,inter_x,train_y,inter_y = train_test_split(X,Y,test_size=0.30, random_state=42)
test_x,valid_x,test_y,valid_y = train_test_split(inter_x,inter_y,test_size=0.33, random_state=42)

print(n_class)

print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)
print(valid_x.shape)
print(valid_y.shape)

def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

# #Decision Tree train model. Call up my model and name it clf
# from sklearn import tree 
# clf_dt = tree.DecisionTreeClassifier()
# #Call up the model to see the parameters you can tune (and their default setting)
# print(clf_dt)
# #Fit clf to the training data
# clf_dt = clf_dt.fit(X_train, y_train)
# dt_expected = y_test
# dt_predicted = clf_dt.predict(X_test)
# print(clf_dt.score(X_test, y_test))
# # summarize the fit of the model
# print("accuracy: " + str(metrics.accuracy_score(dt_expected, dt_predicted)))
# print(metrics.classification_report(dt_expected, dt_predicted))


feature_size = X.shape[1]
print(feature_size)

with tf.name_scope("input_parameters"):
    feature_size=2500
    n_class = 25

with tf.name_scope("Claims_Data_Input"):
    x = tf.placeholder(tf.float32,[None,feature_size], name ="diag_proc_demo_data")
    y_ = tf.placeholder(tf.float32,[None,n_class], name="mdc_value")

with tf.name_scope('weights'):
    def weight_variable(feature_size, n_class, name=None):
        initial = tf.truncated_normal([feature_size, n_class])
        weights = tf.Variable(initial, name=name)
        variable_summaries(weights)
        return weights

with tf.name_scope('biases'):
    def bias_variable(n_class, name=None):
        initial = tf.truncated_normal([n_class])
        biases = tf.Variable(initial, name=name)
        variable_summaries(biases)
        return biases

with tf.name_scope('mdc_neural_network'):
    weights = weight_variable(feature_size, n_class, name='Weight')
    biases = bias_variable(n_class, name='biases')
    logits = tf.matmul(x, weights) + biases
    tf.summary.histogram('output_layer', logits)

with tf.name_scope("error_rate"):
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=y_))

with tf.name_scope("error_optimizer"):
    optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

with tf.name_scope("predicted_mdc"):
    y_conv = tf.nn.softmax(logits=logits)

with tf.name_scope("training_accuracy"):
    correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(y_,1))
    training_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.name_scope("mini_accuracy"):
    correct_prediction_mini = tf.equal(tf.argmax(logits,1), tf.argmax(y_,1))
    mini_accuracy = tf.reduce_mean(tf.cast(correct_prediction_mini, tf.float32))

with tf.name_scope("validation_accuracy"):
    correct_prediction_valid = tf.equal(tf.argmax(logits,1), tf.argmax(y_,1))
    validation_accuracy = tf.reduce_mean(tf.cast(correct_prediction_valid, tf.float32))

with tf.name_scope("test_accuracy"):
    correct_prediction_test= tf.equal(tf.argmax(logits,1), tf.argmax(y_,1))
    test_accuracy = tf.reduce_mean(tf.cast(correct_prediction_test, tf.float32))

tf.summary.scalar("error_rate", loss)
tf.summary.scalar("training_accuracy", training_accuracy)
tf.summary.scalar("validation_accuracy", validation_accuracy)

# TB - Merge summaries 
summarize_all = tf.summary.merge_all()

tbWriter = tf.summary.FileWriter(logPath, sess.graph)

#initialize all variables.
init = tf.global_variables_initializer()
sess.run(init)

saver = tf.train.Saver()

#saver

training_epochs = 1501
batch_size = 100
loss_history = np.empty(shape=[1],dtype=float)
print(train_y.shape[0]//batch_size)
for step in range(training_epochs):
    # Pick an offset within the training data, which has been randomized.
    for iteration in range(train_y.shape[0]//batch_size):
        offset = (iteration * batch_size) % (train_y.shape[0] - batch_size)
        batch_data = train_x[offset:(offset + batch_size), :]
        batch_labels = train_y[offset:(offset + batch_size), :]
        _, loss_value, summary = sess.run([optimizer, loss, summarize_all], feed_dict={x: batch_data,y_: batch_labels})
        loss_history = np.append(loss_history,loss_value)
        if iteration % 100 == 0:
             print('iteration:{} loss:{:.6f} mini batch accuracy: {:.4f}'.format(iteration, loss_value,  sess.run(mini_accuracy, feed_dict={x:batch_data, y_: batch_labels})))
    train_accuracy =  sess.run(training_accuracy, feed_dict={x:train_x, y_: train_y})
    print('step:{} loss:{:.6f} train accuracy: {:.4f}'.format(step, loss_value, train_accuracy))
    valid_accuracy = sess.run(validation_accuracy, feed_dict={x:valid_x, y_: valid_y})
    print('step:{} validation accuracy: {:.4f}'.format(step, valid_accuracy))
    tbWriter.add_summary(summary,step)
    if valid_accuracy >=0.905:
        break



plt.plot(range(len(loss_history)),loss_history)
plt.axis([0,training_epochs,0,np.max(loss_history)])
plt.show()

test_y_orig = sess.run(tf.argmax(test_y,axis=1))
print('test y  ', test_y_orig)

pred_y = sess.run(tf.argmax(y_conv,axis=1), feed_dict={x:test_x})
print('predicted y  ', pred_y)

test_accuracy = sess.run(test_accuracy,feed_dict={x:test_x, y_: test_y})
print('test accuracy: {:.4f}'.format(test_accuracy))

save_path = saver.save(sess, "/tmp/model_mdc.ckpt")
print("Model saved in path: %s" % save_path)