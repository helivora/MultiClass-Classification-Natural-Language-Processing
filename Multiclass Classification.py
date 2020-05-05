
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 06:38:55 2020

@author: VoraH
"""

import pandas as pd
import numpy as np
from scipy.stats import randint
import seaborn as sns # used for plot interactive graph. 
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
#import warnings
#warnings.filterwarnings("ignore", category=FutureWarning)

# loading data
df = pd.read_csv('partclass3.csv',encoding='cp1252')
df.shape

df1 = df[['PartNum','Company','Category','PartDescription']].copy()

#####df1 = df1[pd.notnull(df1['PartDescription'])]

df1.columns = ['PartNUM','Company','Product', 'Consumer_complaint'] 

df1.shape

total = df1['Consumer_complaint'].notnull().sum()
round((total/len(df)*100),1)


pd.DataFrame(df1.Product.unique()).values


df2 = df1

# Create a new column 'category_id' with encoded categories 
df2['category_id'] = df2['Product'].factorize()[0]
category_id_df = df2[['Product', 'category_id']].drop_duplicates()


# Dictionaries for future use
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'Product']].values)


# New dataframe
df2.head()


###
fig = plt.figure(figsize=(8,6))
colors = ['grey','grey','grey','grey','grey','grey','grey','grey','grey',
    'grey','darkblue','darkblue','darkblue']
df2.groupby('Product').Consumer_complaint.count().sort_values().plot.barh(
    ylim=0, color=colors, title= 'NUMBER OF COMPLAINTS IN EACH PRODUCT CATEGORY\n')
plt.xlabel('Number of ocurrences', fontsize = 10);



###################################################################################
##Remove punctuation

df2['Consumer_complaint'] = df2['Consumer_complaint'].str.replace('[^\w\s]','')
df2['Consumer_complaint'].head()

###Remove stopwords
from nltk.corpus import stopwords
stop = stopwords.words('english')
df2['Consumer_complaint'] = df2['Consumer_complaint'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
df2['Consumer_complaint'].head()

###Remove digits/numeric data
import re 
  
def remove(list): 
    #pattern = '[0-9,@,#,!,*,-,_,:]'
    pattern = '[0-9]'
    list = [re.sub(pattern, ' ', i) for i in list] 
    return list

def keeponlyalphan(list):
    pattern = '[^a-zA-Z0-9\s]'
    list = [re.sub(pattern, ' ', i) for i in list] 
    return list


df2['Consumer_complaint'] = (keeponlyalphan(df2['Consumer_complaint']))
df2['Consumer_complaint'] = (remove(df2['Consumer_complaint']))
df2['Consumer_complaint'].head()
############################################################################################


####
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5,
                        ngram_range=(1, 2), 
                        stop_words='english')


# We transform each complaint into a vector
features = tfidf.fit_transform(df2.Consumer_complaint).toarray()

labels = df2.category_id

print("Each of the %d complaints is represented by %d features (TF-IDF score of unigrams and bigrams)" %(features.shape))


# Finding the three most correlated terms with each of the product categories
N = 3
for Product, category_id in sorted(category_to_id.items()):
  features_chi2 = chi2(features, labels == category_id)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  print("\n==> %s:" %(Product))
  print("  * Most Correlated Unigrams are: %s" %(', '.join(unigrams[-N:])))
  print("  * Most Correlated Bigrams are: %s" %(', '.join(bigrams[-N:])))
  
X = df2['Consumer_complaint'].copy() # Collection of documents
y = df2['Product'].copy() # Target or the labels we want to predict (i.e., the 13 different complaints of products)

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.25,
                                                    random_state = 0)

#X_train = X_train1['Consumer_complaint'].copy()
#X_test= X_test1['Consumer_complaint'].copy()


models = [
    RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]

# 5 Cross-validation
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))

entries = []
for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))
    
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])


mean_accuracy = cv_df.groupby('model_name').accuracy.mean()
std_accuracy = cv_df.groupby('model_name').accuracy.std()

acc = pd.concat([mean_accuracy, std_accuracy], axis= 1, 
          ignore_index=True)
acc.columns = ['Mean Accuracy', 'Standard deviation']
acc


plt.figure(figsize=(8,5))
sns.boxplot(x='model_name', y='accuracy', 
            data=cv_df, 
            color='lightblue', 
            showmeans=True)
plt.title("MEAN ACCURACY (cv = 5)\n", size=14);



X_train, X_test, y_train, y_test,indices_train,indices_test = train_test_split(features, 
                                                               labels, 
                                                               df2.index, test_size=0.25, 
                                                               random_state=1)
model = LinearSVC()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8,8))
sns.heatmap(conf_mat, annot=True, cmap="Blues", fmt='d',
            xticklabels=category_id_df.Product.values, 
            yticklabels=category_id_df.Product.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title("CONFUSION MATRIX - LinearSVC\n", size=16);


model1 = MultinomialNB()
model1.fit(X_train,y_train)
y_pred1 = model1.predict(X_test)
y_probs1 = model1.predict_proba(X_test)


model2 = LogisticRegression()
model2.fit(X_train,y_train)
y_pred2 = model2.predict(X_test)
y_probs2 = model2.predict_proba(X_test)

probMNB = y_probs1.max(axis=1)
probLGR = y_probs2.max(axis=1)

y_pred_df = pd.DataFrame(data=y_pred)
y_prob_df = pd.DataFrame(data=probMNB)

y_test.reset_index(drop=True,inplace = True)

testind = y_test.index
df3 = df2.iloc[testind]

df3.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True,inplace = True)

y_test_df = pd.DataFrame(data = y_test)
final_outcome_test = pd.concat([df3,y_test_df,y_pred_df], axis=1, ignore_index=True)

################################ALL Parts Validate#################################

all_parts_1 = pd.read_csv('partclassallparts.csv',encoding='cp1252')
all_parts_1.columns = ['RowID','Prod_no','Company', 'Consumer_complaint', 'Flag'] 
all_parts_1.fillna("NA") 

all_parts_1['Consumer_complaint'] = all_parts_1['Consumer_complaint'].astype(str)
##Remove punctuation

all_parts_1['Consumer_complaint'] = all_parts_1['Consumer_complaint'].str.replace('[^\w\s]','')

###Remove stopwords
from nltk.corpus import stopwords
stop = stopwords.words('english')
all_parts_1['Consumer_complaint'] = all_parts_1['Consumer_complaint'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

###Remove digits/numeric data

all_parts_1['Consumer_complaint'] = (keeponlyalphan(all_parts_1['Consumer_complaint']))
all_parts_1['Consumer_complaint'] = (remove(all_parts_1['Consumer_complaint']))

#############################################15######################
is_15 = all_parts_1["Company"]==15
all_parts_15 = all_parts_1[is_15]

all_parts_15_X = all_parts_15['Consumer_complaint'].copy()


features_15 = tfidf.transform(all_parts_15.Consumer_complaint).toarray()

y_pred_15 = model.predict(features_15)
y_pred_15 = pd.DataFrame(data = y_pred_15)
y_prob_15 = model2.predict_proba(features_15)
y_prob_15 = y_prob_15.max(axis=1)


all_parts_15.reset_index(drop=True, inplace=True)

y_prob_15 = pd.DataFrame(data= y_prob_15)
y_prob_15.reset_index(drop=True, inplace=True)

class_cat_15 = pd.concat([all_parts_15,y_pred_15,y_prob_15],axis = 1)
class_cat_15.columns = ['RowID','Product','Company','Description','Flag','category_id','SimilarityIndex']

class_cat_15 = pd.merge(class_cat_15,category_id_df[['Product','category_id']],on ='category_id',how = 'left' )
class_cat_15 = class_cat_15[['Company','Product_x','Product_y']]
class_cat_15.columns = ['Company','PartNum','MtlAnalysis']
###############################################Company 11#####################################

is_11 = all_parts_1["Company"]==11
all_parts_11 = all_parts_1[is_11]

all_parts_11_X = all_parts_11['Consumer_complaint'].copy()

features_11 = tfidf.transform(all_parts_11.Consumer_complaint).toarray()

y_pred_11 = model.predict(features_11)
y_pred_11 = pd.DataFrame(data = y_pred_11)
y_prob_11 = model2.predict_proba(features_11)
y_prob_11 = y_prob_11.max(axis=1)


all_parts_11.reset_index(drop=True, inplace=True)

y_prob_11 = pd.DataFrame(data= y_prob_11)
y_prob_11.reset_index(drop=True, inplace=True)

class_cat_11 = pd.concat([all_parts_11,y_pred_11,y_prob_11],axis = 1)
class_cat_11.columns = ['RowID','Product','Company','Description','Flag','category_id','SimilarityIndex']
class_cat_11 = pd.merge(class_cat_11,category_id_df[['Product','category_id']],on ='category_id',how = 'left' )


class_cat_11 = class_cat_11[['Company','Product_x','Product_y']]
class_cat_11.columns = ['Company','PartNum','MtlAnalysis']

##############################################################################################
