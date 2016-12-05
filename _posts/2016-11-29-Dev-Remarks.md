---
layout: post
comments: true
title:  "Development Remarks"
excerpt: "Dev notes to trace experience progress"
date:   2016-11-29
mathjax: true
---

# Prelude

In the process of developing all kinds of machine learning algorithms from scratch, I found it's really tough to make progress with my own expectation. You want the code to be generative, but it just sucks in the mess data structure. You want the code to express some quick feedback, but it just sucks with performance issues, etc,. So, I think it may be valuable to trace the problems I met, the solutions I found, and the ways I finally use to apply to my own project.

For current Python ecosystem, **SciPy**, the [`scikit-learn`](http://scikit-learn.org/stable/) is one wonderful project to be referenced. Lots of problems that beyond our own domain knowledge can be found and researched there.

# Naive Bayes

### - Data Structure Design Trap

When I first create the NB implementation with one method to synthesize data, it's a trap to consider both routines at the same time. My first implementation has to be overthrown as there're too many trails of sythesized method, especially the interface design of NB's train() method.

From the succinct perspective view, the training process only requires input data `X` and output/target data `y`. Thus a clear interface:

```python
def train(X, y):
    """
    Input
    -----
    - X: 2-D numpy array with shape (num_train, num_attrs)
    - y: 1-D numpy array with shape (num_train, )
    """
    pass
```

is all we need. You don't need to give the `X` or `y` more generative data format, like dictionary, which is the preprocessing work of users or other layers. At this basic layer, we only need to choose the data format based on performance quality.

### - Python matrix v.s. dictionary Performance

For Python, it's much like MatLab, it's better at matrix computation instead of other dynamic allocated data structure like `dict`, which is benefited from the fixed continuous data allocation in memeory. When you convert contents of one text into `Bag-of-Words` format, numpy array format may give your **1000x** performance improvement comparing the use of `dict`.

### - Sparse Matrix

When dealing with large scale data set, at least the level of IMDB data, it's very important to use sparse matrix to store bag-of-words data representation. Let's do the math. For IMDB data,

- It contains positive and negative comments, 12500 per class, i.e. 25000 rows in total.
 - Its volumn of vocabulary is 95073, ignoring stop words, i.e. 95073 columns.
 - Each frequency of word is represented in integer format, i.e. 4 bytes.

In all, we need $$25000 \times 95073 \times 4 \div 1024^3 = 8$$ GB memory, which is definitely unmanageable for current computer.

Directly implement the library of sparse matrix is painful. Fortunately, we can use `scipy.sparse` for our work. The below two types are helpful:

```python
# Compressed Sparse Column matrix
csc_matrix((data, indices, indptr), [shape=(M, N)])

# Compressed Sparse Row matrix
csr_matrix((data, indices, indptr), [shape=(M, N)])
```

. To construct the data matching above interface, the implementation of `scikit-learn` is one good example:

```python
def _make_int_array():
    """Construct an array.array of a type suitable for scipy.sparse indices."""
    return array.array(str("i"))


j_indices = []

# get efficient array 'array.array'.
indptr = _make_int_array()
values = _make_int_array()

# The first pointer should point to the 
# initial location, i.e. position 0.
#
# It's used to construct the parameter of
# 'csc_matrix((data, indices, indptr), [shape=(M, N)])'
indptr.append(0)


for doc in X:
    feature_counter = {}
    for feature in doc:
        try: feature_idx = voc_dict[feature]
	    if not feature_idx in feature_counter:
	        feature_counter[feature_idx] = 1
	    else:
	        feature_counter[feature_idx] += 1
	except KeyError:
	    # Ignore out-of-vocabulary items for fixed_vocab=True
	    continue   

    # As array.array, it uses method 'extend'
    # instead of 'append' to append element.
    j_indices.extend(feature_counter.keys())
    values.extend(feature_counter.values())
    indptr.append(len(j_indices))
```

Here we notice that the `indptr` can be treated as the endpoint of each doc's chunk.

TK

### - Use Naive Bayes classifier in scikit-learn

The desgin of scikit-learn's naive bayes classifier is elegant. There're 4 steps to finish

1. Get the raw data from files.
2. Extract the `counts` feature based on `bag-of-words` model.
3. Transform 'counts' feature into `tf-idf`(or `tf`) feature.
4. Train the Naive Bayes model based on `tf-idf` features and corresponding target values.
5. Predict the test data set.

**Step 1**: get raw data

```python
# Get IMDB data
import os
def getIMDBData(dirPre, dataLabel):
    res = {}
    for dl in dataLabel:
        dirPath = os.path.join(dirPre, dl)
        fileNames = os.listdir(dirPath)

        docs = []
        for fN in fileNames:
            doc = open(os.path.join(dirPath, fN), 'r').read()
            docs.append(doc)
            
        res[dl] = docs
        
    return res
    
dataPathPre = os.path.join('dataset', 'hw1_dataset_nb')
devType = ['train', 'test']
dataLabel = ['pos', 'neg']

IMDB_train = getIMDBData(os.path.join(dataPathPre, devType[0]), dataLabel)
IMDB_test = getIMDBData(os.path.join(dataPathPre, devType[1]), dataLabel)
IMDB_stop_words = open(os.path.join(dataPathPre, 'sw.txt'), 'r').read()
stop_words = IMDB_stop_words.split()
```

**Step 2**: extract count feature

```python
# Extract 'counts' feature based on 'bag-of-word'
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(analyzer="word", stop_words=stop_words)
X_train_counts = count_vect.fit_transform(IMDB_train['pos'] + IMDB_train['neg'])
X_train_counts.shape
```

**Step 3**: transform into tf-idf feature

```python
# Get 'tf-idf' feature from 'counts' feature
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape
```

**Step 4**: train naive bayes based

```python
# Train the MultinomialNB based on 'tf-idf' feature,
# and their corresponding targets.
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, [1] * 12500 + [0] * 12500)
```

**Step 5**: prediction

```python
# Predict the feature
docs_new = IMDB_test['neg'][0:5] + IMDB_test['pos'][0:5]
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)

docs_target_names = ['negative', 'positive']
for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, docs_target_names[category]))
    print '\n'
```

In order to make the whole process compact, scikit-learn provides `pipeline` to integrate the whole above prcoess.

```python
# Build pipeline of above steps
from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),
])

# Train pipeline
text_clf = text_clf.fit(IMDB_train['pos'] + IMDB_train['neg'], [1] * 12500 + [0] * 12500)

# Evaluate performance of test set
import numpy as np
predicted = text_clf.predict(IMDB_test['pos'] + IMDB_test['neg'])
np.mean(predicted == [1] * 12500 + [0] * 12500) 
```



