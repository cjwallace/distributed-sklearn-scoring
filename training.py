# # Training

# Before distributing a model with pyspark, we must train a model!
# Unfortunately, pyspark does not allow us to distribute the training of
# arbitrary sklearn models - doing so requires orchestrating distributed
# compute within the model training code, which is the reason spark has its
# own ML algorithm implementations.
# However, pyspark does allow us to distribute pre-trained sklearn models using
# user defined functions.
# Vectorized user defined functions allow us to amortize some of the overhead
# of python -> jvm -> python conversion cost of UDFs.

# ## Imports

import pandas as pd
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

# ## Load data

# We're using a canned dataset, so this is easy.

data = sns.load_dataset('iris')

data.head()

# ## Set up train and test data

feature_columns = [
    'sepal_length',
    'sepal_width',
    'petal_length',
    'petal_width'
]

target_column = 'species'

X_train, X_test, y_train, y_test = train_test_split(
    data[feature_columns],
    data[target_column],
    stratify=data[target_column],
    test_size=0.1
)


# ## Train a simple model

model = LogisticRegression()
model.fit(X_train, y_train)

# ## Evaluate

confusion_matrix(y_test, model.predict(X_test))

# ## Persist

# OK, we've trained a simple model and we're happy with it's performance.
# We'd like to run this prediction on a lot more data.

joblib.dump(model, 'model.pkl')