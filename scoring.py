# # Scoring

# We have a pre-trained model file, and we'd like to apply it to
# distributed data.

import joblib

import pandas as pd
import pyspark.sql.functions as F

from functools import reduce
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from pyspark.sql.types import DoubleType
from sklearn.datasets import load_iris


# ## Initialize a spark session

spark = (
    SparkSession
    .builder
    .master('yarn')
    .appName('distribute-sklearn')
    .getOrCreate()
)

sc = spark.sparkContext


# ## Create a spark dataframe.

# Out in the real world, we wouldn't have a canned dataset, and our data may
# be much larger.
# To replicate that, we'll create an artificial dataset from the train and test
# data and replicate it to create a much larger set.

iris = load_iris()
X = iris.data


# Since we're now just predicting, we don't need to split train/test.

df = spark.createDataFrame(
  pd.DataFrame(X).reset_index(),
  schema=["id", "sepal_length", "sepal_width", "petal_length", "petal_width"]
)


# Artificially inflate the dataset to make spark do some work.

big_df = reduce(DataFrame.unionAll, (df for i in range(1000)))
big_df.count()


# Load the model from local memory and broadcast it to the cluster.
# This makes the model object available on worker nodes.

local_model = joblib.load("model.pkl")
broadcast_model = sc.broadcast(local_model)


# Create a pandas UDFs to distribute the predict function

@pandas_udf(returnType=DoubleType())
def predict_udf(*cols):
    """
    Takes several iris flower features (each a pd.Series)
    and returns a pd.Series of predicted species.
    """
    model = broadcast_model.value
    X = pd.concat(cols, axis="columns")
    predictions = pd.Series(model.predict(X))
    return predictions

# Define the input feature columns to pass to the UDF

feature_columns = [c for c in big_df.columns if c != "id"]


# Distribute scoring over the cluster

predictions = big_df.select(
  F.col("id"),
  predict_udf(*feature_columns).alias('predicted_species')
)

predictions.show()