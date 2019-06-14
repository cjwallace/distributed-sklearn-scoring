# # Scoring

# We have a pre-trained model file, and we'd like to apply it to
# distributed data.

import joblib
import seaborn as sns

from functools import reduce
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import StructType, FloatType, StructField

# Use the same canned dataset we trained on.

data = sns.load_dataset('iris').drop('species', axis='columns')

# Initialize a spark session

spark = (
    SparkSession
    .builder
    .master('yarn')
    .appName('distribute-sklearn')
    .getOrCreate()
)

sc = spark.sparkContext

# Create a spark dataframe.
# Out in the real world, we wouldn't have a canned dataset, and our data may
# be much larger.
# To replicate that, we'll artificially explode our data to a much larger set.

df = spark.createDataFrame(data)

big_df = reduce(DataFrame.unionAll, (df for i in range(10000)))

big_df.count()

# We could use pandas scalar UDFs, in which case we need a function that takes
# pandas Series and returns pandas series.
# Instead, we'll use the grouped map, which takes a grouped spark
# dataframe and operates on each group as a pandas dataframe, then joins the
# results.
# To do this split, we must create an id to group by. For our purposes, we
# don't care what the groups are - we just want to split the frame up to
# distribute compute.

# FIXME

# Using pandas grouped map UDFs requires specifying the output schema of the
# dataframe returned by the UDF.

output_schema = StructType([
  StructField('sepal_length', FloatType(), False),
  StructField('sepal_width', FloatType(), False),
  StructField('petal_length', FloatType(), False),
  StructField('petal_width', FloatType(), False),
  StructField('predicted_species', FloatType(), False)
])


# Now use pandas UDFs to distribute a function

@pandas_udf(output_schema, PandasUDFType.GROUPED_MAP)
def predict(df):
    """
    Takes a dataframe of iris flower features and returns the same dataframe
    with a predicted species column appended.
    """
    df['predicted_species'] = model.predict(df)
    return df

model = joblib.load('model.pkl')


