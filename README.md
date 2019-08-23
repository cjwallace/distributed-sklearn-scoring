# distributing sklearn

Minimal project distributing _scoring_ with a pretrained sklearn model with pyspark.
Assumes that sklearn, pandas as pyarrow are available on cluster nodes, which will likely require setting specific `PYSPARK_PYTHON` and `PYSPARK_DRIVER_PYTHON`.