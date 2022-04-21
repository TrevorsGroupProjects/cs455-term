import os
import sys
import tempfile
import shutil

from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import Normalizer
from pyspark.ml.linalg import Vectors
from pyspark.mllib.stat import Statistics
from pyspark.mllib.util import MLUtils

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: state_to_code_mapper.py <file>", file=sys.stderr)
        sys.exit(-1)
        
    #file_path = "hdfs://cheyenne:41760/dropout_data/usa_drop_out_data.csv"

    spark = SparkSession\
        .builder\
        .appName("USA Dropout Refinement")\
        .getOrCreate()

    input_path = sys.argv[1]
    print(input_path)
    # output_path = sys.argv[2]
    # Reads into a dataframe.
    # Need to figure out the format.
    #df = spark.read.format("libsvm").load(input_path).cache()

    # dataFrame = spark.createDataFrame([
    #     (0, Vectors.dense([1.0, 0.5, -1.0]),),
    #     (1, Vectors.dense([2.0, 1.0, 1.0]),),
    #     (2, Vectors.dense([4.0, 10.0, 2.0]),)
    # ], ["id", "features"])

    # newFrame.coalesce(1).write.csv(output_path) # ouptut to file

    spark.stop()