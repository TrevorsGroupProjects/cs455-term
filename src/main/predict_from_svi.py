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

import NeuralNetworkPyspark

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: predict_from_svi.py <file>", file=sys.stderr)
        sys.exit(-1)

    spark = SparkSession\
        .builder\
        .appName("PredictEducation")\
        .getOrCreate()

    input_path = sys.argv[1]
    # output_path = sys.argv[2]
    # Reads into a dataframe.
    # Need to figure out the format.
    #df = spark.read.format("libsvm").load(input_path).cache()
    
    df = spark.read.csv(input_file)
    
    df = df.drop("County-State")
    
    columns = df.columns
    
    print(columns)
    
    
    

    spark.stop()
