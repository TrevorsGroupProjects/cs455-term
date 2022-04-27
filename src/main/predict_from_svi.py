import os
import sys
import tempfile
import shutil
import pickle

from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import Normalizer
from pyspark.ml.linalg import Vectors
from pyspark.mllib.stat import Statistics
from pyspark.mllib.util import MLUtils

import NeuralNetworkPyspark

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: predict_from_svi.py <neural_network_file> <test_data_file>", file=sys.stderr)
        sys.exit(-1)

    spark = SparkSession\
        .builder\
        .appName("PredictEducation")\
        .getOrCreate()
        
    spark.sparkContext.setLogLevel("WARN")

    nn_path = sys.argv[1]
    input_file = sys.argv[2]
    # Reads into a dataframe.
    # Need to figure out the format.
    #df = spark.read.format("libsvm").load(input_path).cache()
    
    df = spark.read.option("header", True).csv(input_file)
    
    #df = df.drop("County-State")
    
    columns = df.columns
    
    print(columns)
    
    with open(nn_path, 'rb') as nnpkl:
        nn = pickle.load(nnpkl)
        
    print(nn)
    
    #if "SVI" in columns:
    #    columns = columns.remove("SVI")
    #    columns.insert(0, "SVI")
    #    df_reordered = df.select(columns)
    #    df_reordered.show()
    #else:
     #   print("\n\n!!!!Missing SVI In Columns, exiting\n\n")
     #   sys.exit(-1)
        
    

    spark.stop()
