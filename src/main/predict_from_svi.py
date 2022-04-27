import os
import sys
import tempfile
import shutil
import pickle
import numpy as np

from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import Normalizer
from pyspark.ml.linalg import Vectors
from pyspark.mllib.stat import Statistics
from pyspark.mllib.util import MLUtils

import NeuralNetworkPyspark

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: predict_from_svi.py <neural_network_file> <test_data_file>", file=sys.stderr)
        sys.exit(-1)

    spark = SparkSession\
        .builder\
        .appName("PredictEducation")\
        .getOrCreate()
        
    spark.sparkContext.setLogLevel("WARN")
    spark.sparkContext.addPyFile("./src/main/NeuralNetworkPyspark.py")

    nn_path = sys.argv[1]
    input_file = sys.argv[2]
    
    df = spark.read.option("header", True).csv(input_file)
    

    input_columns = ["SVI"] 
    columns = df.columns

    output_columns = [col for col in columns if col not in input_columns]           
 
    
    with open(nn_path, 'rb') as nnpkl:
        nn = pickle.load(nnpkl)
        
    test_rdd = df.rdd.map(lambda x: (np.array([x[:len(input_columns)]]).astype(np.float), np.array([x[len(input_columns):]]).astype(np.float)))
    y_test = nn.use(test_rdd)
    #print(y_test)
    
    print(nn.cost_history)
    print("\n\n")
    print(nn.acc_history)

    spark.stop()
