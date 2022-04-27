# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 09:32:46 2022

@author: Rene
"""

import os
import sys
import tempfile
import shutil
import numpy as np

from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import Normalizer
from pyspark.ml.linalg import Vectors
from pyspark.mllib.stat import Statistics
from pyspark.mllib.util import MLUtils

import NeuralNetworkPyspark as npys
#from NeuralNetworkPyspark impoirt NeuralNetworkPyspark

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: predict_from_svi.py <file>", file=sys.stderr)
        sys.exit(-1)

    spark = SparkSession\
        .builder\
        .appName("PredictEducation")\
        .getOrCreate()
    
    spark.sparkContext.addPyFile("./src/main/NeuralNetworkPyspark.py")
    spark.sparkContext.setLogLevel("WARN")

    input_path = sys.argv[1]
    # output_path = sys.argv[2]
    
    df = spark.read.option("header", True).csv(input_path)
    
    df = df.drop("County-State")
    
    columns = df.columns
    
    # print(columns)
    
    if "SVI" not in columns:
        print("\n\n!!!!Missing SVI In Columns, exiting\n\n")
        sys.exit(-1)
    
    input_columns = ["SVI"] 
    output_columns = [col for col in columns if col not in input_columns]    
    
    # print(output_columns)
    # x = df_reordered.select(input_columns)
    # x.show()
    # y = df_reordered.select(output_columns)
    # y.show()

    for input_header in input_columns:
        columns.remove(input_header)
        columns.insert(0, input_header)
    df_reordered = df.select(columns)
    # df_reordered.show()

    n_inputs = len(input_columns)
    n_outputs = len(output_columns)
    
    #print(os.getcwd())
     
    nn = npys.NeuralNetworkPyspark(n_inputs, n_outputs)
    # nn.printWeights()

    train, test = df_reordered.randomSplit([0.8, 0.2], seed=42)

    train_rdd = train.rdd.map(lambda x: (np.array([x[:n_inputs]]).astype(np.float), np.array([x[n_inputs:]]).astype(np.float)))
    test_rdd = test.rdd.map(lambda x: (np.array([x[:n_inputs]]).astype(np.float), np.array([x[n_inputs:]]).astype(np.float)))
    

    # print(train_rdd.take(1))
    # print("Train set size:", train_rdd.count())
    # print("Test set size:", test_rdd.count())

    # nn.collectMeansAndStandards(train_rdd, verbose=True)

    nn.train(train_rdd)
    
    y_test = nn.use(test_rdd)
    
    print(y_test[0])
    print("\n\n")
    print(test_rdd.map(lambda x: x[n_inputs:]).take(1))
    
    print("\n\n!!!DONE!!!\n\n")
    spark.stop()



