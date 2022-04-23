# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 09:32:46 2022

@author: Rene
"""

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
    
    spark.sparkContext.addPyFile("NeuralNetworkPyspark.py")
    

    input_path = sys.argv[1]
    # output_path = sys.argv[2]
    # Reads into a dataframe.
    # Need to figure out the format.
    #df = spark.read.format("libsvm").load(input_path).cache()
    
    df = spark.read.option("header", True).csv(input_path)
    
    df = df.drop("County-State")
    
    columns = df.columns
    
    print(columns)
    
    if "SVI" not in columns:
        #print("\n\n!!!!Missing SVI In Columns, exiting\n\n")
        sys.exit(-1)
    
        
    columns.remove("SVI")
    columns.insert(0, "SVI")
    df_reordered = df.select(columns)
    df_reordered.show()
    
    input_columns = ["SVI"] 
    output_columns = [col for col in columns if col not in input_columns]    

    x = df_reordered.select(input_columns)
    x.show()

    y = df_reordered.select(output_columns)
    y.show()    

    n_inputs = len(x.columns)
    n_outputs = len(y.columns)
    
    print(os.getcwd())
     
    nn = npys.NeuralNetworkPyspark(n_inputs, n_outputs)
    
    rdd = df_reordered.rdd
    #rdd.collect()    
    print(rdd.collect())
    
    nn.collectMeansAndStandards(rdd)            

    spark.stop()



