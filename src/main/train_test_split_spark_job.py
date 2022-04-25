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
        print("\n\n!!!!Missing SVI In Columns, exiting\n\n")
        sys.exit(-1)
    
        
    columns.remove("SVI")
    columns.insert(0, "SVI")
    df_reordered = df.select(columns)
    df_reordered.show()
    
    input_columns = ["SVI"] 
    output_columns = [col for col in columns if col not in input_columns]    
    
    #print(output_columns)
    x = df_reordered.select(input_columns)
    x.show()

    y = df_reordered.select(output_columns)
    y.show()    

    n_inputs = len(x.columns)
    n_outputs = len(y.columns)
    
    #print(os.getcwd())
     
    nn = npys.NeuralNetworkPyspark(n_inputs, n_outputs)
    
    rdd = df_reordered.rdd
    #rdd.collect()    
    #print(rdd.collect())
    #x_rdd = x.rdd
    #y_rdd = y.rdd
    
    import numpy as np
    #print(rdd.map(lambda x: (np.mean(x[0], axis=0 ) ) ))
    #y_test = y_rdd.map(lambda v: np.mean([float(v[i]) for i in range(len(v))], axis=0))
    #y = np.array(y_rdd.map(lambda v: [float(v[i]) for i in range(len(v))]).collect())
    #y = y_rdd.map(lambda v: [float(v[i]) for i in range(len(v))])
    #y = np.mean(np.array(y.map(lambda v: v)), axis=0)
    #print(y)
    #print(y.collect())
    #y = np.array(y_test)
    
    #y_np_test_mean = np.mean(y_np_test, axis=0)
            

    #working = x_rdd.map(lambda x: float(x[0])).mean()
    #print(working)
    #print(type(working))
    #print(x_rdd.mean())   
    #print(y_rdd.mapValues(lambda y: y).collect())
        
    nn.collectMeansAndStandards(rdd)            
    
   # def standardizeX(self, X):
   #     return (X - self.Xmeans) / self.Xstds    

    #test = (rdd.map(lambda x: (x[:1], (x[1:]) ))).collect()
    #X_means = []
    #for i in range(n_inputs):
    #    X_means.append(rdd.groupBy(lambda x: x[:n_inputs]).map(lambda x: float(x[:1][0][i])).mean())
    
    #Testing standardize and unstandardize on one column
    print(rdd.groupBy(lambda x: x[n_inputs:]).map(lambda x: (float(x[:1][0][0]) - nn.Xmeans[0]) / nn.Xstds[0]).map(lambda x: x * nn.Xstds[0] + nn.Xmeans[0]).collect())    
    
    #print(rdd.groupBy(lambda x: x[n_inputs:]).map(lambda x: float(x[n_inputs:][0][i])).collect())

    #T_means = []
    #for i in range(n_outputs):
    #    T_means.append(rdd.groupBy(lambda x: x[n_inputs:]).map(lambda x: float(x[:1][0][i])).mean())
    #t2  = rdd.groupBy(lambda x: x[:n_inputs]).map(lambda x: float(x[:n_inputs][0][1])).mean()
    #X_means = rdd.groupBy(lambda x: x[:2]).map(lambda x: float(x[:2][0][0])).mean()
    #X_means = (rdd.groupBy(lambda x: x[:2]).map(lambda x: [float(x[:2][0][i]) for i in range(len(x[:2][0]))]).foreach(print)).collect()
    #print(t)
    #print(X_means)
    #print(T_means)
    #nn.train(rdd)
    print("\n\n!!!DONE!!!")
    spark.stop()



