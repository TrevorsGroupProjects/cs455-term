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

    df = df.drop("County-State")

    input_columns = ["SVI"] 
    columns = df.columns

    output_columns = [col for col in columns if col not in input_columns]           
    for input_header in input_columns:
        columns.remove(input_header)
        columns.insert(0, input_header)
    df_reordered = df.select(columns)
    # df_reordered.show()

    n_inputs = len(input_columns)
    n_outputs = len(output_columns)
    train, test = df_reordered.randomSplit([0.8, 0.2], seed=27)

    with open(nn_path, 'rb') as nnpkl:
        nn = pickle.load(nnpkl)
    
    # test_rdd = df.rdd.map(lambda x: (np.array([x[:len(input_columns)]]).astype(np.float), np.array([x[len(input_columns):]]).astype(np.float)))

    train_rdd = train.rdd.map(lambda x: (np.array([x[:n_inputs]]).astype(np.float), np.array([x[n_inputs:]]).astype(np.float)))
    test_rdd = test.rdd.map(lambda x: (np.array([x[:n_inputs]]).astype(np.float), np.array([x[n_inputs:]]).astype(np.float)))

    train_t = np.array(train_rdd.map(lambda x: (x[1])).collect())
    test_t = np.array(test_rdd.map(lambda x: (x[1])).collect())

    # nn.train(train_rdd, num_epochs=10)
    y_train = np.array(nn.use(train_rdd))
    y_test = np.array(nn.use(test_rdd))

    RMSEtrain = np.sqrt( np.mean( y_train - train_t)**2 )
    RMSEtest = np.sqrt( np.mean( y_test - test_t)**2 )

    print(f"Model final training Root Mean Squared Error: {RMSEtrain:.6f}")
    print(f"Model test Root Mean Squared Error: {RMSEtest:.6f}")
    

    spark.stop()
