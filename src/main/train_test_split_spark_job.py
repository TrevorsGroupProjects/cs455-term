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
import pickle as pkl
import re

from pyspark.sql import SparkSession
from pyspark.sql import DataFrame

import NeuralNetworkPyspark as npys
#from NeuralNetworkPyspark impoirt NeuralNetworkPyspark

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: train_test_split_spark_job.py <file>", file=sys.stderr)
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
    # input_columns = ["SVI", "Drop_Out_Rate_By_County", "Graduation-Rate"] 
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

    m = re.search(r'(?P<Path>[\w\W+]+\/)', input_path)

    test_df = test.rdd.toDF()
    test_df.show()
    # test_df.write.option("header", True).csv(m.group("Path") + "test03.csv")
    
    train_rdd = train.rdd.map(lambda x: (np.array([x[:n_inputs]]).astype(np.float), np.array([x[n_inputs:]]).astype(np.float)))
    test_rdd = test.rdd.map(lambda x: (np.array([x[:n_inputs]]).astype(np.float), np.array([x[n_inputs:]]).astype(np.float)))
    

    # print(train_rdd.take(1))
    # print("Train set size:", train_rdd.count())
    # print("Test set size:", test_rdd.count())

    # nn.collectMeansAndStandards(train_rdd, verbose=True)

    def experiment1(train_rdd, test_rdd, n_epochs_choices, n_hidden_units_per_layer_choices,activation_function_choices, learning_rate=0.1):
        header_names = ['epochs', 'nh', 'lr', 'act func', 'RMSE Train', 'RMSE Test']
        train_t = np.array(train_rdd.map(lambda x: (x[1])).collect())
        test_t = np.array(test_rdd.map(lambda x: (x[1])).collect())
        dataF = []
        for n_epochs in n_epochs_choices:
            for n_hidden_units_per_layer in n_hidden_units_per_layer_choices:
                for activation_function in activation_function_choices:
                    nn = npys.NeuralNetworkPyspark(n_inputs, n_outputs, n_hidden_units_per_layer, activation_function)
                    nn.train(train_rdd, n_epochs, learning_rate, verbose=False)
                    y_train = np.array(nn.use(train_rdd))
                    y_test = np.array(nn.use(test_rdd))
                    RMSEtrain = np.sqrt( np.mean( y_train - train_t)**2 )
                    RMSEtest = np.sqrt( np.mean( y_test - test_t)**2 )
                    dataF.append([ n_epochs, n_hidden_units_per_layer, learning_rate, activation_function, RMSEtrain, RMSEtest ])
        return dataF

    
    # nn.train(train_rdd)
    
    # y_test = nn.use(test_rdd)
    # t_test = test_rdd.map(lambda x: (x[1])).collect()

    # #m = re.search(r'(?P<Path>[\w\W+]+\/)', input_path)
    # RMSEtest = np.sqrt( np.mean( (np.array(y_test) - np.array(t_test))**2 ) )

    # print(f"RMSE: {RMSEtest:.4f}")
    #print(y_test[0])
    #print("\n\n")
    #print(test_rdd.map(lambda x: x[n_inputs:]).take(1))
    #print(type(y_test))     
    
    #test_df = test.rdd.toDF()
    #test_df.write.option("header", True).csv(m.group("Path") + "test01.csv")
    # epochs = [10, 50, 100]
    # networks = [[], [3,10,10], [3,5,5], [3, 5, 10, 10, 5], [3, 5, 10, 10, 10, 10]]
    # act_funcs = ['tanh', 'sig']
    epochs = [250]
    networks = [[], [3, 5, 10, 10, 10, 10]]
    act_funcs = ['tanh']

    nn_data = experiment1(train_rdd, test_rdd, epochs, networks, act_funcs)

    print(nn_data)

    # with open("nn03.pkl", 'wb') as nnf:
    #     pkl.dump(nn, nnf, pkl.HIGHEST_PROTOCOL) 

    print("\n\n!!!DONE!!!\n\n")
    spark.stop()



