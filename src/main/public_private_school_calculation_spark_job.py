# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 08:25:33 2022

@author: Rene
"""

import os
import sys
import tempfile
import shutil
import re

import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from subprocess import Popen, PIPE


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: public_private_school_calculation_spark_job.py <directory>", file=sys.stderr)
        sys.exit(-1)
        
    #file_path = "hdfs://cheyenne:41760/dropout_data/usa_drop_out_data.csv"

    spark = SparkSession\
        .builder\
        .appName("Get Totals and Averages Of Schools Per District!")\
        .getOrCreate()

    directory_path = sys.argv[1]
    print(directory_path)    

    #This is assuming that you are not targeting an hdfs with an FQDN!
    if "." in directory_path:
        print("\n\n!!!!!!!Currently only targets Directories....EXITING!!!!!!\n\n")
        spark.stop()
        sys.exit(-1)
    
    hadoop_home = os.environ['HADOOP_HOME']
    hadoop_home = hadoop_home + "/bin/hadoop"
    
    process = Popen(f'{hadoop_home} fs -ls {directory_path}', shell=True, stdout=PIPE, stderr=PIPE)
    std_out, std_err = process.communicate()
    list_of_file_names = [fn for fn in std_out.decode().split('\n')[1:]][:-1]
    list_of_file_names = [re.search("\/(?P<FileName>\w+\.\w+)", str(fn)).group("FileName") for fn in list_of_file_names]
    
    arcgis_dfs = []
    dfs = []    
    starting_df = None
    for file_name in list_of_file_names:
        #if ".csv" in file_name:
        df = spark.read.option("header", True).csv(directory_path + "/" + file_name)
        if "County-State" not in df.columns:
            arcgis_dfs.append(df)
        elif starting_df != None:
            start_df = df
        else:
            dfs.append(df)
            
    
    if len(arcgis_dfs) != 0:
        for adf in arcgis_dfs:
            #Drop the schools that don't have a county
            adf = adf.na.drop(subset=["COUNTY"])
            adf = adf.withColumn("County-State", F.concat_ws("-", F.upper(F.col("COUNTY")), F.col("STATE")))
            adf.show()
            
        
            
        
        
    
    
    
