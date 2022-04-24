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
        print("Usage: public_private_school_percentage_by_county_spark_job.py <directory>", file=sys.stderr)
        sys.exit(-1)
        
    #file_path = "hdfs://cheyenne:41760/dropout_data/usa_drop_out_data.csv"
    
    spark = SparkSession\
        .builder\
        .appName("Get Averages Of Schools Per District!")\
        .getOrCreate()

    directory_path = sys.argv[1]
    #print(directory_path)    

    #This is assuming that you are not targeting an hdfs with an FQDN!
    if "." in directory_path:
        print("\n\n!!!!!!!Currently only targets Directories....EXITING!!!!!!\n\n")
        spark.stop()
        sys.exit(-1)
    
    total_school_count = "Total-School-Count"
    percentage_private_school = "Percentage-Private-School"
    percentage_public_school = "Percentage-Public-School"
     
    hadoop_home = os.environ['HADOOP_HOME']
    hadoop_home = hadoop_home + "/bin/hadoop"
    
    process = Popen(f'{hadoop_home} fs -ls {directory_path}', shell=True, stdout=PIPE, stderr=PIPE)
    std_out, std_err = process.communicate()
    list_of_file_names = [fn for fn in std_out.decode().split('\n')[1:]][:-1]
    list_of_file_names = [re.search("\/(?P<FileName>\w+\.\w+)", str(fn)).group("FileName") for fn in list_of_file_names]
    
    dfs = []    
    starting_df = None
    for file_name in list_of_file_names:
        #if ".csv" in file_name:
        df = spark.read.option("header", True).csv(directory_path + "/" + file_name)
        dfs.append(df)
    
    starting_df = dfs[0]
    for i in range(1, len(dfs)):
        starting_df = starting_df.join(dfs[i], ["County-State"])
   
   
    starting_df = starting_df.withColumn(total_school_count, int(F.col("Public") + int(F.col("Private"))))\
        .withColumn(percentage_private_school, F.col("Private") / F.col(total_school_count))\
        .withColumn(percentage_public_school, F.col("Public") / F.col(total_school_count))

    starting_df.show() 
   
    columns = starting_df.columns()
    
    print(columns)
    
   
    #distinct_counties = starting_df.select("County-State").distinct().collect()
    #print(len(distinct_counties))
    #for df in dfs:
    #    distinct_counties_in_df = df.select("County-State").distinct().collect()
    #    missing_counties = [county_state[0] for county_state in distinct_counties_in_df if county_state not in distinct_counties]
    #    filtered_df = df.filter(F.col("County-State").isin(missing_counties))
        #filtered_df.show()
    #    starting_df = starting_df.union(filtered_df)
    
    #starting_df.show()
    #starting_df.write.option("header", True).csv(directory_path + "/MergedArcGISUrban")
    #count = starting_df.groupBy("County-State").agg()
    #counts = starting_df.groupBy("County-State").count()
    #counts = counts.withColumn(new_column_name, F.col("count"))
    #counts = counts.select("County-State", new_column_name)
  
    #counts.coalesce(1).write.option("header", True).csv(directory_path + "/MergedArcGISUrban")
    
        
    spark.stop()           
        
        
    
    
    
