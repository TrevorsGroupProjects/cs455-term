# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 21:01:18 2022

@author: Rene
"""
import os
import sys
import tempfile
import shutil
import re

import pyspark.sql.functions as F
from pyspark.sql import SparkSession


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: mean_household_data_spark_job.py <file>", file=sys.stderr)
        sys.exit(-1)
        
    #file_path = "hdfs://cheyenne:41760/dropout_data/usa_drop_out_data.csv"

    spark = SparkSession\
        .builder\
        .appName("USA Median Household Refinement")\
        .getOrCreate()

    input_path = sys.argv[1]
    
    if ".csv" not in input_path:
        print("\n\n!!!!!!!Currently only targets .csv files....EXITING!!!!!!\n\n")
        spark.stop()
        sys.exit(-1)
        
    df = spark.read.option("header", True).csv(input_path)
    
    
    state_by_state_postal_codes = {"Alabama": "AL", "Alaska": "AK", "Arizona" :	"AZ", 
                                   "Arkansas" : "AR",
                                   "California": "CA",
                                   "Colorado": "CO",
                                   "Connecticut": "CT",
                                   "Delaware":	"DE",
                                   "Ohio": "OH",
                                   "Florida": 	"FL",
                                   "Georgia": "GA",
                                   "Hawaii": "HI",
                                   "Idaho":	"ID",
                                   "Illinois" : "IL",
                                   "Indiana": "IN",
                                   "Iowa":	"IA",
                                   "Kansas": "KS",
                                   "Kentucky":	"KY",
                                   "Louisiana":"LA",
                                   "Maine": "ME",
                                   "Maryland":	"MD",
                                   "Massachusetts": "MA",
                                   "Michigan": "MI",
                                   "Minnesota": "MN",
                                   "Mississippi": "MS",
                                   "Missouri": "MO",
                                   "Montana": "MT",
                                   "Nebraska": "NE",
                                   "Nevada": "NV",
                                   "New Hampshire": "NH",
                                   "New Jersey": "NJ",
                                   "New Mexico": "NM",
                                   "New York":	"NY",
                                   "North Carolina": "NC",
                                   "North Dakota": "ND",
                                   "Oklahoma": "OK",
                                   "Oregon": "OR",
                                   "Pennsylvania": "PA",
                                   "Rhode Island": "RI",
                                   "South Carolina": "SC",
                                   "South Dakota": "SD",
                                   "Tennessee": "TN",
                                   "Texas": "TX",
                                   "Utah": "UT",
                                   "Vermont": "VT",
                                   "Virginia": "VA",
                                   "Washington": "WA",
                                   "West Virginia": "WV",
                                   "Wisconsin": "WI",
                                   "Wyoming": "WY"}
    
    
    
    #Add a new column for the county and state postal abbrev
    #df["County-State"] = df["County"] + state_by_state_postal_codes[df["State"]]
    from itertools import chain    

    mapping_expr = F.create_map([F.lit(x) for x in chain(*state_by_state_postal_codes.items())])
    
    df = df.withColumn("County", F.regexp_replace(F.col("NAME"), 'County', ''))
    df.show()
    
    df = df.withColumn("County-State", F.concat_ws("-", F.upper(F.trim(F.col("County"))), mapping_expr.getItem(F.col("State"))))
    df.show()
    
    #NAME (type: esriFieldTypeString, alias: Name, SQL Type: sqlTypeOther, length: 255, nullable: true, editable: true)
    #State (type: esriFieldTypeString, alias: State, SQL Type: sqlTypeOther, length: 255, nullable: true, editable: true)
    #B19049_001E (type: esriFieldTypeDouble, alias: Median Household Income in past 12 months (inflation-adjusted dollars to last year of 5-year range), SQL Type: sqlTypeOther, nullable: true, editable: true)
    #columns_to_select("")
    
        
    #Save the new dataframe as a csv
    #m = re.search(r'(?P<Path>[\w\W+]+\/)', input_path)
    #df.coalesce(1).write.format("text").option("header", "false").mode("append").save(m.group('Path') + "ProcessedDropOutRatesPerCounty") 
    #df.write.option("header", True).csv(m.group('Path') + "ProcessedMedianIncomePerCounty")
    

    spark.stop()