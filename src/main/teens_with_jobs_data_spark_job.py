import os
import sys
import tempfile
import shutil
import re

import pyspark.sql.functions as F
from pyspark.sql import SparkSession


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: teens_with_jobs_data_spark_job.py <file>", file=sys.stderr)
        sys.exit(-1)
        
    #file_path = "hdfs://cheyenne:41760/dropout_data/usa_drop_out_data.csv"

    spark = SparkSession\
        .builder\
        .appName("Percentage of Teens with Jobs Refinement")\
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
    
    df = df.withColumn("County-State", F.concat_ws("-", F.trim(F.col("County")), mapping_expr.getItem(F.col("State"))))
    df.show()
    
    #Column Name Reference
    #"B23027_002E": "Population age 16 to 19 years", 
    #"B23027_003E": "Population age 16 to 19 years who worked in the past 12 months"
    
    #All Caps 'Name' because that is the actual name of the column
    columns_to_drop = ["NAME", "State", "County", "B23027_002E", "B23027_003E"]
    df = df.withColumn("Percentage of Teens Who Work Part or Full time", F.col("B23027_003E") / F.col("B23027_002E")).drop(*columns_to_drop)
    df.show()
    
    #Save the new dataframe as a text file that is similar to the other input data
    m = re.search(r'(?P<Path>[\w\W+]+\/)', input_path)
    #df.coalesce(1).write.format("text").option("header", "false").mode("append").save(m.group('Path') + "ProcessedDropOutRatesPerCounty") 
    df.write.csv(m.group('Path') + "ProcessedPercentageOfTeensWithJobs")
    

    spark.stop()
