import os
import sys
import tempfile
import shutil

import pyspark.sql.functions as F
from pyspark.sql import SparkSession


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: state_to_code_mapper.py <file>", file=sys.stderr)
        sys.exit(-1)
        
    #file_path = "hdfs://cheyenne:41760/dropout_data/usa_drop_out_data.csv"

    spark = SparkSession\
        .builder\
        .appName("USA Dropout Refinement")\
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
    
    columns_to_drop = ["Name", "State", "Total Population 16 to 19 Years", "Total_Male_Dropout", "Total_Female_Dropout", "Total_Dropout", "County"]
    df = df.withColumn("Drop_Out_Rate_By_County", F.col("Total_Dropout") / F.col("Total Population 16 to 19 Years")).drop(*columns_to_drop)
    df.show()
     

    spark.stop()
