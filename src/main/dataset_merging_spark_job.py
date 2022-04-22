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
        print("Usage: teens_with_jobs_data_spark_job.py <directory>", file=sys.stderr)
        sys.exit(-1)
        
    #file_path = "hdfs://cheyenne:41760/dropout_data/usa_drop_out_data.csv"

    spark = SparkSession\
        .builder\
        .appName("Merge All DataSets!")\
        .getOrCreate()

    directory_path = sys.argv[1]
    
    #This is assuming that you are not targeting an hdfs with an FQDN!
    if "." in directory_path:
        print("\n\n!!!!!!!Currently only targets Directories....EXITING!!!!!!\n\n")
        spark.stop()
        sys.exit(-1)
    
    process = Popen(f'fs -ls {directory_path}', shell=True, stdout=PIPE, stderr=PIPE)
    std_out, std_err = process.communicate()
    list_of_file_names = [fn.split(' ')[-1].split('/')[-1] for fn in std_out.decode().split('\n')[1:]][:-1]
    print(list_of_file_names)
    
    #Add a new column for the county and state postal abbrev
    #df["County-State"] = df["County"] + state_by_state_postal_codes[df["State"]]
    #from itertools import chain    

    #mapping_expr = F.create_map([F.lit(x) for x in chain(*state_by_state_postal_codes.items())])
    
    #df = df.withColumn("County-State", F.concat_ws("-", F.trim(F.col("County")), mapping_expr.getItem(F.col("State"))))
    #df.show()
    
    #Column Name Reference
    #"B23027_002E": "Population age 16 to 19 years", 
    #"B23027_003E": "Population age 16 to 19 years who worked in the past 12 months"
    
    #All Caps 'Name' because that is the actual name of the column
    #columns_to_drop = ["NAME", "State", "County", "B23027_002E", "B23027_003E"]
    #df = df.withColumn("Percentage of Teens Who Work Part or Full time", F.col("B23027_003E") / F.col("B23027_002E")).drop(*columns_to_drop)
    #df.show()
    
    #Save the new dataframe as a text file that is similar to the other input data
    #m = re.search(r'(?P<Path>[\w\W+]+\/)', input_path)
    #df.coalesce(1).write.format("text").option("header", "false").mode("append").save(m.group('Path') + "ProcessedDropOutRatesPerCounty") 
    #df.write.csv(m.group('Path') + "ProcessedPercentageOfTeensWithJobs")
    

    spark.stop()