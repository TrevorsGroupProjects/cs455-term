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

def adjustCounties(county_string):
    if county_string[len(county_string)-7:len(county_string)] == ' County':
        county_string = county_string[0:len(county_string)-7]
    return county_string.upper()

def processGradData(spark, input_folder_path, output_folder):
    # Load Data Into RDD
    rdd = spark.sparkContext.textFile(f'{input_folder_path}', minPartitions=15)

    # Split Entries Into Columns
    rdd = rdd.map(lambda x: x.split(","), preservesPartitioning=True)

    # Keep Only Relevant Columns:
    # 3 = State Abbreviation
    # 4 = County
    # 88 = Graduation Rate
    rdd = rdd.map(lambda x: [x[3], x[4], x[88]], preservesPartitioning=True)

    # Separate Headers From RDD
    rdd = rdd.zipWithIndex().filter(lambda tup: tup[1] > 0).map(lambda tup: tup[0], preservesPartitioning=True)
    header = rdd.first()
    rdd = rdd.zipWithIndex().filter(lambda tup: tup[1] > 0).map(lambda tup: tup[0], preservesPartitioning=True)

    # Remove Non-County Entries
    df = spark.createDataFrame(rdd, header)
    nonstate_list = ['US', 'State Abbreviation', 'state']
    noncounty_list = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho',
        'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi',
        'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio',
        'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia',
        'Washington', 'West Virginia', 'Wisconsin', 'Wyoming']
    for string in nonstate_list:
        df = df.filter(df.state != string)
    for string in noncounty_list:
        df = df.filter(df.county != string)

    # Format The County Column
    rdd = df.rdd.map(lambda x: (x[0], adjustCounties(x[1]), x[2]), preservesPartitioning=True)

    # Save RDD to HDFS
    rdd = rdd.coalesce(1)
    rdd.saveAsTextFile(output_folder)

    # Return the RDD
    return rdd


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: predict_from_svi.py <hostname> <port> <input_path> <output_path>", file=sys.stderr)
        sys.exit(-1)

    # Handle Arguments
    input_path = sys.argv[3]
    hdfs = "spark://" + sys.argv[1] + ":" + sys.argv[2]
    output_path = sys.argv[4]

    # Create Spark Session
    spark = SparkSession.builder.master(hdfs).appName("PredictFromSVI").getOrCreate()

    # Read and Process Graduation Data into RDD
    grad_rdd = processGradData(spark, f"{input_path}/grad_data", f"{output_path}/grad_out")

    # Reads into a dataframe.
    # Need to figure out the format.
    # df = spark.read.format("libsvm").load(input_path).cache()

    # dataFrame = spark.createDataFrame([
    #     (0, Vectors.dense([1.0, 0.5, -1.0]),),
    #     (1, Vectors.dense([2.0, 1.0, 1.0]),),
    #     (2, Vectors.dense([4.0, 10.0, 2.0]),)
    # ], ["id", "features"])

    # newFrame.coalesce(1).write.csv(output_path) # ouptut to file

    spark.stop()
