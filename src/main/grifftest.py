import sys
import time
from pyspark.sql import SparkSession


# Handle Arguments
input_folder_path = sys.argv[1]
hdfs = "spark://" + sys.argv[3] + ":" + sys.argv[4]
output_folder = sys.argv[2]

# Create Spark Session
spark = SparkSession.builder.master(hdfs).appName("GriffinApplication").getOrCreate()

# Load Data Into RDD
rdd = spark.sparkContext.textFile(f'{input_folder_path}', minPartitions=15)
print("Read Data Into RDD Successfully")

# Split Entries Into Columns
rdd = rdd.map(lambda x: x.split(","), preservesPartitioning=True)

# Keep Only Relevant Columns:
# 3 = State Abbreviation
# 4 = County
# 88 = Graduation Rate
rdd = rdd.map(lambda x: [x[3], x[4], x[88]], preservesPartitioning=True)

# Separate Headers From RDD
header = rdd.first()
rdd = rdd.zipWithIndex().filter(lambda tup: tup[1] > 0).map(lambda tup: tup[0], preservesPartitioning=True)
rdd = rdd.zipWithIndex().filter(lambda tup: tup[1] > 0).map(lambda tup: tup[0], preservesPartitioning=True)

# Save RDD to HDFS
rdd.saveAsTextFile(output_folder)
print("Successfully Saved RDD as Text File")