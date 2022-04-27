import sys

from pyspark.sql import SparkSession

def processData(spark, input_folder_path, output_folder):
    # Load Data Into DataFrame
    df = spark.read.option("multiline",True).json(f'{input_folder_path}')

    # Remove All Unneeded Columns
    df = df.select(df.COUNTY, df.ST_ABBR.alias("STATE"), df.RPL_THEMES.alias("SVI"))

    # Remove Entries With Missing Data
    df = df.filter(df.COUNTY != '')
    df = df.filter(df.STATE != '')
    df = df.filter(df.SVI.isNotNull())

    # Format County-State Column
    rdd = df.rdd.map(lambda x: (x[0].upper()+"-"+x[1], float(x[2])), preservesPartitioning=True)

    # Convert Back To DataFrame With Headers
    header = ('County-State', 'SVI')
    df = spark.createDataFrame(rdd, header)

    # Output DataFrame to HDFS
    df = df.repartition(1)
    df.write.csv(output_folder, header=True)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: svi_data_spark_job.py <hostname> <port> <input_path> <output_path>", file=sys.stderr)
        sys.exit(-1)

    # Handle Arguments
    input_path = sys.argv[3]
    hdfs = "spark://" + sys.argv[1] + ":" + sys.argv[2]
    output_path = sys.argv[4]

    # Create Spark Session
    spark = SparkSession.builder.master(hdfs).appName("ProcessSVIData").getOrCreate()

    # Process Public School Data and Output To HDFS
    processData(spark, f"{input_path}/svi_data", f"{output_path}/svi_out")

    spark.stop()
