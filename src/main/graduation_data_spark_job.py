import sys

from pyspark.sql import SparkSession

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
    rdd = rdd.map(lambda x: [x[4], x[3], x[88]], preservesPartitioning=True)

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

    # Format The County and State Columns
    rdd = df.rdd.map(lambda x: (adjustCounties(x[0])+"-"+x[1], x[2]), preservesPartitioning=True)

    # Write DataFrame to CSV File
    header = ('County-State', 'Graduation-Rate')
    df = spark.createDataFrame(rdd, header)
    df = df.repartition(1)
    df.write.csv(output_folder, header=True)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: graduation_data_spark_job.py <hostname> <port> <input_path> <output_path>", file=sys.stderr)
        sys.exit(-1)

    # Handle Arguments
    input_path = sys.argv[3]
    hdfs = "spark://" + sys.argv[1] + ":" + sys.argv[2]
    output_path = sys.argv[4]

    # Create Spark Session
    spark = SparkSession.builder.master(hdfs).appName("ProcessGraduationData").getOrCreate()

    # Process Graduation Data and Output To HDFS
    processGradData(spark, f"{input_path}/grad_data", f"{output_path}/grad_out")

    spark.stop()
