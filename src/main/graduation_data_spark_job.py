import sys

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

def adjustCounties(county_string):
    if county_string[len(county_string)-7:len(county_string)] == ' County':
        county_string = county_string[0:len(county_string)-7]
    return county_string.upper()

def processGradData(spark, input_folder_path, output_folder):
    # Load Data Into RDDs
    rdds = []
    rdds.append(spark.sparkContext.textFile(f'{input_folder_path}/grad_data2010.csv', minPartitions=15))
    rdds.append(spark.sparkContext.textFile(f'{input_folder_path}/grad_data2011.csv', minPartitions=15))
    rdds.append(spark.sparkContext.textFile(f'{input_folder_path}/grad_data2012.csv', minPartitions=15))
    rdds.append(spark.sparkContext.textFile(f'{input_folder_path}/grad_data2013.csv', minPartitions=15))
    rdds.append(spark.sparkContext.textFile(f'{input_folder_path}/grad_data2014.csv', minPartitions=15))
    rdds.append(spark.sparkContext.textFile(f'{input_folder_path}/grad_data2015.csv', minPartitions=15))
    rdds.append(spark.sparkContext.textFile(f'{input_folder_path}/grad_data2016.csv', minPartitions=15))
    rdds.append(spark.sparkContext.textFile(f'{input_folder_path}/grad_data2017.csv', minPartitions=15))
    rdds.append(spark.sparkContext.textFile(f'{input_folder_path}/grad_data2018.csv', minPartitions=15))
    rdds.append(spark.sparkContext.textFile(f'{input_folder_path}/grad_data2019.csv', minPartitions=15))
    rdds.append(spark.sparkContext.textFile(f'{input_folder_path}/grad_data2020.csv', minPartitions=15))
    rdds.append(spark.sparkContext.textFile(f'{input_folder_path}/grad_data2021.csv', minPartitions=15))

    # Split Entries Into Columns
    for i in range(len(rdds)):
        rdds[i] = rdds[i].map(lambda x: x.split(","), preservesPartitioning=True)

    # Keep Only Relevant Columns:
    # 3 = State Abbreviation
    # 4 = County
    # Column of 'High school graduation raw value' in 2010: 88  
    # Column of 'High school graduation raw value' in 2011: 88  
    # Column of 'High school graduation raw value' in 2012: 93  
    # Column of 'High school graduation raw value' in 2013: 99  
    # Column of 'High school graduation raw value' in 2014: 115 
    # Column of 'High school graduation raw value' in 2015: 115 
    # Column of 'High school graduation raw value' in 2016: 115 
    # Column of 'High school graduation raw value' in 2017: 115 
    # Column of 'High school graduation raw value' in 2018: 128 
    # Column of 'High school graduation raw value' in 2019: 133
    # Column of 'High school graduation raw value' in 2020: 182
    # Column of 'High school graduation raw value' in 2021: 477
    year = 2010
    for i in range(len(rdds)):
        if year == 2010 or year == 2011: rdds[i] = rdds[i].map(lambda x: [x[4], x[3], x[88]], preservesPartitioning=True)
        elif year == 2012: rdds[i] = rdds[i].map(lambda x: [x[4], x[3], x[93]], preservesPartitioning=True)
        elif year == 2013: rdds[i] = rdds[i].map(lambda x: [x[4], x[3], x[99]], preservesPartitioning=True)
        elif year >= 2014 and year <= 2017: rdds[i] = rdds[i].map(lambda x: [x[4], x[3], x[115]], preservesPartitioning=True)
        elif year == 2018: rdds[i] = rdds[i].map(lambda x: [x[4], x[3], x[128]], preservesPartitioning=True)
        elif year == 2019: rdds[i] = rdds[i].map(lambda x: [x[4], x[3], x[133]], preservesPartitioning=True)
        elif year == 2020: rdds[i] = rdds[i].map(lambda x: [x[4], x[3], x[182]], preservesPartitioning=True)
        elif year == 2021: rdds[i] = rdds[i].map(lambda x: [x[4], x[3], x[477]], preservesPartitioning=True)
        year = year + 1

    # Separate Headers From RDD
    for i in range(len(rdds)):
        rdds[i] = rdds[i].zipWithIndex().filter(lambda tup: tup[1] > 0).map(lambda tup: tup[0], preservesPartitioning=True)
        rdds[i] = rdds[i].zipWithIndex().filter(lambda tup: tup[1] > 0).map(lambda tup: tup[0], preservesPartitioning=True)

    # Append all year RDDs
    rdd = []
    year = 2010
    for entry in rdds:
        if year == 2010:
            rdd = entry
        else:
            rdd = rdd + entry
        year = year + 1

    # Remove Non-County Entries
    header = ['COUNTY', 'STATE', 'RATE']
    df = spark.createDataFrame(rdd, header)
    nonstate_list = ['US', 'State Abbreviation', 'state', '']
    noncounty_list = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho',
        'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi',
        'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio',
        'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia',
        'Washington', 'West Virginia', 'Wisconsin', 'Wyoming', '']
    for string in nonstate_list:
        df = df.filter(df.STATE != string)
    for string in noncounty_list:
        df = df.filter(df.COUNTY != string)
    df = df.filter(df.RATE != '')

    # Format The County and State Columns
    rdd = df.rdd.map(lambda x: (adjustCounties(x[0])+"-"+x[1], float(x[2])), preservesPartitioning=True)

    # Write DataFrame to CSV File
    header = ('County-State', 'Graduation-Rate')
    df = spark.createDataFrame(rdd, header)

    df = df.groupBy('County-State').avg('Graduation-Rate')
    df = df.select(col("County-State"), col("avg(Graduation-Rate)").alias("Graduation-Rate"))

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
