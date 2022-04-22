#!/bin/bash

# Properties
filepath="./src/main/grifftest.py"
input="/input/test_data"
output="/output/test_out"
hostname="breckenridge"
port="41595"

# Delete output folder
$HADOOP_HOME/bin/hadoop fs -rm -r $output

# Run job
$SPARK_HOME/bin/spark-submit $filepath $input $output $hostname $port

# Show output
echo "Job Output"
echo "----------"
$HADOOP_HOME/bin/hadoop fs -cat $output/*