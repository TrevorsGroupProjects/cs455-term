Team 6
Trevor Holland, Rene Martinez, Griffin Zavadil

+++Files+++
<List of files>

+++Instructions for running files+++
NeuralNetworkPyspark.py is a class used to analyze data from RDDs over the Spark framwork. It is not a file that should be run or submitted, but a library file necessary for further analysis.

train_test_split_spark_job.py is a spark job used for tuning hyper-parameters of the neural network and experimenting on the predictive power of SVI in concert with other data. Ultimately, we stuck with our original hypothesis for developing a model and used only SVI to predict other factors.
To run: 
$SPARK_HOME/bin/spark-submit <path to train_test_split_spark_job.py > <hdfs path to merged data file>

predict_from_svi.py is a spark job that will load a pkl stored neural network and evaluate the model stored within the neural network.
To run:
$SPARK_HOME/bin/spark-submit <path to predict_from_svi.py > <path to pkl file of neural network> <hdfs path to merged data file>

These Jupyter Notebooks were developed for the sole purpose of downloading and storing data from ArcGIS Online. In order to run these notebooks you must install the ‘arcgis’ python package available from esri (https://developers.arcgis.com/python/guide/install-older-versions/).
dropout-dpp – Downloads data from ACS Youth School and Work Activity Variables: https://www.arcgis.com/home/item.html?id=5c798c532ad5448ea9e973de8ddf8076 
median-household-income – Downloads data from ACS Median Household Income Variables : https://www.arcgis.com/home/item.html?id=45ede6d6ff7e4cbbbffa60d34227e462
public_private_school_arcgis – Downloads data from US Schools and School District Characteristics: https://www.arcgis.com/home/item.html?id=1577f4b9b594482684952d448aa613c7
total_teens_with_jobs – Downloads data from ACS Labor Force Participation by Age Variables : https://www.arcgis.com/home/item.html?id=60cca9ccc99f4ecfb42c6f2e79f2ec66

Once the package is installed, run the entire notebook to obtain the csv file that is processed by our spark jobs. Here is a breakdown of the csv file produced by each notebook that must be staged on your hdfs:
dropout-dpp -> usa_drop_out_data.csv
median-household-income -> usa_median_household_income.csv
public_private_school_arcgis -> usa_private_schools.csv & usa_public_schools.csv
total_teens_with_jobs -> usa_total_teens_with_jobs.csv


It is highly recommended that each of the above files get staged and processed in their own directories as certain jobs are intending to process files in the entire directory!

Public School Data:
*Note: The output files from this spark job will be in a subdirectory of the output directory specified, called "public_school_out"
$SPARK_HOME/bin/spark-submit <path to public_school_data_spark_job.py> <hostname> <port> <path to input directory> <path to output directory>
Example:
$SPARK_HOME/bin/spark-submit ./src/main/public_school_data_spark_job.py breckenridge 41595 /input /output

Private School Data:
*Note: The output files from this spark job will be in a subdirectory of the output directory specified, called "private_school_out"
$SPARK_HOME/bin/spark-submit <path to private_school_data_spark_job.py> <hostname> <port> <path to input directory> <path to output directory>
Example:
$SPARK_HOME/bin/spark-submit ./src/main/private_school_data_spark_job.py breckenridge 41595 /input /output

SVI Data:
*Note: The output files from this spark job will be in a subdirectory of the output directory specified, called "svi_out"
$SPARK_HOME/bin/spark-submit <path to svi_data_spark_job.py> <hostname> <port> <path to input directory> <path to output directory>
Example:
$SPARK_HOME/bin/spark-submit ./src/main/svi_data_spark_job.py breckenridge 41595 /input /output

Graduation Data:
*Note: The output files from this spark job will be in a subdirectory of the output directory specified, called "grad_out"
$SPARK_HOME/bin/spark-submit <path to graduation_data_spark_job.py> <hostname> <port> <path to input directory> <path to output directory>
Example:
$SPARK_HOME/bin/spark-submit ./src/main/graduation_data_spark_job.py breckenridge 41595 /input /output

Dropout Data:
$SPARK_HOME/bin/spark-submit <path to dropout_data_spark_job.py> <hdfs path to usa_drop_out_data.csv>
Example:
$SPARK_HOME/bin/spark-submit src/main/dropout_data_spark_job.py hdfs://cheyenne:41760/dropout_data/usa_drop_out_data.csv

Teens with Jobs:
$SPARK_HOME/bin/spark-submit <path to teens_with_jobs_data_spark_job.py > <hdfs path to usa_total_teens_with_jobs.csv>
Example:
$SPARK_HOME/bin/spark-submit src/main/teens_with_jobs_data_spark_job.py hdfs://cheyenne:41760/teens_with_jobs_data/usa_total_teens_with_jobs.csv

Median Household Income:
$SPARK_HOME/bin/spark-submit <path to median_household_data_spark_job.py > <hdfs path to usa_median_household_income.csv >
Example:
$SPARK_HOME/bin/spark-submit src/main/median_household_data_spark_job.py hdfs://cheyenne:41760//MedianHouseholdIncome/usa_median_household_income.csv

Public Private School Count By County:
*Note: This job merges data from the same target directory, this is integral in filling missing gaps in data between Urban Sustain and ArcGIS datasets. 
*Note: The data generated from this job is intended to be used in calculating the percentage of schools per county, this data should not be staged for merging into the final data set!
$SPARK_HOME/bin/spark-submit <path to public_private_school_count_by_county_spark_job.py > <hdfs path Directory with Public school or Private school data> <public | private>
Examples:
$SPARK_HOME/bin/spark-submit src/main/public_private_school_count_by_county_spark_job.py hdfs://cheyenne:41760/PublicSchoolData public
$SPARK_HOME/bin/spark-submit src/main/public_private_school_count_by_county_spark_job.py hdfs://cheyenne:41760/PrivateSchoolData private

Public Private School Percentages by County:
*Note This job is expecting files for public and private school counts per county in the same directory. 
$SPARK_HOME/bin/spark-submit <path to public_private_school_percentages_by_county.py > <hdfs path Directory with Public and Private school counts by county data> 
Example:
$SPARK_HOME/bin/spark-submit src/main/public_private_school_percentages_by_county.py hdfs://cheyenne:41760/SchoolCountsByCounty

Dataset Merging:
*Note: It is important to stage the previously processed parts of the data into one directory for this job, it joins each data set by ‘County-Name’
*Note: The output from this job is the dataset that should be used for training a neural network
$SPARK_HOME/bin/spark-submit <path to dataset_merging_spark_job.py > <hdfs path Directory with All Processed data with ‘County-State’ as a key> 
$SPARK_HOME/bin/spark-submit src/main/dataset_merging_spark_job.py hdfs://cheyenne:41760/ProcessedData



