from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder \
    .appName("Data_Quality_Assessment") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.sql.files.maxPartitionBytes", "67108864") \
    .config("spark.executor.processTreeMetrics.enabled", "false") \
    .config("spark.driver.processTreeMetrics.enabled", "false") \
    .getOrCreate()

# 设置日志级别为 WARN
spark.sparkContext.setLogLevel("WARN")

# 数据路径
folder_path = "E:/master/data_mining/10G_data_new/10G_data_new"

# 按分区加载数据（增加分区数）
df = spark.read.parquet(folder_path).repartition(50)

# 统计缺失值数量
missing_data = df.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns])

# 显示缺失值统计
missing_data.show()
