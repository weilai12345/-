from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.sql.types import DoubleType, LongType, IntegerType

spark = SparkSession.builder \
    .appName("Data_Quality_Assessment") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.sql.files.maxPartitionBytes", "67108864") \
    .getOrCreate()

# 加载数据
folder_path = "E:/master/data_mining/10G_data_new/10G_data_new"
df = spark.read.parquet(folder_path)

# 数值类型字段的异常值处理
def detect_and_handle_outliers(df, column_name):   
    # 计算四分位数
    quantiles = df.approxQuantile(column_name, [0.25, 0.75], 0.01)  # 降低误差
    Q1, Q3 = quantiles[0], quantiles[1]
    IQR = Q3 - Q1

    # 调整异常值范围
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # 计算异常值
    anomaly_count = df.filter((col(column_name) < lower_bound) | (col(column_name) > upper_bound)).count()
    total_count = df.count()
    anomaly_ratio = anomaly_count / total_count if total_count > 0 else 0
    
    print(f"列 {column_name} 的异常值数量: {anomaly_count}")
    print(f"列 {column_name} 的异常值比例: {anomaly_ratio:.4f}")

    # 替换异常值为 None
    return df.withColumn(column_name, 
                         when((col(column_name) < lower_bound) | (col(column_name) > upper_bound), None)
                         .otherwise(col(column_name)))

# 处理列：确保列名正确
columns_to_process = ['age', 'income']  

for column_name in columns_to_process:
    print(f"正在处理列: {column_name}")
    df = detect_and_handle_outliers(df, column_name)

# 结束 Spark 会话
spark.stop()