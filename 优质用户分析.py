from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_date, datediff, when, from_json, expr, ceil
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType
from pyspark.sql.window import Window
import pyspark.sql.functions as F
from pyspark.ml.feature import QuantileDiscretizer

# 初始化优化配置的Spark会话
spark = SparkSession.builder \
    .appName("OptimizedHighValueUser") \
    .config("spark.driver.memory", "16g") \
    .config("spark.executor.memory", "16g") \
    .config("spark.sql.shuffle.partitions", "400") \
    .config("spark.default.parallelism", "400") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .config("spark.sql.adaptive.skewJoin.enabled", "true") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.kryoserializer.buffer.max", "512m") \
    .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC") \
    .config("spark.network.timeout", "600s") \
    .getOrCreate()

# 定义JSON结构模式
login_schema = StructType([StructField("login_count", IntegerType())])
purchase_schema = StructType([
    StructField("avg_price", DoubleType()),
    StructField("total_purchases", IntegerType())
])

try:
    # ========== 数据加载优化 ==========
    # 只读取必要字段，提前过滤无效数据
    df = spark.read.parquet("E:/master/data_mining/30G_data_new/30G_data_new") \
        .select("id", "last_login", "login_history", "purchase_history") \
        .filter("id IS NOT NULL AND last_login IS NOT NULL")
    
    # ========== 并行化JSON解析 ==========
    # 使用selectExpr实现并行解析
    df = df.selectExpr(
        "id",
        "to_timestamp(last_login, 'yyyy-MM-dd\"T\"HH:mm:ssXXX') as last_login",
        "from_json(login_history, 'login_count INT') as login_info",
        "from_json(purchase_history, 'avg_price DOUBLE, total_purchases INT') as purchase_info"
    )
    
    # ========== 字段提取与类型转换 ==========
    df = df.select(
        "id",
        "last_login",
        F.coalesce(col("login_info.login_count"), F.lit(0)).alias("frequency"),
        (F.coalesce(col("purchase_info.avg_price"), F.lit(0.0)) * 
         F.coalesce(col("purchase_info.total_purchases"), F.lit(0))).alias("total_spent")
    ).persist()  # 缓存中间结果
    
    # ========== 计算Recency ==========
    df = df.withColumn(
        "recency",
        F.when(datediff(current_date(), col("last_login")) < 0, 0)
        .otherwise(datediff(current_date(), col("last_login")))
    ).na.fill({"recency": 365})
    
    # ========== 批量分位数分箱 ==========
    # 使用QuantileDiscretizer进行高效分箱
    discretizer = QuantileDiscretizer(
        numBuckets=5,
        inputCols=["recency", "frequency", "total_spent"],
        outputCols=["recency_score", "frequency_score", "total_spent_score"],
        relativeError=0.01  # 提高计算速度
    )
    df = discretizer.fit(df).transform(df)
    
    # 反转recency得分（值越小得分越高）
    df = df.withColumn("recency_score", 6 - col("recency_score"))
    
    # ========== 加权得分计算 ==========
    df = df.withColumn(
        "total_rfm_score",
        (col("recency_score") * 0.4) + 
        (col("frequency_score") * 0.3) + 
        (col("total_spent_score") * 0.3)
    )
    
    # ========== 动态阈值计算优化 ==========
    # 使用抽样数据计算阈值
    sample_df = df.sample(fraction=0.1, seed=42)
    threshold = sample_df.approxQuantile(
        "total_rfm_score", 
        probabilities=[0.9999], 
        relativeError=0.001
    )[0]
    
    # ========== 结果筛选与缓存 ==========
    high_value_users = df.filter(col("total_rfm_score") >= threshold) \
                       .persist()
    
    # 触发Action并打印结果
    print("===== 高价值用户样例 =====")
    high_value_users.select("recency_score", "frequency_score", "total_spent_score").show(5)
    print(f"优质用户数量: {high_value_users.count()}")
    
    # 可选：保存结果
    # high_value_users.write.parquet("output_path", mode="overwrite")

except Exception as e:
    print(f"处理过程中发生错误: {str(e)}")
    raise

finally:
    # 释放缓存
    if 'df' in locals():
        df.unpersist()
    if 'high_value_users' in locals():
        high_value_users.unpersist()
    spark.stop()