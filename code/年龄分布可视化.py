import matplotlib.pyplot as plt
import matplotlib as mpl
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import when, col

# 设置中文字体（必须在绘图前设置）
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
mpl.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 初始化 Spark 会话
spark = SparkSession.builder \
    .appName("AgeDistributionVisualization") \
    .config("spark.driver.memory", "16g") \
    .config("spark.executor.memory", "16g") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.files.maxPartitionBytes", "128m") \
    .config("spark.default.parallelism", "200") \
    .getOrCreate()

# 数据加载
folder_path = "E:/master/data_mining/30G_data_new/30G_data_new"
df = spark.read.parquet(folder_path).select("age")

# 年龄段分组计算
df_age_groups = df.withColumn(
    "age_group",
    when((col("age") < 30), "青年")
    .when((col("age") >= 30) & (col("age") < 60), "中年")
    .otherwise("老年")
)

# 使用Spark聚合
age_counts = df_age_groups.groupBy("age_group").count().orderBy("age_group").collect()

# 生成可视化数据
age_groups = [row["age_group"] for row in age_counts]
counts = [row["count"] for row in age_counts]

# 可视化
plt.figure(figsize=(10, 6), dpi=100)
bars = plt.bar(age_groups, counts, 
               color=['#4C72B0', '#55A868', '#C44E52', '#8172B2'],
               width=0.6)

# 添加数值标签
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:,}',
             ha='center', va='bottom')

plt.title('用户年龄段分布', fontsize=14, pad=20)
plt.xlabel('年龄段', fontsize=12)
plt.ylabel('用户数量', fontsize=12)
plt.xticks(fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.7)

spark.stop()
plt.tight_layout()
plt.show()