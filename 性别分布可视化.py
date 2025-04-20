import matplotlib.pyplot as plt
import matplotlib as mpl
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import when, col

# 设置中文字体
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
mpl.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 初始化Spark
spark = SparkSession.builder \
    .appName("RawGenderDistribution") \
    .config("spark.sql.shuffle.partitions", "100") \
    .getOrCreate()

#数据加载与预处理
df = spark.read.parquet("E:/master/data_mining/30G_data_new/30G_data_new") \
    .select(
        F.trim(F.col("gender")).alias("gender"))

#性别分布统计
gender_stats = df.groupBy("gender").count() \
    .orderBy(F.desc("count")) \
    .toPandas()

#创建可视化
plt.figure(figsize=(12, 6))

bars = plt.bar(
    gender_stats['gender'],
    gender_stats['count'],
    color='#2c7fb8',
    edgecolor='white',
    width=0.7
)

plt.xticks(
    rotation=45,
    ha='right',
    fontsize=10,
    wrap=True  
)

# 添加数据标签
max_count = gender_stats['count'].max()
for idx, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2.,
        height + max_count*0.02,
        f'{height:,}',
        ha='center',
        va='bottom',
        rotation=90 if len(gender_stats) > 5 else 0,  
        fontsize=8
    )


plt.title("性别分布", pad=20, fontsize=14)
plt.ylabel("用户数量", labelpad=10)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()

# 保存输出
plt.savefig('raw_gender_distribution.png', dpi=300, bbox_inches='tight')
spark.stop()
plt.show()
