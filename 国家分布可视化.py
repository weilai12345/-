import matplotlib.pyplot as plt
import matplotlib as mpl
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# 设置中文字体
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']
mpl.rcParams['axes.unicode_minus'] = False

# 初始化Spark会话
spark = SparkSession.builder \
    .appName("CountryPieChart") \
    .config("spark.sql.shuffle.partitions", "100") \
    .getOrCreate()

try:
    df = spark.read.parquet("E:/master/data_mining/30G_data_new/30G_data_new") \
        .select(
            F.trim(F.col("country")).alias("country") 
        ).filter(
            F.col("country").isNotNull()  
        )

    country_stats = df.groupBy("country").count() \
        .orderBy(F.desc("count")) \
        .limit(10) \
        .toPandas()
    
    total = df.count()
    top_total = country_stats['count'].sum()
    if total > top_total:
        country_stats = country_stats.append(
            {"country": "其他", "count": total - top_total}, 
            ignore_index=True
        )

    plt.figure(figsize=(12, 12))
    wedges, texts, autotexts = plt.pie(
        country_stats['count'],
        labels=country_stats['country'],
        colors=plt.cm.tab20.colors,
        autopct=lambda p: f'{p:.1f}%' if p >= 2 else '',
        pctdistance=0.85,
        startangle=90,
        wedgeprops={'width': 0.4, 'edgecolor': 'white', 'linewidth': 1}
    )
    
    plt.text(0, 0, f"总用户数\n{total:,}", 
             ha='center', va='center', fontsize=14, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.legend(
        wedges,
        [f"{row['country']} ({row['count']:,})" for _, row in country_stats.iterrows()],
        title="国家/地区分布",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize=10
    )
    
    plt.title("国家分布比例", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig('country_distribution_pie.png', dpi=300, bbox_inches='tight')
    plt.show()

except Exception as e:
    print(f"执行错误: {str(e)}")
finally:
    spark.stop()