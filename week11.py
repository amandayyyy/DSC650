cat << 'EOF' > games_ml.py
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import happybase

# -----------------------------
# STEP 1: Spark Session
# -----------------------------
spark = (SparkSession.builder
         .appName("GamesML")
         .enableHiveSupport()
         .getOrCreate())

# -----------------------------
# STEP 2: Load Hive Table
# -----------------------------
df = spark.sql("SELECT * FROM games")

# Drop rows with nulls to avoid ML errors
df = df.na.drop()

# -----------------------------
# STEP 3: Assemble Feature Vector
# -----------------------------
feature_columns = [
    "year_published",
    "min_players",
    "max_players",
    "play_time",
    "min_age",
    "users_rated",
    "bgg_rank",
    "complexity_average",
    "owned_users"
]

assembler = VectorAssembler(
    inputCols=feature_columns,
    outputCol="features"
)

data = assembler.transform(df).select("features", "rating_average")

# -----------------------------
# STEP 4: Train/Test Split
# -----------------------------
train, test = data.randomSplit([0.7, 0.3], seed=42)

# -----------------------------
# STEP 5: Train ML Model
# -----------------------------
rf = RandomForestRegressor(
    labelCol="rating_average",
    featuresCol="features"
)

model = rf.fit(train)

# -----------------------------
# STEP 6: Evaluate
# -----------------------------
predictions = model.transform(test)

evaluator_rmse = RegressionEvaluator(
    labelCol="rating_average",
    predictionCol="prediction",
    metricName="rmse"
)

evaluator_r2 = RegressionEvaluator(
    labelCol="rating_average",
    predictionCol="prediction",
    metricName="r2"
)

rmse = evaluator_rmse.evaluate(predictions)
r2 = evaluator_r2.evaluate(predictions)

print("RMSE:", rmse)
print("R2 Score:", r2)

# -----------------------------
# STEP 7: Write Metrics to HBase
# -----------------------------
metrics = [
    ("games1", "cf:rmse", str(rmse)),
    ("games1", "cf:r2", str(r2))
]

def write_to_hbase(partition):
    connection = happybase.Connection('master')
    connection.open()
    table = connection.table('games_metrics')
    for row in partition:
        key, col, val = row
        table.put(key, {col: val})
    connection.close()

rdd = spark.sparkContext.parallelize(metrics)
rdd.foreachPartition(write_to_hbase)

# -----------------------------
# STEP 8: Stop Spark
# -----------------------------
spark.stop()
EOF
