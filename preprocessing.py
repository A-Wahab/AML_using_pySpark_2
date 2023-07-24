from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler


def preprocess(spark, data):
    target_var = data.columns[-1]
    data = handle_null_missing_values(data)
    data = encode(data, spark)
    input_var = data.drop(target_var, axis=1)
    input_var = remove_outliers(input_var, spark)
    input_var = scale_data(input_var, spark)
    data = pd.concat([input_var, data[target_var]], axis=1)
    return data


def handle_null_missing_values(data):
    # For PySpark, handling null/missing values can be done using DataFrame functions
    # The 'na' functions are used for this purpose.
    for col in data.columns:
        if data.filter(data[col].isNull() | data[col].isNaN()).count() > 0:
            if isinstance(data.schema[col].dataType, StringType):
                data = data.fillna(method='bfill', subset=[col]).fillna(method='ffill', subset=[col])
            else:
                mean_val = data.select(col).agg({col: 'mean'}).collect()[0][0]
                data = data.fillna(mean_val, subset=[col])
    return data


def encode(data, spark):
    # Use StringIndexer for encoding string columns
    # Integer columns will be converted to double for later processing
    for col in data.columns:
        if isinstance(data.schema[col].dataType, StringType):
            indexer = StringIndexer(inputCol=col, outputCol=col + "_index")
            data = indexer.fit(data).transform(data).drop(col)

    return data


def remove_outliers(data):
    for col in data.columns:
        quantiles = data.approxQuantile(col, [0.25, 0.75], 0.05)
        Q1 = quantiles[0]
        Q3 = quantiles[1]
        IQR = (Q3 - Q1) * 1.5
        lower = Q1 - IQR
        upper = Q3 + IQR

        # Use a filter to remove outliers and then fill them with the mean value
        data = data.filter((data[col] >= lower) & (data[col] <= upper))
        mean_without_outliers = data.agg({col: 'mean'}).collect()[0][0]
        data = data.fillna(mean_without_outliers, subset=[col])

    return data


def scale_data(data, spark):
    # Use StandardScaler to scale the numeric columns
    features = data.columns
    features.remove("target")  # Replace "target" with the actual column name of the target variable
    assembler = VectorAssembler(inputCols=features, outputCol="features")
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features")

    pipeline = Pipeline(stages=[assembler, scaler])
    model = pipeline.fit(data)
    data = model.transform(data).drop(*features).withColumnRenamed("scaled_features", "features")

    return data


def split(data):
    # PySpark dataframes inherently handle splitting into training and testing sets
    # The randomSplit method can be used for this purpose.
    return data.randomSplit([0.8, 0.2], seed=42)


# Sample usage
if __name__ == "__main__":
    spark = SparkSession.builder.appName("Preprocessing").getOrCreate()
    data = spark.read.csv("path_to_your_data.csv", header=True, inferSchema=True)
    preprocessed_data = preprocess(spark, data)
    train_data, test_data = split(preprocessed_data)
