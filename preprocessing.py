from pyspark.sql.types import StringType, IntegerType, NumericType
from pyspark.ml.feature import StringIndexer, StandardScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
import streamlit as st


def preprocess(data):
    data = handle_null_missing_values(data)
    data = encode(data)
    target_var = data.columns[-1]
    input_vars = data.drop(target_var)
    input_vars = remove_outliers(input_vars)
    input_vars = scale_data(input_vars)
    st.write(input_vars)
    data = input_vars.withColumn("Target", data[target_var])
    return data


def handle_null_missing_values(data):
    for col in data.columns:
        if data.filter(data[col].isNull()).count() > 0:
            if isinstance(data.schema[col].dataType, StringType):
                data = data.fillna(method='bfill', subset=[col]).fillna(method='ffill', subset=[col])
            else:
                mean_val = data.select(col).agg({col: 'mean'}).collect()[0][0]
                data = data.fillna(mean_val, subset=[col])
    return data

    # for col in data.columns:
    #     if data[col].isna().sum() > 0:
    #         if type(data[col].loc[0]) == 'str':
    #             data[col].fillna(method='bfill', inplace=True)
    #             data[col].fillna(method='ffill', inplace=True)
    #         else:
    #             data[col].fillna(np.round(np.mean(data[col]), decimals=0), inplace=True)
    # return data


def encode(data):
    for col in data.columns:
        if isinstance(data.schema[col].dataType, StringType):
            indexer = StringIndexer(inputCol=col, outputCol=col + "_indexed")
            data = indexer.fit(data).transform(data).drop(col)

    return data
    # encoder = LabelEncoder()
    # for col in data.columns:
    #     if type(data[col].loc[0]) == 'str':
    #         if not data[col].loc[0].isnumeric():
    #             data[col] = encoder.fit_transform(data[col])
    #         else:
    #             data[col] = int(data[col])
    # return data


def remove_outliers(data):
    for col in data.columns:
        if isinstance(data.schema[col].dataType, (IntegerType, NumericType)):
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
    # for col in data.columns:
    #     Q1 = data[col].quantile(.25)
    #     Q3 = data[col].quantile(.75)
    #     IQR = (Q3 - Q1) * 1.5
    #     lower = Q1 - IQR
    #     upper = Q3 + IQR
    #     if np.sum(((data[col] < lower) | (data[col] > upper))) > 0:
    #         # data[col] = np.clip(lower=data[col].quantile(.25), upper=data[col].quantile(.75))
    #         mean_without_outliers = np.mean(data[col][(data[col] >= lower) & (data[col] <= upper)])
    #         indices = data.index[(data[col] < lower) | (data[col] > upper)]
    #         data.loc[indices, col] = mean_without_outliers
    #
    # return data


def scale_data(data):
    features = data.columns
    assembler = VectorAssembler(inputCols=features, outputCol="features")
    scaler = StandardScaler(inputCol="features", outputCol='scaled_features', withMean=True)
    pipeline = Pipeline(stages=[assembler, scaler])
    model = pipeline.fit(data)
    data = model.transform(data).drop(*features).withColumnRenamed("scaled_features", "features")
    st.write(data.show())
    return data
    # scaler = StandardScaler()
    # return pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
