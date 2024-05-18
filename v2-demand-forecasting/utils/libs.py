# Python
import re
import math
import time
import warnings
import os
import pickle
import argparse
import numpy as np
import pandas as pd
# from graphframes import *
from itertools import combinations, chain
from datetime import date, datetime, timedelta
from typing import List, Tuple, Optional, Union
from dateutil.relativedelta import relativedelta

#  Pyspark
import findspark
from pyspark import SparkContext
from pyspark.conf import SparkConf
from pyspark.sql import Row, SparkSession, DataFrame, functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window

# Pyspark ml
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import *
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier, GBTClassifier
from pyspark.ml.regression import DecisionTreeRegressor, RandomForestRegressor, GBTRegressor
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator, RegressionEvaluator

# Google Cloud
from google.cloud import storage

# from google.cloud import bigquery

# Tensorflow
# from tensorflow.keras.layers import Dense, Dropout, Embedding, Input, Reshape, Concatenate
# from tensorflow.keras.models import Model, Sequential
# from tensorflow.keras.metrics import RootMeanSquaredError
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras import backend as K

warnings.filterwarnings('ignore')


def extract_date(df_extract_date, columns='SalDate'):
    """Extract data to date features.

    :param spark: Spark session object.
    :return: Spark DataFrame.
    """
    df_extract_date = (df_extract_date
                       .withColumn('Yearday', F.dayofyear(F.col(columns)))
                       .withColumn('Month', F.month(F.col(columns)))
                       .withColumn('DayOfWeek', F.dayofweek(F.col(columns)))
                       .withColumn('DayOfMonth', F.dayofmonth(F.col(columns)))
                       .withColumn('Year', F.year(F.col(columns)))
                       .withColumn('Quarter', F.quarter(F.col(columns)))
                       .withColumn('WeekOfYear', F.weekofyear(F.col(columns)))
                       .withColumn('MonthQuarter', F.ceil(F.month(columns) / 3)))

    return df_extract_date


def stratified_split_train_test(df, frac, label, join_on, seed=42):
    fractions = df.select(label).distinct().withColumn("fraction", F.lit(frac)).rdd.collectAsMap()
    df_frac = df.stat.sampleBy(label, fractions, seed)
    df_remaining = df.join(df_frac, on=join_on, how="left_anti")
    return df_frac, df_remaining


def cast_double_types(df_cast, c=[]):
    for col in c:
        df_cast = df_cast.withColumn(
            col,
            F.col(col).cast("double")
        )
    return df_cast


def cast_int_types(df_cast, c=[]):
    for col in c:
        df_cast = df_cast.withColumn(
            col,
            F.col(col).cast("int")
        )
    return df_cast


def cast_date_types(df_cast, c=[]):
    for col in c:
        df_cast = df_cast.withColumn(
            col,
            F.col(col).cast(DateType())
        )
    return df_cast


def get_date_eom(dt):
    date_dt = datetime.strptime(str(dt), "%Y-%m-%d").date()
    next_month = date_dt.replace(day=28) + timedelta(days=4)
    date_EOM = next_month - timedelta(days=next_month.day)

    return date_EOM


def add_months(dt, n):
    # Return as last day of month + n (Support input both string and date)

    date_dt = datetime.strptime(str(dt), "%Y-%m-%d").date()
    added_dt = date_dt + relativedelta(months=n)
    added_date_EOM = get_date_eom(added_dt)

    return added_date_EOM


def add_days(dt, d):
    date_dt = datetime.strptime(str(dt), "%Y-%m-%d").date()
    added_dt = date_dt + relativedelta(days=d)

    return added_dt


def get_date_list(dt_start: Union[str, datetime.date], dt_end: Union[str, datetime.date]):
    date_list = []

    dt_start_eom = get_date_eom(dt_start)
    dt_end_eom = get_date_eom(dt_end)

    added_dt = dt_start_eom
    while added_dt <= dt_end_eom:
        date_list.append(added_dt)
        added_dt = add_months(added_dt, 1)

    return date_list


def save_pickle(file, pth):
    if pth.startswith('gs://'):
        blob = get_gcs_blob(pth)
        f = pickle.dumps(file)
        blob.upload_from_string(f)

    else:
        with open(pth, 'wb') as f:
            pickle.dump(file, f)


def load_pickle(pth):
    if pth.startswith('gs://'):
        blob = get_gcs_blob(pth)
        f = blob.download_as_string()
        file = pickle.loads(f)

    else:
        with open(pth, 'rb') as f:
            file = pickle.load(f)

    return file


def get_gcs_list_dir(pth, get_file_name=False):
    # Split gs:// path
    pth_split = pth.split('/')
    bucket_name = pth_split[2]

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    list_blobs = bucket.list_blobs()

    list_dir = []

    for blob in list_blobs:
        if get_file_name:
            list_dir.append(blob.name)
        else:
            list_dir.append('/'.join(blob.name.split('/')[:-1]))

    all_dir = list(set(list_dir))

    if len(pth_split) > 3:
        # Remove starting gs://bucket-name
        pth_replace = pth.replace(f'gs://{bucket_name}/', '')

        if pth_replace.endswith('/'):
            pth_replace = pth_replace[:-1]

        select_dir = [directory for directory in all_dir if directory.startswith(pth_replace)]

    else:
        select_dir = all_dir

    return select_dir


def get_gcs_blob(pth):
    # Split gs:// path
    pth_split = pth.split('/')
    bucket_name = pth_split[2]
    file_pth = '/'.join(pth_split[3:])

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)

    blob = bucket.blob(file_pth)

    return blob


def check_file_exist(pth, gcs_list_dir=None):
    if pth.startswith('gs://'):
        if gcs_list_dir is None:
            is_file_exist = any(get_gcs_list_dir(pth, get_file_name=True))

        else:
            pth_split = pth.split('/')
            file_pth = '/'.join(pth_split[3:])
            is_file_exist = any(directory.startswith(file_pth) for directory in gcs_list_dir)
    else:
        is_file_exist = os.path.exists(pth)

    return is_file_exist


def save_spark_csv(df, pth, filename):
    if not pth.endswith('/'):
        pth += '/'

    csv_location = pth + "temp.folder"
    file_location = pth + f'{filename}.csv'

    df.repartition(1).write.csv(path=csv_location, mode="overwrite", header="true")

    # Get single file location
    print(f'Save CSV at {file_location}')

    if pth.startswith('gs://'):

        list_dir = get_gcs_list_dir(csv_location, get_file_name=True)
        file_loc = [file_name for file_name in list_dir if file_name.endswith('.csv')][0]
        file_name = file_loc.split('/')[-1]
        file_pth = f'{csv_location}/{file_name}'
        os.system(f"gsutil cp {file_pth} {file_location}")
        os.system(f"gsutil rm -r {csv_location}")

    else:
        file_loc = [file_name for file_name in os.listdir(csv_location) if file_name.endswith('.csv')][0]
        file_pth = f'{csv_location}/{file_loc}'
        os.system(f"cp {file_pth} {file_location}")
        os.system(f"rm - r {csv_location}")


def create_transform_pipeline(df_train: DataFrame,
                              cat_cols: List[str],
                              num_cols: List[str],
                              onehot_ec: bool = False) -> PipelineModel:
    """Create transformation pipeline model

     Args:
         df_train: Traning dataframe
         cat_cols: List of categorical columns
         num_cols: List of numerical columns
         onehot_ec: Bool specifying transforming categorical columns into one-hot
         (Unnecessary for tree-based model)

     Returns:
         Pipeline model for feature transformation
     """

    stages = []

    print(f'Transforming cat features into vector')
    for col in cat_cols:

        # String Indexer & One-hot encoder
        indexer = StringIndexer(inputCol=col,
                                outputCol=f'{col}_idx')
        if onehot_ec:
            print('Perform one-hot encoding')
            vector = OneHotEncoder(inputCol=f'{col}_idx',
                                   outputCol=f'{col}_vec')
            stages += [indexer, vector]
        else:
            print('Do not perform one-hot encoding')
            stages += [indexer]

    features = num_cols + [f'{col}_vec' if onehot_ec else f'{col}_idx' \
                           for col in cat_cols]
    assem = VectorAssembler(inputCols=features,
                            outputCol="features_col")

    stages += [assem]
    pipeline = Pipeline(stages=stages)
    model = pipeline.fit(df_train)

    return model


def fit_model_spark(df_train: DataFrame,
                    model_class=RandomForestRegressor,
                    feature_col: str = 'features_col',
                    label_col: str = 'TotalQtySale', **kwargs):
    """ Fitting pyspark model
     Args:
         df_train: Traning dataframe
         model_class: Pyspark model type
         feature_col: Feature column name
         label_col: Label column name
         **kwargs: Model parameters

     Returns:
         trained model class
     """

    m = model_class(labelCol=label_col,
                    featuresCol=feature_col,
                    predictionCol="prediction", **kwargs)

    model = m.fit(df_train)

    return model


def percentile_cutoff(df: DataFrame, perc_cutoff: float, col_list: List[str], col_name: str) -> List[str]:
    """Return list of categories that have counts higher than percentile cutoff

    Args:
        df: Input Pyspark DataFrame with key column, date column, and chosen col_list
        perc_cutoff: Percentile for cutoff
        col_list: Categorical columns to be count on. If single columns, it will group based on that column.
        If multiple columns, it will treat multiple columns as one-hot encoding columns.
        col_name: Name after grouping (use only when len(col_list) > 1)

    Returns:
        List of categories that have counts higher than percentile cutoff
    """

    if len(col_list) > 1:
        df_grp_count = df.groupby().agg(*[F.sum(col).alias(col) for col in col_list])
        df_count_pd = df_grp_count.toPandas().T.reset_index()
        df_count_pd.columns = [col_name, 'count']

    else:
        df_grp_count = df.groupby(col_list).agg(F.count(df.columns[0]).alias('count'))
        df_count_pd = df_grp_count.toPandas().reset_index()

    df_count_pd = df_count_pd.sort_values('count', ascending=False)
    df_count_pd['cumsum'] = df_count_pd['count'].cumsum()
    df_count_pd['perc_cumsum'] = df_count_pd['cumsum'] / df_count_pd.sum()['count']

    # Cut at perc_cutoff
    col_select = df_count_pd[df_count_pd['perc_cumsum'] <= perc_cutoff][col_name].to_list()

    return col_select


def rmsle_spark(df: DataFrame, label_col: str = 'TotalQtySale', prediction_col: str = 'prediction') -> float:
    sle = ((F.log(F.col(prediction_col) + F.lit(1)) - F.log(F.col(label_col) + F.lit(1))) ** 2).alias('sle')
    df = df.select(sle)
    return np.sqrt(df.groupby().agg(F.avg('sle').alias('msle')).collect()[0][0])


def mape_spark(df: DataFrame, label_col: str = 'TotalQtySale', prediction_col: str = 'prediction') -> float:
    ape = ((F.abs(F.col(prediction_col) - F.col(label_col)) * F.lit(100)) / F.col(label_col)).alias('ape')
    df = df.select(ape)
    return df.groupby().agg(F.avg('ape').alias('mape')).collect()[0][0]


def oos_num_perc(df: DataFrame, label_col: str = 'TotalQtySale', prediction_col: str = 'prediction') -> float:
    df = df.withColumn('oos_num', F.when(F.col(prediction_col) > F.col(label_col), 0
                                         ).otherwise(F.col(label_col) - F.col(prediction_col)))
    return df.groupby().agg(F.sum('oos_num') / F.sum(label_col)).alias('oos_num_perc').collect()[0][0]


def demand_forecasting_evaluator(df_predict: DataFrame, label_col: str = 'TotalQtySale',
                                 prediction_col: str = "prediction"):
    result_dict = {}

    # Performance Evaluation: Overall
    evaluator = RegressionEvaluator(labelCol=label_col, predictionCol=prediction_col)

    for metric_name in ['rmse', 'mse', 'r2', 'mae']:
        metric_result = evaluator.setMetricName(metric_name).evaluate(df_predict)
        print(f'{metric_name}: {metric_result}')
        result_dict[metric_name] = metric_result

    # MAPE
    mape_score = mape_spark(df_predict, label_col, prediction_col)
    print(f'mape: {mape_score}')
    rmsle_score = rmsle_spark(df_predict, label_col, prediction_col)
    print(f'rmsle: {rmsle_score}')

    result_dict['mape'] = mape_score
    result_dict['rmsle'] = rmsle_score

    result_dict['accuracy'] = (100 - ((np.exp(rmsle_score) - 1) * 100))
    print(f'accuracy: {(100 - ((np.exp(rmsle_score) - 1) * 100))}')

    # Performance Evaluation: OUT OF STOCK (OOS)
    all_test_number = df_predict.count()
    sell_test_number = df_predict.filter(F.col(label_col) > 0).count()
    oos_number = df_predict.filter(df_predict[prediction_col] < df_predict[label_col]).count()
    oos_sales_perc = oos_num_perc(df_predict, label_col, prediction_col)

    result_dict['percent_oos'] = oos_number * 100 / all_test_number
    print(f"OOS Percentage: {result_dict['percent_oos']}")
    result_dict['percent_oos_sale'] = oos_number * 100 / sell_test_number
    print(f"OOS Percentage (only sale): {result_dict['percent_oos_sale']}")
    result_dict['percent_sales_oos'] = oos_sales_perc
    print(f"OOS Sales percentage: {result_dict['percent_sales_oos']}")

    return result_dict
