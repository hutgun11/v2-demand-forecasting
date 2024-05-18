import os
import re
import pickle
import numpy as np
from typing import List, Tuple, Optional, Union

#  Pyspark
from pyspark.sql import SparkSession, DataFrame, functions as F
from pyspark.sql.types import *

# Pyspark ml
from pyspark.ml import Pipeline
from pyspark.ml.feature import *

import tensorflow as tf

# tensorflow
from tensorflow.keras.layers import Dense, Dropout, Embedding, Input, Reshape, Concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanSquaredLogarithmicError

from sklearn.metrics import mean_squared_log_error


## GCS FUNCTION ##
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


def get_gcs_blob(pth):
    # Split gs:// path
    pth_split = pth.split('/')
    bucket_name = pth_split[2]
    file_pth = '/'.join(pth_split[3:])

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)

    blob = bucket.blob(file_pth)

    return blob


def check_file_exist(pth):
    if pth.startswith('gs://'):
        blob = get_gcs_blob(pth)
        is_file_exist = blob.exist

    else:
        is_file_exist = os.path.exists(pth)

    return is_file_exist


### MAIN FUNCTIONS ###
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


def label_encoding_spark(df_train: DataFrame, df_val: DataFrame,
                         df_test: DataFrame, cat_cols: List[str]):
    """Encoding each categorical column into integer, using StringIndexer

     Args:
         df_train: Traning Pyspark dataframe
         df_val: Validating Pyspark dataframe
         df_test: Testing Pyspark dataframe
         cat_cols: List of categorical columns

     Returns:
         Train, Val, and test Pyspark DataFrame, and label_class to map label encoder back to category
     """
    label_class = []
    stages = []

    print(f'Fitting pipeline encoding label')
    for col in cat_cols:
        # String Indexer
        indexer = StringIndexer(inputCol=col,
                                outputCol=f'{col}_idx')
        stages += [indexer]

    pipeline = Pipeline(stages=stages)
    model = pipeline.fit(df_train)

    print(f'Transform cat features into vector')

    # Transform
    df_train_tf = model.transform(df_train).cache()
    df_val_tf = model.transform(df_val).cache()
    df_test_tf = model.transform(df_test).cache()

    print('Save label_class')
    for col in cat_cols:
        label_class_each = df_train_tf[[f'{col}_idx']].schema.fields[0].metadata["ml_attr"]["vals"]
        label_class.append(label_class_each)

    return df_train_tf, df_val_tf, df_test_tf, label_class


def embedding_preprocess(df_transform: DataFrame, cat_cols: List[str],
                         cat_vector_cols: List[str], oth_cols: List[str],
                         label_col: str) -> Tuple[List[np.array], np.array]:
    """Transform features into numpy vector for embedding training

    Args:
        df_transform: Transformed Pyspark dataframe (have col_idx for category columns)
        cat_cols: List of categorical columns
        cat_vector_cols: List of categorical vector column (will treat as one type of category)
        oth_cols: List of other columns
        label_col: Name of label column

    Returns:
        List of numpy array of the train dataframe for each category,
        order from categorical column -> categorical vector column -> other column
    """

    X_embedding = []

    # the cols to be embedded: rescaling to range [0, # values)
    for c in cat_cols:
        input_array = np.array(df_transform[[f'{c}_idx']].collect()).ravel()
        X_embedding.append(input_array)

    # Put promotion as vector
    X_embedding.append(np.array(df_transform[[cat_vector_cols]].collect()))

    # the rest of the columns
    X_embedding.append(np.array(df_transform[[oth_cols]].collect()))

    # Get y
    y = np.array(df_transform[[label_col]].collect()).ravel()

    # Change all numpy array to float type
    X_embedding_float = []

    for array in X_embedding:
        X_embedding_float.append(array.astype(float))

    return X_embedding_float, y.astype(float)


def combined_network(X_embedding: List[np.array], cat_cols: List[str],
                     cat_vector_cols: List[str]) -> Model:
    """Combined tensorflow.keras network for embedding layer training

    Args:
        X_embedding: List of train inputs in form of numpy array
        (use function embedding_preprocess to generate)
        cat_cols: List of categorical columns
        cat_vector_cols: List of categorical vector columns (now only support one column (promotion))

    Returns:
        Tensorflow.Keras Model with embedding layers
    """

    inputs = []
    embeddings = []
    emb_dict = {}

    # create embedding layer for each categorical variables
    for i, cat in enumerate(cat_cols):
        cat_size = len(np.unique(X_embedding[i]))  # Get unique no of category
        cat_embsize = min(50, cat_size // 2 + 1)

        print(f'Category column {cat}: size={cat_size}, embedding size={cat_embsize}')

        emb_dict[cat] = Input(shape=(1,))
        embedding = Embedding(cat_size + 1,
                              cat_embsize,
                              input_length=1)(emb_dict[cat])
        embedding = Reshape(target_shape=(cat_embsize,))(embedding)

        # Append
        inputs.append(emb_dict[cat])
        embeddings.append(embedding)

    # Embedding vector cat vars (promotion)
    cat_vector_size = len(cat_vector_cols)
    cat_vector_embsize = min(25, cat_vector_size // 4 + 1)

    print(f'Category vector column: size={cat_vector_size}, embedding size = {cat_vector_embsize}')

    emb_vec_input = Input(shape=(cat_vector_size,))
    embedding = Embedding(cat_vector_size,
                          cat_vector_embsize,
                          input_length=cat_vector_size)(emb_vec_input)
    embedding = Reshape(target_shape=(cat_vector_size * cat_vector_embsize,))(embedding)
    inputs.append(emb_vec_input)
    embeddings.append(embedding)

    # concat continuous variables with embedded variables
    cont_input = Input(shape=(X_embedding[-1].shape[1],))
    # embedding = BatchNormalization()(cont_input)
    embedding = Dense(16, input_shape=X_embedding[-1].shape)(cont_input)
    inputs.append(cont_input)
    embeddings.append(embedding)

    # Concatenate
    x = Concatenate()(embeddings)

    # add user-defined fully-connected layers separated with dropout layers
    x = Dense(100, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(10, activation='relu')(x)
    output = Dense(1, activation='relu')(x)  # Use relu as ending
    model = Model(inputs, output)

    return model


def get_embedding_weights(model, cat_cols: List[str], cat_vector_cols: List[str],
                          label_class: List[str]):
    """Get the mapping dict between categorical columns and weight of embedding layer

    Args:
        model: tensorflow keras model
        cat_cols: List of categorical columns
        cat_vector_cols: List of categorical vector columns
        label_class: Mapping of encoder to real category

    Returns:
        Dict of categorical column and weight for each category
    """

    # Get weights from embedding layers
    weights_all = []

    for layer in model.layers:
        layer_config = layer.get_config()
        if layer_config['name'].startswith('embedding'):
            weights_all.append(layer.get_weights()[0])

    # Supose we have only one cat vector (promotion)
    cat_weights = weights_all[:-1]
    cat_vector_weights = weights_all[-1]

    cat_weights_all = {}
    cat_vector_weights_all = {}

    # cat vars
    for index, cat in enumerate(cat_cols):
        cat_weights_all[cat] = {}
        print(label_class[index])

        for i, label in enumerate(label_class[index]):
            print(i)
            print(label)
            cat_weights_all[cat][label] = cat_weights[index][i]

    # cat vector vars
    for index, cat_vector in enumerate(cat_vector_cols):
        cat_vector_weights_all[cat_vector] = cat_vector_weights[index]

    embedding_full = {0: cat_weights_all, 1: cat_vector_weights_all}

    return embedding_full


def RMSLE(y_true, y_pred):
    """
        The Root Mean Squared Log Error (RMSLE) metric
    """
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


spark = SparkSession.builder.master("local[*]") \
    .getOrCreate()

### PARAMS ###

# Columns name
KEY_COLS = ['_c0']
DATE_COL = 'Date'
DATE_COL_EOM = f'{DATE_COL}_EOM'
LABEL_COL = 'TotalQtySale'

# Date Parameters
START_DT = '2020-01-01'
START_DEV_DT = '2020-06-01'
START_OOT_DT = '2020-12-01'
END_DT = '2020-12-31'

# File path parameters
ROOT_DIR = f'gs://cj-demand-forecasting/'
# ROOT_DIR = '/Users/shirt/cbg/cj-demand-forecasting/'
data_nm = 'tildi_demandforecasting_type1_v1'
FEATURE_PTH = ROOT_DIR + f"features/"
MISC_PTH = ROOT_DIR + "misc/"
MODEL_PTH = ROOT_DIR + "model/"
EMBEDDING_DICT = MISC_PTH + f'cat_embedding_dict.pkl'
EMBEDDING_INFO = MISC_PTH + f'cat_embedding_info.pkl'

RANDOM_STATE = 2021

if ROOT_DIR.startswith('gs:'):
    print('Train on GCP')
    gcp_run = True
    from google.cloud import storage

else:
    gcp_run = False


def main():
    df = spark.read.format('delta').load(FEATURE_PTH + 'combined_ft')

    promo_list = [col for col in df.columns if bool(re.match(r"^[A-Z]{2}[0-9]{3}$", col))]
    promotion_select = percentile_cutoff(df, perc_cutoff=0.99, col_list=promo_list, col_name='promo')

    # Categorical variables
    CAT_COLS = ['product_cat_lv1', 'product_cat_lv2_cj', 'DayOfWeek', 'DistrictEN', 'ProvinceEN',
                'Quarter', 'MonthQuarter', 'DayOfMonth']

    # Categorical vector variables (embedding together as layer)
    CAT_VECTOR_COLS = promotion_select

    # Use original vars + top10 features
    NUM_COLS = ['Month', 'avgPriceDis', 'avgPrice', 'supPrice', 'WeekOfYear', 'TotalQtySale_avg_180d_mc',
                'TotalQtySale_avg_90d_mc', 'TotalQtySale_sum_180d_mc', 'TotalQtySale_avg_60d_mc',
                'TotalQtySale_sum_90d_mc_dow', 'TotalQtySale_sum_60d_mc_dow', 'TotalNetSale_avg_180d_mc_inc0',
                'TotalQtySale_avg_180d_mc_dow', 'TotalNetSale_avg_180d_mc_dow', 'TotalQtySale_sum_180d_mc_dow']

    # Preprocessing
    for col in CAT_COLS:
        df = df.withColumn(col, F.col(col).cast(StringType()))

    df = df.replace(float('nan'), None)
    df = df.fillna(0, subset=NUM_COLS + CAT_VECTOR_COLS)
    df = df.fillna('others', subset=CAT_COLS)

    # Split dev/test(oot)
    df_dev = df[(df[DATE_COL] >= START_DEV_DT) & (df[DATE_COL] < START_OOT_DT)].cache()
    df_test = df[(df[DATE_COL] >= START_OOT_DT) & (df[DATE_COL] <= END_DT)].cache()

    # Split train/val
    (df_train, df_val) = df_dev.randomSplit([0.9, 0.1], seed=RANDOM_STATE)

    df_train.cache()
    df_val.cache()

    df_train_tf, df_val_tf, df_test_tf, label_class = label_encoding_spark(df_train, df_val, df_test, CAT_COLS)

    # Save cat_cols, cat_vector_cols, and label_class
    save_list = [CAT_COLS, CAT_VECTOR_COLS, label_class]
    save_pickle(save_list, EMBEDDING_INFO)

    # Preprocess for tensorflow
    print('Preprocessing train')
    X_train_emb, y_train = embedding_preprocess(df_train_tf, CAT_COLS, CAT_VECTOR_COLS, NUM_COLS, LABEL_COL)
    print('Preprocessing val')
    X_val_emb, y_val = embedding_preprocess(df_val_tf, CAT_COLS, CAT_VECTOR_COLS, NUM_COLS, LABEL_COL)
    print('Preprocessing test')
    X_test_emb, y_test = embedding_preprocess(df_test_tf, CAT_COLS, CAT_VECTOR_COLS, NUM_COLS, LABEL_COL)

    spark.catalog.clearCache()  # Clear cache

    embedding_model_pth = ROOT_DIR + 'embedding_model/saved_model.pb'

    if not check_file_exist(embedding_model_pth):
        print('Embedding model does not exist, retrain embedding model')
        mirrored_strategy = tf.distribute.MirroredStrategy()

        with mirrored_strategy.scope():
            model = combined_network(X_train_emb, CAT_COLS, CAT_VECTOR_COLS)
            opt = Adam(0.0001)
            msle_loss = tf.keras.losses.MeanSquaredLogarithmicError(reduction=tf.keras.losses.Reduction.SUM)
            msle_metric = MeanSquaredLogarithmicError()

        model.compile(optimizer=opt,
                      loss=msle_loss,
                      metrics=[msle_metric])

        model.fit(X_train_emb, y_train, epochs=100,
                  validation_data=(X_val_emb, y_val))

        model.save(ROOT_DIR + 'embedding_model')

    else:
        print('Embedding Model exist, load model')
        model = load_model(MISC_PTH + 'embedding_model')

    # Save weight
    print('Save embedding weights')
    embedding_weights = get_embedding_weights(model, CAT_COLS, CAT_VECTOR_COLS, label_class)

    save_pickle(embedding_weights, EMBEDDING_DICT)

    print('Prediction on test set')

    y_test_pred = model.predict(X_test_emb)
    rmsle_score = RMSLE(y_test, y_test_pred)

    print(f'Test RMSLE: {rmsle_score}')


if __name__ == '__main__':
    main()
