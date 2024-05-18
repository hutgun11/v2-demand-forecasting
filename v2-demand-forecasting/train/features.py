# import sys
#
# sys.path.append('../')

from train.config import *
from train import spark


def get_basic_feature(df: DataFrame, promo_perc_cutoff: float = 0.99) -> DataFrame:
    """Return dataframe with basic features:
    location of branch, time-based features, one-hot promotion columns, category features,
    store cluster, and discount feature.

    Args:
        df: Input Pyspark DataFrame with key column, date column, and BranchCode
        promo_perc_cutoff: Percentage cutoff to select promotion that has count higher than threshold

    Returns:
        A new dataframe of key column, date column + branch location features + time-based features
        + one hot promotions + category features + store cluster feature + discount_feature
    """

    df = get_locations_feature(df).cache()  # Location
    df = extract_date(df, columns=DATE_COL).cache()  # Date features
    df = get_one_hot_promotion(df, perc_cutoff=promo_perc_cutoff).cache()  # Get one-hot promotion
    df = get_category_feature(df)  # Get categorical
    df = get_store_cluster_feature(df)  # Get store cluster
    df = get_discount_feature(df)  # Get discount feature

    return df


def get_locations_feature(df: DataFrame) -> DataFrame:
    """Return dataframe with province, district, and zipcode, joining with dataframe on BranchCode
    Requires LOCATION_FILE (specify in __init__.py)

    Args:
        df: Input Pyspark DataFrame with key column, date column, ZipCode and ProvinceEN columns.

    Returns:
        df: A new dataframe of key column, date column + location columns
    """

    # locations
    locations: DataFrame = spark.read.format("csv").load(LOCATION_FILE, header=True, inferSchema=True)
    locations = locations.select(['BranchCode', 'ZipCode', 'ProvinceEN', 'DistrictEN'])
    # Clean empty string at the end
    locations = locations.withColumn('ProvinceEN', F.regexp_replace('ProvinceEN', '[ \f\t\v]$', '')).cache()
    df = df.join(locations, on='BranchCode', how='left')

    return df


def get_lat_long_feature(df: DataFrame) -> DataFrame:
    """Return dataframe with latitude and longitude columns based on ZipCode.
    If ZipCode doesn't match, it will use the average latitude/longitude of the province.
    Requires LATLONG_FILE (specify in __init__.py)

    Args:
        df: Input Pyspark DataFrame with key column, date column, ZipCode and ProvinceEN columns.

    Returns:
        df: A new dataframe of key column, date column + latitude + longitude columns
    """

    location_lat_long: DataFrame = spark.read.format("csv").load(LATLONG_FILE, header=True, inferSchema=True)

    df_loc_zip = location_lat_long.groupby('ZipCode').agg(F.avg('latitude').alias('latitude'),
                                                          F.avg('longitude').alias('longitude'))

    df_loc_province = location_lat_long.groupby('ProvinceEN').agg(F.avg('latitude').alias('latitude_prov'),
                                                                  F.avg('longitude').alias('longitude_prov'))

    df = (df.join(df_loc_zip, on='ZipCode', how='left')
          .join(df_loc_province, on='ProvinceEN', how='left'))

    df = (df.withColumn('latitude', F.coalesce(F.col('latitude'), F.col('latitude_prov')))
          .withColumn('longitude', F.coalesce(F.col('longitude'), F.col('longitude_prov'))))

    return df.select(*KEY_COLS, DATE_COL, DATE_COL_EOM, *['latitude', 'longitude'])


def get_one_hot_promotion(df: DataFrame, perc_cutoff: float = 0.99) -> DataFrame:
    """Return dataframe with one-hot encoding of promotion column

    Args:
        df: Input Pyspark DataFrame with key column, date column and 'types' (promotion)
        perc_cutoff: Percentile for promotion cutoff.
        Promotion with cumulative counts lower than the percentile are removed.

    Returns:
        A new dataframe adding promotion flag for each promotion that has counts higher than percentile.
    """

    df = df.withColumnRenamed('types', 'promotion')
    column_list = [c for c in df.columns if c != 'promotion']

    promo_collect = df[['promotion']].distinct().collect()
    promo_list = [row[0].split('|') for row in promo_collect]

    all_promotion = []

    for promo in promo_list:
        all_promotion += promo
    all_promotion_only = [i for i in all_promotion if i != 'NORMAL']
    unique_promo = list(set(all_promotion_only))

    for promo in unique_promo:
        df = df.withColumn(promo, F.when(F.col('promotion').contains(promo), 1).otherwise(0))
    df = df.drop('promotion')

    if perc_cutoff < 1:
        # Get promotion count (for percentile cut)
        promotion_select = percentile_cutoff(df, perc_cutoff, unique_promo, 'promo')
    else:
        promotion_select = unique_promo

    df = df.select(*column_list, *promotion_select)

    return df


def get_category_feature(df: DataFrame) -> DataFrame:
    """Return dataframe with categorical level 1 (basic) and 2 (deeper category), joining with MaterialCode.
    The function requires cj_category_v{version}.csv and cat_mapping_lv1_cj.csv in misc folder

    Args:
        df: Input Pyspark DataFrame with key column, date column, and MaterialCode

    Returns:
        A new dataframe of key column, date column + new columns product_cat_lv1 and product_cat_lv2_cj
    """

    category_file = spark.read.format("csv").load(CATEGORY_FILE, header=True, inferSchema=True)
    cat_mapping_lv1 = spark.read.format("csv").load(CAT_MAPPING_FILE, header=True, inferSchema=True)

    category_file = category_file[['Class Name', 'Product ID']]
    category_file = category_file.toDF(*['product_cat_lv2_cj', 'MaterialCode'])
    cat_mapping_lv1 = cat_mapping_lv1[['product_cat_lv1', 'product_cat_lv2_cj']]

    df = df.join(category_file, on=['MaterialCode'], how='left'
                 ).join(cat_mapping_lv1, on='product_cat_lv2_cj', how='left')
    df = df.fillna('others', subset=['product_cat_lv1', f'product_cat_lv2_cj'])

    return df


def get_consecutive_feature(df: DataFrame, group_cols: List[str], group_name: str,
                            flag_col: str, agg_day_list: List[int], lag_day: int = 5) -> DataFrame:
    """Return dataframe with column zero_consc_{group_name} to count current consecutive zero based on
    selected group, and max_zero_consc_{agg_day}d_{group_name} which is the maximum consecutive zero
    in that group in {agg_day}days.

    Args:
        df: Input Pyspark DataFrame with key column, date column and group_col + flag_col specified
        group_cols: Columns to be grouped on
        group_name: Suffix of the column
        flag_col: Column with flag 0/1 for counting zeros
        agg_day_list: List of days to calculate maximum consecutive zero
        lag_day: Time lag to calculate feature, eg. 5 days means to count zero only data before latest 5 days ago

    Returns:
        A new dataframe of key column, date column + new column zero_consc_{group_name}
        and max_zero_consc_{agg_day}d_{group_name} for each agg_day in agg_day list
    """

    DAYS_TO_SECONDS = 86400
    MAX_DATE = 90  # Count to only last 90 days
    days = lambda d: d * DAYS_TO_SECONDS

    df_grp = (df.groupby(*group_cols, DATE_COL)
              .agg(F.max(flag_col).alias(flag_col)))

    w = Window.partitionBy(*group_cols) \
        .orderBy(F.col(DATE_COL).cast("timestamp").cast("long")) \
        .rangeBetween(-days(MAX_DATE + lag_day), -days(1 + lag_day))

    df_grp = (df_grp.withColumn('last_sell_date', F.max(F.when(F.col('label') == 1, F.col(DATE_COL))).over(w))
              .withColumn(f'zero_consc_{group_name}',
                          F.datediff(F.col(DATE_COL), F.col('last_sell_date')) - lag_day)
              .fillna(MAX_DATE, subset=[f'zero_consc_{group_name}']))

    df_consc = df.join(df_grp, on=group_cols + [DATE_COL], how='left')
    df_consc = df_consc.select(*KEY_COLS, *group_cols, DATE_COL, DATE_COL_EOM, f'zero_consc_{group_name}')

    # Generate maximum consecutive days
    max_cols = []
    for agg_day in agg_day_list:
        w_agg = Window.partitionBy(group_cols) \
            .orderBy(F.col(DATE_COL).cast("timestamp").cast("long")) \
            .rangeBetween(-days(agg_day + lag_day + 1), -days(1 + lag_day))
        max_func = calculate_max(agg_day, f'zero_consc_{group_name}', group_name, w_agg)
        max_cols += [max_func]

    df_consc_full = df_consc.select(*df_consc.columns, *max_cols).cache()

    return df_consc_full.select(*KEY_COLS, DATE_COL, DATE_COL_EOM,
                                *[col for col in df_consc_full.columns if 'zero_consc' in col])


def get_ts_agg_feature(df: DataFrame, group_cols: List[str], agg_cols: List[str], group_name: List[str],
                       agg_day_list, sum_ratio_list: List[Tuple[int]] = None, lag_day: int = 5) -> DataFrame:
    """Return dataframe with time-series aggregation function: sum, avg, std and cv (coefficient of variation)
     on columns based on selected group for each number of days.

    Args:
        df: Input Pyspark Dataframe with key column, date column and group_col + agg_cols specified
        group_cols: Columns to be grouped on
        agg_cols: List of columns to be aggregated
        group_name: Suffix of the column
        agg_day_list: List of days to calculate aggregation functions
        sum_ratio_list: List of day ratio (num,denom) to calculate sum ratio between select days
        lag_day: Time lag to calculate feature, eg. 5 days means to aggregate only data before latest 5 days ago

    Returns:
        A new dataframe of key column, date column + new columns {col}_{agg_func}_{agg_day}d_{group_name} and
        {col}_{agg_func}_{agg_day}d_{group_name}_inc0 (agg_func are sum, avg, stddev, and cv)
    """
    DAYS_TO_SECONDS = 86400
    days = lambda d: d * DAYS_TO_SECONDS

    df = df.select(*KEY_COLS, DATE_COL, DATE_COL_EOM, *agg_cols, *group_cols, 'label')
    df_grp = df.groupby(*group_cols, DATE_COL
                        ).agg(*[F.sum(col).alias(col) for col in agg_cols],
                              F.sum('label').alias('label'),
                              F.count('label').alias('count')).cache()

    # Create empty list
    label_sum_cols, sum_cols_all, avg_cols_all, diff_sq_cols_all, std_cols_all, cv_cols_all = \
        tuple([[] for i in range(6)])

    for agg_day in agg_day_list:
        w = Window.partitionBy(group_cols) \
            .orderBy(F.col(DATE_COL).cast("timestamp").cast("long")) \
            .rangeBetween(-days(agg_day + lag_day + 1), -days(1 + lag_day))
        label_sum_func = (F.sum('label').over(w).alias(f'label_sum_{agg_day}d'))  # Total sum label
        count_sum_func = (F.sum('count').over(w).alias(f'count_sum_{agg_day}d'))  # Total column
        label_sum_cols += [label_sum_func, count_sum_func]

        for col in agg_cols:
            sum_cols_all, avg_cols_all, diff_sq_cols_all, std_cols_all, cv_cols_all = \
                _get_ts_agg_cols(col, agg_day, group_name, w,
                                 sum_cols_all, avg_cols_all, diff_sq_cols_all, std_cols_all, cv_cols_all)

    df_grp = df_grp.select(DATE_COL, *group_cols, *agg_cols,
                           *sum_cols_all, 'label', 'count', *label_sum_cols).cache()
    df_grp = df_grp.select(*df_grp.columns, *avg_cols_all).cache()
    df_grp = df_grp.select(*df_grp.columns, *diff_sq_cols_all).cache()
    df_grp = df_grp.select(*df_grp.columns, *std_cols_all).cache()
    df_grp = df_grp.select(*df_grp.columns, *cv_cols_all).cache()

    # Add ratio feature
    if sum_ratio_list is not None:
        df_grp = get_ratio_cols(df_grp, agg_cols, group_name, sum_ratio_list)

    df_final = df.join(df_grp, on=group_cols + [DATE_COL], how='left')

    agg_col_str = ['_sum', '_avg', '_stddev', '_cv']
    rmv_col_str = ['label', 'diff_sq', 'count']

    all_agg_cols = [col for col in df_final.columns if
                    any([col_str in col for col_str in agg_col_str])
                    & ~any([rmv_str in col for rmv_str in rmv_col_str])]

    return df_final.select(*KEY_COLS, DATE_COL, DATE_COL_EOM, *all_agg_cols)


def get_ratio_cols(df, agg_cols, group_name, sum_ratio_list):
    for col in agg_cols:
        for (num_day, denom_day) in sum_ratio_list:
            df = df.withColumn(f'{col}_sum_ratio_{num_day}d{denom_day}d_{group_name}',
                               F.col(f'{col}_sum_{num_day}d_{group_name}') /
                               F.col(f'{col}_sum_{denom_day}d_{group_name}'))
    return df


def _get_ts_agg_cols(col, agg_day, group_name, w,
                     sum_cols_all, avg_cols_all, diff_sq_cols_all, std_cols_all, cv_cols_all):
    # Generate all time-series function and append to existing lists

    sum_func = calculate_sum(agg_day, col, group_name, w)
    sum_cols = [sum_func]

    avg_func, avg_sell_func = calculate_avg(agg_day, col, group_name)
    avg_cols = [avg_func, avg_sell_func]

    diff_sq_func, diff_sq_sell_func, std_func, std_sell_func = calculate_stddev(agg_day, col, group_name, w)
    diff_sq_cols = [diff_sq_func, diff_sq_sell_func]
    std_cols = [std_func, std_sell_func]

    cv_func, cv_sell_func = calculate_cv(agg_day, col, group_name)
    cv_cols = [cv_func, cv_sell_func]

    sum_cols_all += sum_cols
    avg_cols_all += avg_cols
    diff_sq_cols_all += diff_sq_cols
    std_cols_all += std_cols
    cv_cols_all += cv_cols

    return sum_cols_all, avg_cols_all, diff_sq_cols_all, std_cols_all, cv_cols_all


def calculate_sum(agg_day, col, group_name, w):
    # Calculate sum, partition by w

    sum_col_name = f'{col}_sum_{agg_day}d_{group_name}'
    sum_func = F.sum(col).over(w).alias(sum_col_name)
    return sum_func


def calculate_max(agg_day, col, group_name, w):
    # Calculate max, partition by w

    if f'_{group_name}' in col:
        col_name = col.replace(f'_{group_name}', '')
    else:
        col_name = col
    max_col_name = f'{col_name}_max_{agg_day}d_{group_name}'
    max_func = (F.max(col).over(w)).alias(max_col_name)

    return max_func


def calculate_cv(agg_day, col, group_name):
    # Manually calculate cv for excluding 0 and including 0, required stddev and avg columns

    avg_col_name = f'{col}_avg_{agg_day}d_{group_name}'
    std_col_name = f'{col}_stddev_{agg_day}d_{group_name}'

    cv_func = (F.col(f'{std_col_name}_inc0') / F.col(f'{avg_col_name}_inc0')
               ).alias(f'{col}_cv_{agg_day}d_{group_name}_inc0')
    cv_sell_func = (F.col(std_col_name) / F.col(avg_col_name)
                    ).alias(f'{col}_cv_{agg_day}d_{group_name}')
    return cv_func, cv_sell_func


def calculate_stddev(agg_day, col, group_name, w):
    # Manually calculate stddev for excluding 0 and including 0, required average label_sum and count_sum

    avg_col_name = f'{col}_avg_{agg_day}d_{group_name}'
    std_col_name = f'{col}_stddev_{agg_day}d_{group_name}'

    diff_sq_func = ((F.col(col) - F.col(f'{avg_col_name}_inc0')) ** 2).alias(f'{col}_{agg_day}_diff_sq_inc0')
    std_func = F.sqrt((F.sum(f'{col}_{agg_day}_diff_sq_inc0').over(w)) / F.col(f'count_sum_{agg_day}d')
                      ).alias(f'{std_col_name}_inc0')  # ALL
    diff_sq_sell_func = (F.when(F.col('label') == 1, ((F.col(col) - F.col(avg_col_name)) ** 2
                                                      )).otherwise(0)).alias(f'{col}_{agg_day}_diff_sq')
    std_sell_func = F.sqrt((F.sum(f'{col}_{agg_day}_diff_sq').over(w)) / F.col(f'label_sum_{agg_day}d')
                           ).alias(std_col_name)  # Only sell
    return diff_sq_func, diff_sq_sell_func, std_func, std_sell_func


def calculate_avg(agg_day, col, group_name):
    # Manually calculate average for excluding 0 and including 0, required label_sum and count_sum

    sum_col_name = f'{col}_sum_{agg_day}d_{group_name}'
    avg_col_name = f'{col}_avg_{agg_day}d_{group_name}'
    avg_func = (F.col(sum_col_name) / F.col(f'label_sum_{agg_day}d')).alias(f'{avg_col_name}_inc0')  # ALL
    avg_sell_func = (F.col(sum_col_name) / F.col(f'count_sum_{agg_day}d')).alias(avg_col_name)  # Only sell

    return avg_func, avg_sell_func


def get_embedding_feature(df: DataFrame, cat_cols: List[str], embedding_promo: bool = True) -> DataFrame:
    """Return dataframe with embedding columns
     The function requires EMBEDDING_FILE specify in __init__.py

     Args:
         df: Input Pyspark DataFrame with key column, date column, categorical columns,
         and one-hot promotion columns if embedding_promo is True
         cat_cols: List of categorical columns (must contains in the key of cat_embedding_dict.pickle)
         embedding_promo: Bool of generating promotion embedding columns

     Returns:
         A new dataframe of key column, date column + new embedding categorical columns
     """

    # Load embedding dict
    embedding_dict = load_pickle(EMBEDDING_FILE)
    cat_dict = embedding_dict[0]
    cat_vector_dict = embedding_dict[1]

    # Normal categorical column
    for col in cat_cols:

        vector_shape = len(list(cat_dict[col].values())[0])
        if col in ['Quarter', 'MonthQuarter', 'DayOfMonth', 'DayOfWeek']:
            col_type = IntegerType()
            key_type = int
        else:
            col_type = StringType()
            key_type = str

        # Create dataframe based on value
        schema = StructType([StructField(f'{col}', col_type)] +
                            [StructField(f'{col}_embed{i}', DoubleType()) for i in range(vector_shape)])

        df_temp = spark.createDataFrame([
            [key_type(key), *[float(sub_value) for sub_value in value]]
            for key, value in cat_dict[col].items()], schema)

        df = df.join(df_temp, on=col, how='left')

    # Categorical vector columns (promotion)
    if embedding_promo:
        df = extract_promo_embedding(cat_vector_dict, df)
    embed_col_all = [col for col in df.columns if '_embed' in col]
    df = df.fillna(0, subset=embed_col_all)

    return df.select(*KEY_COLS, DATE_COL, DATE_COL_EOM, *embed_col_all)


def extract_promo_embedding(cat_vector_dict, df):
    # Extract promotion embedding

    promo_list = list(cat_vector_dict.keys())
    df_collect = df[KEY_COLS + promo_list].collect()
    collect_list = [[x[i] for i in range(len(df_collect[0]))] for x in df_collect]
    key_array = [x[:len(KEY_COLS)] for x in collect_list]
    promo_array = np.array([x[len(KEY_COLS):] for x in collect_list])
    cat_vector_embed = np.array(list(cat_vector_dict.values()))
    # Transform by multiple matrix
    cat_vector_embed_tf = np.matmul(promo_array, cat_vector_embed)
    # Create dataframe for joining
    cat_vector_count = cat_vector_embed_tf.shape[0]
    cat_vector_shape = cat_vector_embed_tf.shape[1]
    schema = StructType([StructField(key, LongType()) for key in KEY_COLS] +
                        [StructField(f'promo_embed{i}', DoubleType()) for i in range(cat_vector_shape)])
    df_temp = spark.createDataFrame([key_array[i] + [float(cat_vector_embed_tf[i][j])
                                                     for j in range(cat_vector_shape)]
                                     for i in range(cat_vector_count)], schema)

    df = df.join(df_temp, on=KEY_COLS, how='left')
    return df


def save_feature_delta(df: DataFrame,
                       ft_func,
                       ft_name: str,
                       ft_start_dt: str = START_DT,
                       ft_end_dt: str = END_DT,
                       is_overwrite: bool = True,
                       save_pth: Optional[str] = None,
                       **kwargs):

    """Save feature in delta format to FEATURE_PTH/ft_name
    For setup, refers to https://docs.delta.io/latest/quick-start.html
    *If start date = end date, it will check exists by counting dataframe
    to check if that dataframe on the day exist (for inference).

     Args:
         df: Input Pyspark DataFrame
         ft_func: Function to generate feature
         ft_name: Feature name to write on disk
         ft_start_dt: Start date of feature
         ft_end_dt: End date of feature
         is_overwrite: If exists, will overwrite the feature
         save_pth: Save path, if not specify will save in FEATURE_PTH + ft_name
         **kwargs: Parameters for feature generation

     Returns:
         None
     """

    # Check inference
    if ft_start_dt == ft_end_dt:
        print(f'Save feature for inference: Date {ft_start_dt}')
        check_date = ft_start_dt
    else:
        check_date = None

    # Check save path
    if save_pth is None:
        save_pth = FEATURE_PTH + ft_name

    is_exist, last_eom_date = check_delta_exist(save_pth, check_date)

    str_log = ''
    if is_exist:
        str_log += f'Features {ft_name} exists:'
        if is_overwrite:
            str_log += f' Overwrite'
            print(str_log)
        else:
            str_log += f' Skipping the features'
            print(str_log)
            return None

    ft_start_dt_eom = get_date_eom(ft_start_dt)
    ft_end_dt_eom = get_date_eom(ft_end_dt)

    df_feature = ft_func(df, **kwargs).cache()
    df_feature = df_feature.filter((F.col(DATE_COL_EOM) >= ft_start_dt_eom)
                                   & (F.col(DATE_COL_EOM) <= ft_end_dt_eom)
                                   & (F.col(DATE_COL) >= ft_start_dt)
                                   & (F.col(DATE_COL) <= ft_end_dt))

    print(f'Saving to {save_pth}, date between {ft_start_dt} and {ft_end_dt}')
    df_feature.write.format("delta") \
        .mode("overwrite") \
        .partitionBy(DATE_COL_EOM) \
        .option("header", "true") \
        .save(save_pth)

    return None


def combine_features(df: DataFrame,
                     ft_name_list: List[str],
                     ft_save_pth: str = FEATURE_PTH,
                     save_mode: str = ''):

    """Combined all features, left joining with starting dataframe on KEY_COLS, and save to delta format per month,
    and return combined dataframe

     Args:
         df: Input Pyspark DataFrame
         ft_name_list: Name of feature, must available in ft_save_pth
         ft_save_pth: Path that contain all combine features (default = FEATURE_PTH)
         save_mode: Select between '' (none), overwrite (overwrite all files), and soft overwrite
         (overwrite only missing month)

     Returns:
         Combined DataFrame
     """

    save_modes = ['', 'overwrite', 'soft_overwrite']
    if save_mode not in save_modes:
        raise ValueError(f'Invalid save_mode. Expected one of: {save_modes}')

    if not ft_save_pth.endswith('/'):
        ft_save_pth += '/'

    save_pth = ft_save_pth + 'combined_ft'
    save_pth_temp = ft_save_pth + 'combined_ft_temp'
    is_exist, last_eom_date = check_delta_exist(save_pth)

    save_eom_list = _get_save_eom_list(is_exist, last_eom_date, save_mode)

    print(f'Combine date list: {save_eom_list}')

    for date_dt in save_eom_list:
        print(f'Combined at date {date_dt}')

        df_each_date = df.filter(F.col(DATE_COL_EOM) == date_dt).cache()
        start_count = df_each_date.count()

        for ft in ft_name_list:
            print(f'Joining with {ft}')
            df_ft = spark.read.format('delta').load(ft_save_pth + ft)\
                .filter(F.col(DATE_COL_EOM) == date_dt)
            df_each_date = df_each_date.join(df_ft, on=KEY_COLS + [DATE_COL, DATE_COL_EOM], how='left')
            df_each_date.write.format("delta") \
                .mode("overwrite") \
                .option("overwriteSchema", "true") \
                .option("header", "true") \
                .save(save_pth_temp)

            # Reload
            df_each_date = spark.read.format('delta').load(save_pth_temp)\
                .filter(F.col(DATE_COL_EOM) == date_dt)

            # Check dup
            assert df_each_date.count() == start_count

        df_join = spark.read.format('delta').load(save_pth_temp)\
            .filter(F.col(DATE_COL_EOM) == date_dt)
        print(f'Shape ({df_join.count()}, {len(df_join.columns)})')

        print(f'Saving to {save_pth}: date {date_dt}')
        df_join.write.format("delta") \
            .mode("overwrite") \
            .partitionBy(DATE_COL_EOM) \
            .option("header", "true") \
            .option("replaceWhere", f"{DATE_COL_EOM} == '{date_dt}'") \
            .save(save_pth)

        spark.catalog.clearCache()  # Clear cache

    print('Load combined')
    df = spark.read.format('delta').load(save_pth)

    # print('Delete temp folder')
    # if ROOT_DIR.startswith('gs://'):
    #     os.system(f"gsutil rm -r {save_pth_temp}")
    # else:
    #     os.system(f"rm -r {save_pth_temp}")

    return df


def _get_save_eom_list(is_exist, last_eom_date, save_mode):

    # Use for combined features to find require save date list

    str_log = ''
    if is_exist:
        str_log += f'Combined features exists:'
        if save_mode == 'overwrite':
            str_log += f' Overwrite'
            print(str_log)
            save_eom_list = DATE_EOM_LIST

        elif (save_mode == 'soft_overwrite') & (last_eom_date is not None):
            str_log += f' Soft Overwrite, starting from {last_eom_date}'
            print(str_log)
            save_eom_list = get_date_list(last_eom_date, END_DT)

        else:
            str_log += f' Skipping write'
            print(str_log)
            save_eom_list = []
    else:
        save_eom_list = DATE_EOM_LIST

    return save_eom_list


def check_delta_exist(pth: str, check_date: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    """Check if specify path exists and also get the latest partition that is saved (partition by date_col_eom)

     Args:
         pth: Specify delta path
         check_date: If specify, check if the data at the exact date exists

     Returns:
         Bool if the path exist the date of latest partition that is not saved.
         If all partition is saved, the date will return None. If check_date specified, it will return Bool if
         the data at that date exist.
     """

    last_eom_date = DATE_EOM_LIST[0]
    is_exist = check_file_exist(pth, GCS_LIST_DIR)

    if is_exist:
        date_order = 0

        while check_file_exist(f'{pth}/Date_EOM={last_eom_date}', GCS_LIST_DIR):
            if last_eom_date == DATE_EOM_LIST[-1]:
                last_eom_date = None
                break
            date_order += 1
            last_eom_date = DATE_EOM_LIST[date_order]

        # Test at exact date if check_date specify
        if check_date is not None:
            df = spark.read.format('delta').load(pth)
            if df.filter(F.col(DATE_COL) == check_date).count() <= 0:
                is_exist = False

    return is_exist, last_eom_date


def get_discount_feature(df: DataFrame) -> DataFrame:
    """Return dataframe with three additional columns.

    Args:
        df: Input PySpark DataFrame with original CJ features.

    Returns:
        A new dataframe of original features + revenue, discount and percentage_discount features.

    """

    df = (df.withColumn('revenue', F.col('avgPrice') - F.col('supPrice'))
          .withColumn('discount', F.col('avgPrice') - F.col('avgPriceDis'))
          .withColumn('percentage_discount', 100 * (F.col('discount') / F.col('avgPrice'))))

    return df


def get_store_cluster_feature(df: DataFrame) -> DataFrame:
    """Return dataframe with three additional columns.

    Args:
        df: Input PySpark DataFrame with original CJ features.

    Returns:
        A new dataframe of original features + revenue, discount and percentage_discount features.

    """

    newstore_feature = spark.read.format('csv').load(NEWSTORE_FILE, inferSchema=True, header=True)

    df = df.join(newstore_feature, how='left', on=['BranchCode'])

    return df


def get_feat_imps(df_pred: DataFrame, model, feature_col: str = 'features_col') -> DataFrame:
    """Get feature importance of a tree-based model

     Args:
         df_pred: Prediction dataframe
         model: Tree-based Pyspark trained model. Must be a model that has featureImportances class
         feature_col: Feature column name

     Returns:
         DataFrame with importance for each feature
     """

    attrs = sorted(
        (attr["idx"], attr["name"]) for attr in (chain(*df_pred
                                                       .schema[feature_col]
                                                       .metadata["ml_attr"]["attrs"].values())))

    feat_imps = [(name, float(model.featureImportances[idx]))
                 for idx, name in attrs
                 if model.featureImportances[idx]]

    schema = StructType([StructField('feature_name', StringType(), True),
                         StructField('importance', DoubleType(), True)])

    df_imps = spark.createDataFrame(data=feat_imps, schema=schema)
    df_imps = df_imps.withColumn('cumsum', F.sum('importance').over(Window.orderBy(F.col('importance').desc())))

    return df_imps.sort(F.col('importance').desc())


def save_spark_model(df_train,
                     model_class,
                     save_pth,
                     feature_col='features_col',
                     label_col='TotalQtySale',
                     transform_model=None,
                     is_overwrite=False,
                     **kwargs):
    """Train and Save spark model

     Args:
         df_train: Traning dataframe
         model_class: Pyspark model type
         save_pth: Path to save model
         feature_col: Feature column name
         label_col: Label column name
         transform_model: Transform model, if specify will add to Pipeline stages
         is_overwrite: Bool if want to overwrite the model or not
         **kwargs: Model parameters

     Returns:
         None
     """

    model_save_pth = save_pth + 'saved_model'
    is_model_exist = check_file_exist(model_save_pth, GCS_LIST_DIR)

    str_log = ''
    if is_model_exist:
        str_log += f'Model exists:'
        if is_overwrite:
            str_log += f' Remove existing model'
            print(str_log)
            os.system(f"gsutil rm -r {model_save_pth}")

        else:
            str_log += f' Skipping'
            print(str_log)
            return None

    model = fit_model_spark(df_train=df_train,
                            model_class=model_class,
                            feature_col=feature_col,
                            label_col=label_col,
                            **kwargs)

    if transform_model is not None:
        # Combine transform model with prediction model
        combined_model = PipelineModel(stages=[transform_model, model])

    else:
        combined_model = model

    combined_model.save(model_save_pth)

    return None
