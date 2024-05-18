import sys

sys.path.append('../')

from train.features import *
from train.config import *
from train import spark
from mmlspark.lightgbm import LightGBMRegressor, LightGBMClassifier

FEATURE_OVERWRITE = False
COMBINE_SAVE_MODE = 'soft_overwrite'
MODEL_OVERWRITE = False


def main():
    print('====== TRAINING MODE ====== ')
    print(f'# SAVE/LOAD FEATURE AT {FEATURE_PTH}')
    print(f'# FEATURE OVERWRITE: {FEATURE_OVERWRITE}, COMBINED SAVE MODE: {COMBINE_SAVE_MODE}')
    print(f'# SAVE/LOAD MODEL AT {MODEL_PTH}')
    print(f'# MODEL OVERWRITE: {MODEL_OVERWRITE}')
    print('============================')

    ####### FEATURES #######
    print(f'Load data at {DATA_PTH}')

    df = spark.read.format('csv').load(DATA_PTH, inferSchema=True, header=True)
    df = df.withColumn(DATE_COL_EOM, F.last_day(DATE_COL))  # Create partition column
    df = df.withColumn('idx', F.monotonically_increasing_id())  # Generate key columns

    # df.show(2)

    # Save & Reload Basic features
    print(f'Save Basic Features')
    save_feature_delta(df, get_basic_feature, 'basic_ft',
                       ft_start_dt=START_DT,
                       ft_end_dt=END_DT,
                       is_overwrite=FEATURE_OVERWRITE,
                       **{'promo_perc_cutoff': 0.99})

    # Reload and use this to create other features
    df = spark.read.format('delta').load(FEATURE_PTH + 'basic_ft')

    if 'label' not in df.columns:
        df = df.withColumn('label', F.when(F.col(LABEL_COL) > 0, 1
                                           ).otherwise(0))

    # df.show(2)

    # Feature parameters
    ts_agg_day_list = [60, 90, 180]
    ts_sum_ratio_list = [(90, 180)]
    ts_agg_cols = ['TotalQtySale', 'TotalNetSale']

    ts_cols_dict = {'mc': ['MaterialCode'],
                    'br': ['BranchCode'],
                    'mc_dow': ['MaterialCode', 'DayOfWeek'],
                    'mc_pv': ['MaterialCode', 'ProvinceEN'],
                    'mc_ds': ['MaterialCode', 'DistrictEN'],
                    'cat_lv2': ['product_cat_lv2_cj'],
                    'cat_lv2_pv': ['product_cat_lv2_cj', 'ProvinceEN'],
                    'cat_lv2_dow': ['product_cat_lv2_cj', 'DayOfWeek'],
                    'cat_lv2_br': ['product_cat_lv2_cj', 'BranchCode'],
                    }

    consc_cols_dict = {'mc_br': ['MaterialCode', 'BranchCode'],
                       'mc_ds': ['MaterialCode', 'DistrictEN'],
                       'cat_lv2_ds': ['product_cat_lv2_cj', 'DistrictEN'],
                       'cat_lv2_br': ['MaterialCode', 'BranchCode']
                       }

    # Mapping between function: features
    FEATURE_DICT = {get_lat_long_feature: ['lat_long_ft'],
                    get_ts_agg_feature: [f'ts_agg_{name}_ft' for name in ts_cols_dict.keys()],
                    get_consecutive_feature: [f'consc_{name}_ft' for name in consc_cols_dict.keys()]}

    # Mapping between features: params
    FEATURE_PARAMS = {}

    for name, ts_group_col in ts_cols_dict.items():
        FEATURE_PARAMS[f'ts_agg_{name}_ft'] = {'group_cols': ts_group_col,
                                               'agg_cols': ts_agg_cols,
                                               'group_name': name,
                                               'agg_day_list': ts_agg_day_list,
                                               'sum_ratio_list': ts_sum_ratio_list,
                                               'lag_day': 5}

    for name, consc_group_col in consc_cols_dict.items():
        FEATURE_PARAMS[f'consc_{name}_ft'] = {'group_cols': consc_group_col,
                                              'flag_col': 'label',
                                              'group_name': name,
                                              'agg_day_list': ts_agg_day_list,
                                              'lag_day': 5}

    # Generate all features
    feature_name_all = []
    for feature_func, feature_name_list in FEATURE_DICT.items():
        for feature_name in feature_name_list:
            print(f'Save features {feature_name}')
            if feature_name in FEATURE_PARAMS.keys():
                params = FEATURE_PARAMS[feature_name]
            else:
                params = {}

            # Save features
            save_feature_delta(df, feature_func, feature_name,
                               ft_start_dt=START_DEV_DT,
                               end_data_dt=END_DT,
                               is_overwrite=FEATURE_OVERWRITE,
                               **params)
            feature_name_all.append(feature_name)

    ####### COMBINED #######
    print(f'Combine features')
    df = combine_features(df,
                          ft_name_list=feature_name_all,
                          ft_save_pth=FEATURE_PTH,
                          save_mode=COMBINE_SAVE_MODE)

    # basic_col_list = ['avgPriceDis', 'avgPrice', 'supPrice', 'DayOfMonth', 'DayOfWeek', 'Quarter',
    #                   'MonthQuarter', 'Month', 'WeekOfYear', 'BG003', 'DS001', 'DS002', 'DS005', 'DS004',
    #                   'DS003', 'FR001', 'GS002', 'DG001', 'FR002', 'TS004', 'GS003', 'TS002', 'DB002', 'BG005',
    #                   'BG007', 'FR010', 'FR003', 'FR006', 'GS014', 'GS006', 'GS007', 'TS001', 'FR004', 'GS015',
    #                   'ST001', 'FR007', 'FR011', 'DG002', 'FR009', 'BG009', 'latitude', 'longitude', 'revenue',
    #                   'discount','percentage_discount','YearOpened','MonthOpened']

    basic_col_list = ['avgPriceDis', 'avgPrice', 'supPrice', 'DayOfMonth', 'DayOfWeek', 'Quarter',
                      'MonthQuarter', 'Month', 'WeekOfYear', 'latitude', 'longitude', 'revenue',
                      'discount', 'percentage_discount', 'YearOpened', 'MonthOpened',
                      'BG003', 'DS001', 'DS002', 'DS005', 'DS004', 'DS003', 'FR001', 'GS002',
                      'DG001', 'FR002', 'GS003']

    ts_col_str = ['_sum', '_avg', '_stddev', '_cv']
    ts_col_list = [col for col in df.columns if
                   any([col_str in col for col_str in ts_col_str])]

    cat_col_list = ['LocationCluster', 'StoreType', 'ProvinceEN', 'DistrictEN',
                    'product_cat_lv1', 'product_cat_lv2_cj']
    consc_col_list = [col for col in df.columns if '_consc' in col]

    # Separate numerical columns & categorical columns
    num_cols = basic_col_list + ts_col_list + consc_col_list
    cat_cols = cat_col_list

    print(f'Number of features: {len(num_cols) + len(cat_cols)}')

    # Select method to handle nulls (now treat as zero (for num_cols) and others (for cat_cols))
    df = (df.fillna(0, subset=num_cols)
          .fillna('others', subset=cat_cols))

    # print(df.show(2))

    ###### MODEL #######

    df_train = df.filter((F.col(DATE_COL_EOM) >= START_DEV_DT_EOM)
                         & (F.col(DATE_COL_EOM) < START_OOT_DT_EOM))

    df_test = df.filter((F.col(DATE_COL_EOM) >= START_OOT_DT_EOM)
                        & (F.col(DATE_COL_EOM) <= END_OOT_DT_EOM))

    # Transform model
    transform_model = create_transform_pipeline(df_train, cat_cols, num_cols, onehot_ec=True)

    check_train_exist = check_file_exist(MODEL_PTH + 'training_dataset', GCS_LIST_DIR)
    check_test_exist = check_file_exist(MODEL_PTH + 'testing_dataset', GCS_LIST_DIR)

    if check_train_exist & check_test_exist:
        print('Transformed Training and Testing Dataset exists, Loading:')

    else:
        print('Transform Training and Testing Dataset:')

        df_train_tf = transform_model.transform(df_train) \
            .select(*KEY_COLS, 'MaterialCode', 'BranchCode', DATE_COL, LABEL_COL, 'features_col')

        df_test_tf = transform_model.transform(df_test) \
            .select(*KEY_COLS, 'MaterialCode', 'BranchCode', DATE_COL, LABEL_COL, 'features_col')

        print('Save train and test features')
        # Save train/test pipeline
        df_train_tf.write.format("delta") \
            .mode("overwrite") \
            .option("overwriteSchema", "true") \
            .option("header", "true") \
            .save(MODEL_PTH + 'training_dataset')

        df_test_tf.write.format("delta") \
            .mode("overwrite") \
            .option("overwriteSchema", "true") \
            .option("header", "true") \
            .save(MODEL_PTH + 'testing_dataset')

    spark.catalog.clearCache()

    # Reload
    df_train_tf = spark.read.format('delta').load(MODEL_PTH + 'training_dataset')
    df_test_tf = spark.read.format('delta').load(MODEL_PTH + 'testing_dataset')

    print(f'Number of training: {df_train_tf.count()}')
    print(f'Number of testing: {df_test_tf.count()}')

    print('Model Fitting')
    # Fit model

    ###### Change model class here ######
    model_class = RandomForestRegressor
    # model_class = LightGBMRegressor
    model_params = {}
    ####################################

    save_spark_model(df_train_tf,
                     model_class,
                     save_pth=MODEL_PTH,
                     label_col=LABEL_COL,
                     transform_model=transform_model,
                     is_overwrite=MODEL_OVERWRITE,
                     **model_params)

    # Load model
    combined_model = PipelineModel.load(MODEL_PTH + 'saved_model')
    model = combined_model.stages[-1]

    # Transform
    df_test_prediction = model.transform(df_test_tf)

    # Save prediction
    print('Save test prediction')
    df_test_prediction.write.format("delta") \
        .mode("overwrite") \
        .option("overwriteSchema", "true") \
        .option("header", "true") \
        .save(MODEL_PTH + 'testing_prediction')

    df_test_prediction = spark.read.format('delta').load(MODEL_PTH + 'testing_prediction')
    # df_test_prediction.show(2)

    # Evaluate
    result = demand_forecasting_evaluator(df_test_prediction, label_col=LABEL_COL, prediction_col="prediction")
    print(result)

    # Feature importance
    print(f'Save feature importance at {MODEL_PTH + "feat_imps/"}')
    df_imps = get_feat_imps(df_test_prediction, model)
    save_spark_csv(df_imps, MODEL_PTH + "feat_imps/", 'importance')

    return None


if __name__ == '__main__':
    # start time
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
