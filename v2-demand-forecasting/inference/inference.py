import sys

sys.path.append('../')

from train.features import *
from train.config import *
from inference import spark
# from mmlspark.lightgbm import LightGBMRegressor, LightGBMClassifier


SCORING_DATE = job_args['scoring_dt'] if 'scoring_dt' in job_args.keys else str(date.today())
SCORING_LAG_DAYS = 7
LATEST_DATA_DT = add_days(SCORING_DATE, -SCORING_LAG_DAYS)

# Path to files

data_nm = 'all_sku_2020_v4'
INFER_DATA_PTH = ROOT_DIR + f'demand_prediction_pipeline/{data_nm}'
INFER_FEATURE_PTH = ROOT_DIR + f'feature_inf/'
SCORING_PTH = ROOT_DIR + 'scoring/'
FEATURE_OVERWRITE = False
COMBINE_SAVE_MODE = 'soft_overwrite'


def get_mock_data(df):
    df_scoring = df.filter(F.col(DATE_COL) == LATEST_DATA_DT).drop(DATE_COL)
    df_scoring = df_scoring.withColumn(DATE_COL, F.lit(SCORING_DATE))
    df_scoring = df_scoring.withColumn(LABEL_COL, None)
    df_full = df.unionByName(df_scoring)

    return df_full


def main():

    print('###### INFERENCE MODE ######')
    print(f'# SCORING DATE {SCORING_DATE}')
    print(f'# SAVE/LOAD INFERENCE FEATURE AT {INFER_FEATURE_PTH}')
    print(f'# FEATURE OVERWRITE: {FEATURE_OVERWRITE}, COMBINED SAVE MODE: {COMBINE_SAVE_MODE}')
    print(f'# SAVE/LOAD MODEL AT {MODEL_PTH}')
    print('###########################')

    ####### FEATURES #######
    print(f'Load data at {DATA_PTH}')

    testing_dataset = spark.read.format('csv').load(INFER_DATA_PTH, inferSchema=True, header=True)

    testing_dataset.show(2)

    df = get_mock_data(testing_dataset)  # Add scoring date
    df = df.withColumn(DATE_COL_EOM, F.last_day(DATE_COL))  # Create partition column

    # Save basic features
    save_feature_delta(df, get_basic_feature, 'basic_ft',
                       ft_start_dt=add_days(SCORING_DATE, 180 + SCORING_LAG_DAYS),
                       end_data_dt=SCORING_DATE,
                       is_overwrite=True,
                       save_pth=INFER_FEATURE_PTH,
                       **{'promo_perc_cutoff': 1})

    # Reload and use this to create other features
    df = spark.read.format('delta').load(FEATURE_PTH + 'basic_ft')

    if 'label' not in df.columns:
        df = df.withColumn('label', F.when(F.col(LABEL_COL) > 0, 1
                                           ).otherwise(0))

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
                    'cat_lv2_ds': ['product_cat_lv2_cj', 'DistrictEN'],
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
                                               'lag_day': SCORING_LAG_DAYS}

    for name, consc_group_col in consc_cols_dict.items():
        FEATURE_PARAMS[f'consc_{name}_ft'] = {'group_cols': consc_group_col,
                                              'flag_col': 'label',
                                              'group_name': name,
                                              'agg_day_list': ts_agg_day_list,
                                              'lag_day': SCORING_LAG_DAYS}

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
                               ft_start_dt=SCORING_DATE,
                               end_data_dt=SCORING_DATE,
                               is_overwrite=FEATURE_OVERWRITE,
                               save_pth=INFER_FEATURE_PTH + feature_name,
                               **params)

            feature_name_all.append(feature_name)

    ####### COMBINED #######
    print(f'Combine features')
    df = combine_features(df,
                          feature_name_all,
                          ft_save_pth=INFER_FEATURE_PTH,
                          save_mode=COMBINE_SAVE_MODE)

    # Select method to handle nulls (now treat as zero (for num_cols) and others (for cat_cols))
    df = df.fillna(0).fillna('others').cache()

    print(df.show(2))

    ###### SCORING #######
    # Feature Transformation
    print(f'Scoring: Load model from {MODEL_PTH}')
    model = PipelineModel.load(MODEL_PTH + 'saved_model')
    df_scoring = model.transform(df).select([*KEY_COLS, DATE_COL,
                                             'MaterialCode', 'BranchCode', 'prediction']).cache()

    # Save prediction in csv
    print(f'Save scoring result at {SCORING_PTH}')
    save_spark_csv(df_scoring, SCORING_PTH, f'demand_forecasting_scoring_{SCORING_DATE}')

    return None


if __name__ == '__main__':
    # start time
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
