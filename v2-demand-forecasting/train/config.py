from utils.libs import *

# Get parameters from submit -- --job-args
# Supported parameters:
# - bucket_name
# - ft_suffix
# - model_suffix
# - scoring_dt (Use for scoring)

parser = argparse.ArgumentParser(description='Run a PySpark job')

parser.add_argument('--job-args', nargs='*',
                    help="Extra arguments to send to the PySpark job")

args = parser.parse_args()
job_args = dict()

if args.job_args:
    job_args_tuples = [arg_str.split('=') for arg_str in args.job_args]
    job_args = {a[0]: a[1] for a in job_args_tuples}

print(f'passing parameters: {job_args}')

# Columns name
KEY_COLS = ['idx']
DATE_COL = 'Date'
DATE_COL_EOM = f'{DATE_COL}_EOM'
LABEL_COL = 'TotalQtySale'

# Date Parameters
# Develop
START_DT = '2020-01-01'
START_DT_EOM = get_date_eom(START_DT)
START_DEV_DT = '2020-06-01'
START_DEV_DT_EOM = get_date_eom(START_DEV_DT)

# Out of time testing
START_OOT_DT = '2020-12-01'
START_OOT_DT_EOM = get_date_eom(START_OOT_DT)
END_DT = '2020-12-31'
END_OOT_DT_EOM = get_date_eom(END_DT)

DATE_EOM_LIST = get_date_list(START_DEV_DT, END_DT)

# File path parameters
if 'bucket_name' in job_args.keys():
    ROOT_DIR = f"gs://{job_args['bucket_name']}/"

    print('Get current path for this bucket')
    GCS_LIST_DIR = get_gcs_list_dir(ROOT_DIR)

else:
    ROOT_DIR = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    GCS_LIST_DIR = None

# data_nm = 'tildi_demandforecasting_type1_v1'  # 'tildi_demandforecasting_type1_poc'
# DATA_PTH = ROOT_DIR + f'data/{data_nm}.csv'

data_nm = 'all_sku_2020_v3'
DATA_PTH = ROOT_DIR + f'demand_prediction_pipeline/{data_nm}'

# Feature Path
if 'ft_suffix' in job_args.keys():
    FEATURE_PTH = ROOT_DIR + f"features_{job_args['ft_suffix']}/"
elif 'scoring_dt' in job_args.keys():
    FEATURE_PTH = ROOT_DIR + f"features_{job_args['scoring_dt']}/"
else:
    FEATURE_PTH = ROOT_DIR + f'features/'

# Model Path
if 'model_suffix' in job_args.keys():
    MODEL_PTH = ROOT_DIR + f"model/model_{data_nm}_{job_args['model_suffix']}/"
else:
    MODEL_PTH = ROOT_DIR + f'model/model_{data_nm}/'

# File directory
MISC_PTH = ROOT_DIR + 'misc/'
NEWSTORE_FILE = MISC_PTH + 'newstore_cluster_features.csv'
LOCATION_FILE = MISC_PTH + 'location.csv'
LATLONG_FILE = MISC_PTH + 'lat_long_province_zipcode.csv'
CATEGORY_FILE = MISC_PTH + 'cj_category_v1.csv'
CAT_MAPPING_FILE = MISC_PTH + f'cat_mapping_lv1_cj.csv'
EMBEDDING_FILE = MISC_PTH + f'cat_embedding_dict.pkl'
