"""
-------------
baseline.py
-------------

This 'CJ-Demand Forecasting' project contains an Apache Spark ML job 
definition that implements baseline for production jobs. 

"""

import sys
sys.path.append('../')

from utils import libs, metrics
from models import dbscan
#from dependencies.spark import start_spark

# define features
label_cols = ['TotalQtySale']
numeric_features = ['avgPriceDis','avgPrice','Yearday','Month',
                    'DayofWeek','Year','Quarter','WeekOfYear','DayofMonth']
string_features = ['BranchCode','MaterialCode','types'] # types = promotions

# initiate variables
select_variables_initial = ['BranchCode','MaterialCode','Date','avgPriceDis',
                            'avgPrice','supPrice','label','totalNetSale',
                            'TotalQtySale','types','Branch','Name']

# define the model name
version = 'BASE_MODEL_001'
versionFeaturePL = "BASE_PL_MODEL_001"

Pipeline_Model_PATH = "model_pipeline_"+str(versionFeaturePL)+".plmodel"
Model_PATH = "model_ml_"+str(version)+".plmodel"

# input
data = './tildi_demandforecasting_type1_poc.csv'
data_location = './location.csv'
data_category = './category.csv'

# training/testing split date
split_date = '2020-12-01'

# output
result_list = []

def main():
    """Main ML script definition.

    :return: None
    """
    # start Spark application and get Spark session, logger and config
    spark, log, config = start_spark(
        app_name='my_ml_job',
        files=['configs/ml_config.json'])

    # log that main ETL job is starting
    log.warn('cj_ml_job is up-and-running')

    # import data 
    df = (spark
          .read 
          .format("csv")
          .option("header", "true")
          .load(data))

    df_location = (spark
                  .read.format("csv")
                  .option("header", "true")
                  .load(data_location))

    df_category = (spark
                  .read.format("csv")
                  .option("header", "true")
                  .load(data_category))

    # cache data
    df.cache()
    df_location.cache()
    df_category.cache()

    # initial variables
    df = df.select(select_variables_initial)

    # join location
    df = df.join(df_location, how='left',on =['BranchCode'] )

    # clean
    df = df.dropna(how='all')

    # casting type
    df = cast_double_types(df, ['avgPriceDis','avgPrice','TotalQtySale','supPrice','label','totalNetSale',])
    df = cast_int_types(df, ['label','ZipCode'])
    df = cast_date_types(df, ['Date'])

    # extracting date feature
    df = extract_date(df,'Date')

    # clean
    df2 = df.dropna()
    # check missing value
    # df2.select(*(sum(col(c).isNull().cast("int")).alias(c) for c in df2.columns)).show()

    # Train/Test Split
    df_train = (df
                .filter(df["Date"]<split_date)) 
    df_test = (df
                .filter((df["Date"]>=split_date)))

    # creating new features: item average as prediction
    df_train_avg = (df_train.groupBy("BranchCode", "MaterialCode")
                           .agg(avg("TotalQtySale")
                           .alias('prediction_avg')))
    df_train_avg_dow = (df_train.groupBy("BranchCode", "MaterialCode", "DayofWeek")
                              .agg(avg("TotalQtySale")
                               .alias('prediction_avg_dow')))
    df_train_avg_by_month = (df_train.groupBy("BranchCode", "MaterialCode", "Month")
                                    .agg(avg("TotalQtySale")
                                    .alias('prediction_avg_month')))
    df_train_avg_all_store = (df_train.groupBy("MaterialCode")
                                      .agg(avg("TotalQtySale")
                                      .alias('prediction_avg_all_store')))
    df_train_avg_dow_all_store = (df_train.groupBy("MaterialCode", "DayofWeek")
                                          .agg(avg("TotalQtySale")
                                          .alias('prediction_avg_dow_all_store')))
    df_train_avg_by_month_all_store = (df_train.groupBy("MaterialCode", "Month")
                                              .agg(avg("TotalQtySale")
                                              .alias('prediction_avg_month_all_store')))

    # join new features
    df_test2 = (df_test
            .join(df_train_avg_all_store, 
                  how='left', 
                  on =['MaterialCode']))
    df_test2 = (df_test2
                .join(df_train_avg_dow_all_store, 
                      how='left', 
                      on =['MaterialCode','DayofWeek']))

    # Create ML Pipiline
    _stages = []
    string_indexer =  [StringIndexer(inputCol = column , \
                                     outputCol = column + '_StringIndexer', 
                                     handleInvalid = "skip") for column in string_features]

    one_hot_encoder = [OneHotEncoder(
        inputCols = [column + '_StringIndexer' for column in string_features ], \
        outputCols =  [column + '_OneHotEncoder' for column in string_features ])]

    vect_indexer = [VectorIndexer(
        inputCol = column + '_OneHotEncoder',
        outputCol = column + '_VectorIndexer', 
        maxCategories=10) for column in string_features]

    assemblerInput =  [f  for f in numeric_features]  
    assemblerInput += [f + "_VectorIndexer" for f in string_features]
    vector_assembler = VectorAssembler(inputCols = assemblerInput, \
                                       outputCol = 'VectorAssembler_features')

    dtr = DecisionTreeRegressor(labelCol="TotalQtySale",  
                                featuresCol = 'VectorAssembler_features', 
                                predictionCol="prediction")
    rfr = RandomForestRegressor(labelCol="TotalQtySale",  
                                featuresCol = 'VectorAssembler_features', 
                                predictionCol="prediction")
    gbtr = GBTRegressor(labelCol="TotalQtySale",  
                        featuresCol = 'VectorAssembler_features', 
                        predictionCol="prediction")

    _stages += string_indexer
    _stages += one_hot_encoder
    _stages += vect_indexer
    _stages += [vector_assembler]
    _stages += [gbtr]

    # Train Model
    pipeline = Pipeline(stages = _stages)
    model = pipeline.fit(df_train)

    # Save and Load Model
    (model.write()
        .overwrite()
        .save(Pipeline_Model_PATH))
    read_plmodel = (PipelineModel.read()
                                .load(Pipeline_Model_PATH))

    # Predict Model
    df_predict = read_plmodel.transform(df_test)

    # Performance Evaluation: Overall
    actual = list(df_predict.select('TotalQtySale')
                            .toPandas()['TotalQtySale'])
    pred = list(df_predict.select('prediction')
                            .toPandas()['prediction'])

    #for metricName in ['rmse','mse','r2','mae']:# all metrics
    for metricName in ['rmse','mae']:
        evaluator = RegressionEvaluator(labelCol="TotalQtySale", predictionCol="prediction", metricName=metricName)
        result = evaluator.evaluate(df_predict)
        print ('%s = %g' % (metricName,result))
        result_list.append(result)

    # mape
    mape = mape_spark(df_predict, label = 'TotalQtySale')
    result_list.append(mape)
    # print("mape =", mape)

    # rmsle
    rmsle = rmsle_spark(df_predict, label = 'TotalQtySale')
    result_list.append(rmsle)
    # print("rmsle =", rmsle)

    # accuracy
    acc = (1-(np.exp(rmsle)-1))*100
    result_list.append(acc)
    # print("acc =", acc)

    print(result_list)
    print("Inference Process: Done")

    # Performance Evaluation: OUT OF STOCK (OOS)
    overall_test_number = df_predict.count()
    oos_number = (df_predict
                .filter((df_predict["prediction"]<df_predict["TotalQtySale"])
                    &(df_predict["TotalQtySale"]!=0)).count())
    print("percent of oos", oos_number*100/overall_test_number) 

    # log the success and terminate Spark application
    log.warn('cj_ml_job is finished')
    
    #spark.stop()
    
    return None

def extract_date(df_extract_date, columns='SalDate'):
    """Extract Features From Date Variable.

    :param spark: Date object.
    :return: Spark DataFrame with date features.
    """
    split_col = F.split(df_extract_date[columns], '-')
    df_extract_date = (df_extract_date
                        .withColumn('DayofMonth', split_col.getItem(2).cast('integer')))
    df_extract_date = (df_extract_date
                        .withColumn('Yearday', F.dayofyear(F.col(columns)))
                        .withColumn('Month', F.month(F.col(columns)))
                        .withColumn('DayofWeek', F.dayofweek(F.col(columns)))
                        .withColumn('Year', F.year(F.col(columns)))
                        .withColumn('Quarter', F.quarter(F.col(columns)))
                        .withColumn('WeekOfYear', F.weekofyear(F.col(columns))))
      
    return df_extract_date

# entry point for CJ PySpark ML application
if __name__ == '__main__':
    # start time
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))