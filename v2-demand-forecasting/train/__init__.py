from dependencies.spark import initial_spark

JAR_PACKAGES = ['io.delta:delta-core_2.12:0.8.0',
                'com.microsoft.ml.spark:mmlspark_2.12:1.0.0-rc3-88-45379694-SNAPSHOT',
                'org.apache.spark:spark-avro_2.12:2.4.4']
SPARK_CONFIG = {'spark.jars.repositories': 'https://mmlspark.azureedge.net/maven',
                'spark.sql.extensions': 'io.delta.sql.DeltaSparkSessionExtension'}


spark, spark_logger, config_dict = initial_spark(app_name='my_demand_forecasting_app',
                                                 master='local[*]',
                                                 jar_packages=JAR_PACKAGES,
                                                 spark_config=SPARK_CONFIG)
