# CJ - Store Item Demand Forecasting Challenge

For most retailers, demand planning systems take a fixed, rule-based approach to forecast and replenishment order management.

Our ScourBoard could see here: http://bit.ly/tildi-scoreboard-dmf

## CJ-Demand Forecasting Project Structure

The CJ demand forecasting project structure is as follows:

```bash
root/
 |-- configs/
 |   |-- ml_config.json
 |-- dependencies/
 |   |-- logging.py
 |   |-- spark.py
 |-- docs/
 |   |-- Fast_DBSCAN_Algorithm.pdf
 |-- jobs/
 |   |-- baseline.py
 |-- models/
 |   |-- dbscan.py
 |   |-- LigthGBM.py
 |-- notebook/
 |   |-- baseline_human.ipynb
 |-- utils/
 |   |-- metrics.py
 |-- tests/
 |   |-- test_ml_job.py
 |   libs.zip
```

## Automated Testing

In order to test with Spark, we use the `pyspark` Python package, which is bundled with the Spark JARs required to programmatically start-up and tear-down a local Spark instance, on a per-test-suite basis (we recommend using the `setUp` and `tearDown` methods in `unittest.TestCase` to do this once per test-suite). Note, that using `pyspark` to run Spark is an alternative way of developing with Spark as opposed to using the PySpark shell or `spark-submit`.

Given that we have chosen to structure our ML jobs in such a way as to isolate the 'Transformation' step into its own function (see 'Structure of an ML job' above), we are free to feed it a small slice of 'real-world' production data that has been persisted locally - e.g. in `./tests` or some easily accessible network directory - and check it against known results (e.g. computed manually or interactively within a Python interactive console session).

To execute the example unit test for this project run,

```bash
pipenv run python -m unittest tests/test_*.py
```
