Benchmarks
==========

This folder contains a couple basic benchmarks to provide examples of reading in the data and creating a submission.

Executing these benchmarks requires python along with the pandas package.

To run them,

1. [Download the data](https://www.kaggle.com/c/job-salary-prediction/data)
2. Modify the get_paths function in util.py to point to the data path and the submission output path on your system
3. Run the benchmarks by executing the corresponding script (e.g. `python random_forest_benchmark.py`)
4. [Make a submission](https://www.kaggle.com/c/job-salary-prediction/submissions/attach) with the output file

The benchmarks are:

 - **mean_benchmark.py**: predicts the mean sale price from the training set
 - **random_forest_benchmark.py**: converts the training set to continuous and categorical features and then trains a random forest

**Note:** the data ingest script requires pandas version 10.1 or later.