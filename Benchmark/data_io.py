import csv
import numpy as np
import os
import pandas as pd
import pickle

def identity(x):
    return x

# For pandas >= 10.1 this will trigger the columns to be parsed as strings
converters = { "FullDescription" : identity
             , "Title": identity
             , "LocationRaw": identity
             , "LocationNormalized": identity
             }

def get_data_path():
    return os.path.join(os.environ["DataPath"],
                        "Adzuna", "Release0")

def get_submissions_path():
    return os.path.join(os.environ["DataPath"],
                        "Adzuna", "Submissions")

def get_train_df():
    train_path = os.path.join(get_data_path(), 
                              "train.csv")

    return pd.read_csv(train_path, converters=converters)

def get_valid_df():
    valid_path = os.path.join(get_data_path(), 
                              "valid.csv")
    return pd.read_csv(valid_path, converters=converters)

def save_model(model, file_name):
    out_path = os.path.join(os.environ["DataPath"],
                            "Adzuna",
                            "TrainedModels",
                            file_name)
    pickle.dump(model, open(out_path, "w"))

def load_model(file_name):
    in_path = os.path.join(os.environ["DataPath"],
                           "Adzuna",
                           "TrainedModels",
                           file_name)
    return pickle.load(open(in_path))

def write_submission(file_name, predictions):
    writer = csv.writer(open(os.path.join(get_submissions_path(), file_name)
                             , "w"), lineterminator="\n")
    valid = get_valid_df()
    rows = [x for x in zip(valid["Id"], predictions)]
    writer.writerow(("Id", "SalaryNormalized"))
    writer.writerows(rows)