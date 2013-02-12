import csv
import os
import pandas as pd
import pickle

def get_data_path():
    return os.path.join(os.environ["DataPath"],
                        "Adzuna", "Release0")

def get_submissions_path():
    return os.path.join(os.environ["DataPath"],
                        "Adzuna", "Submissions")

def get_train_df():
    train_path = os.path.join(get_data_path(), 
                              "train.csv")
    return pd.read_csv(train_path)

def get_valid_df():
    valid_path = os.path.join(get_data_path(), 
                              "valid.csv")
    return pd.read_csv(valid_path)

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
    writer.writerows(predictions)