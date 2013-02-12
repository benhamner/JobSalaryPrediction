import data_io
from features import FeatureMapper, SimpleTransform
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

from pdb import set_trace

def get_title(d):
    pickle.dump(d, open("d.pickle", "w"))
    return d.Title

def feature_extractor():
    features = [('FullDescription-Bag of Words', 'FullDescription', CountVectorizer(max_features=100))]
    combined = FeatureMapper(features)
    return combined

def get_pipeline():
    features = feature_extractor()
    steps = [("extract_features", features),
             ("classify", RandomForestRegressor(n_estimators=25, 
                                                verbose=2,
                                                n_jobs=1,
                                                min_samples_split=30))]
    return Pipeline(steps)

def main():
    print("Reading in the training data")
    train = data_io.get_train_df()

    print("Extracting features and training")
    classifier = get_pipeline()
    classifier.fit(train, train["SalaryNormalized"])

    print("Saving the classifier")
    #data_io.save_model(classifier, "model.pickle")
    
    print("Making predictions") 
    valid = data_io.get_valid_df()
    predictions = classifier.predict(valid)   
    predictions = predictions.reshape(len(predictions), 1)
    data_io.write_submission("random_forest_benchmark.csv", predictions)

if __name__=="__main__":
    main()