import data_io
import numpy as np
import pickle

def main():
    print("Loading the model")
    model = data_io.load_model()
    
    print("Making predictions") 
    valid = data_io.get_valid_df()
    predictions = model * np.ones(len(valid))

    print("Writing predictions to file")
    data_io.write_submission(predictions)

if __name__=="__main__":
    main()