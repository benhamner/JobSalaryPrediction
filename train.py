import data_io
import numpy as np
import pickle

def main():
    print("Reading in the training data")
    train = data_io.get_train_df()

    mean = train["SalaryNormalized"].mean()
    print("The mean salary is %f" % mean)

    print("Saving the model")
    data_io.save_model(mean)
    
if __name__=="__main__":
    main()