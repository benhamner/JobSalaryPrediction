import data_io
import pickle

def main():
    print("Loading the classifier")
    classifier = data_io.load_model()
    
    print("Making predictions") 
    valid = data_io.get_valid_df()
    predictions = classifier.predict(valid)   
    predictions = predictions.reshape(len(predictions), 1)

    print("Writing predictions to file")
    data_io.write_submission(predictions)

if __name__=="__main__":
    main()