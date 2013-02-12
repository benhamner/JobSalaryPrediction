import data_io

def main():
    print("Reading in the training data")
    train = data_io.get_train_df()

    mean = train["SalaryNormalized"].mean()
    print("The mean salary is %f" % mean)

    print("Reading Valid.csv")
    valid = data_io.get_valid_df()

    print("Making predictions")
    predictions = [[mean] for i in range(len(valid))]

    print("Writing the submission") 
    data_io.write_submission("mean_benchmark.csv", predictions)

if __name__=="__main__":
    main()