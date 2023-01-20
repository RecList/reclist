from reclist.datasets import SyntheticDataset
from reclist.reclist import SyntheticRatingClassifierRecList
from reclist.recommenders import randomforest

if __name__ == "__main__":
    # TODO: Add config file for this SyntheticDataset & way to name it while saving it
    # get the synthetic dataset as a RecDataset object
    synthetic_dataset = SyntheticDataset()

    # instantiate a random forest model
    model = randomforest.RandomForest(regression_problem=False)
    model.train(synthetic_dataset.x_train, synthetic_dataset.y_train)

    # instantiate rec_list object, prepared with standard quantitative tests
    # and sensible behavioral tests (check the paper for details!)
    rec_list = SyntheticRatingClassifierRecList(model=model, dataset=synthetic_dataset)
    # invoke rec_list to run tests
    rec_list(verbose=True)
