"""
    This is a sample script showing how to get started with reclist. In particular,
    we are targeting a "complementary items" use cases, such as for example a cart recommender:
    if a shopper added item X to the cart, what is she likely to add next?

    We train a simple, yet effective prod2vec baseline model (https://arxiv.org/abs/2007.14906),
    re-using for convenience a "training embedding" function already implemented by recsys. The script
    show how easy it is to run behavioral tests on a target dataset, in this case a wrapper around a large
    e-commerce dataset (the Coveo Data Challenge dataset: https://github.com/coveooss/SIGIR-ecom-data-challenge).

    If you want to run your own model on the same dataset and the same behavioral test suite, just implement
    your model prediction using the RecModel, and pass it to the CartList.
"""
from reclist.datasets import CoveoDataset
from reclist.recommenders.prod2vec import CoveoP2VRecModel
from reclist.reclist import CoveoCartRecList

if __name__ == "__main__":

    # get the coveo data challenge dataset as a RecDataset object
    coveo_dataset = CoveoDataset()

    # re-use a skip-gram model from reclist to train a latent product space, to be used
    # (through knn) to build a recommender
    model = CoveoP2VRecModel()
    model.train(coveo_dataset.x_train)

    # instantiate rec_list object, prepared with standard quantitative tests
    # and sensible behavioral tests (check the paper for details!)
    rec_list = CoveoCartRecList(
        model=model,
        dataset=coveo_dataset
    )
    # invoke rec_list to run tests
    rec_list(verbose=True)

