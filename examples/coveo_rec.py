from reclist.datasets import CoveoDataset
from reclist.recommenders.prod2vec import P2VRecModel
from reclist.utils.train_w2v import train_embeddings
from reclist.reclist import CoveoCartRecList

if __name__ == "__main__":

    coveo_dataset = CoveoDataset()
    print(len(coveo_dataset.x_train))

    # get skus from training sessions
    x_train_skus = [[e['product_sku'] for e in s] for s in coveo_dataset.x_train]
    embeddings = train_embeddings(sessions=x_train_skus)
    model = P2VRecModel(model=embeddings)

    # instantiate rec_list object
    rec_list = CoveoCartRecList(
        model=model,
        dataset=coveo_dataset
    )
    # invoke rec_list to run tests
    rec_list(verbose=True)

