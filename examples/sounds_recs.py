from reclist.datasets import BBCSoundsDataset
from reclist.recommenders.prod2vec import BBCSoundsP2VRecModel
from reclist.reclist import BBCSoundsSimilarItemRecList

if __name__ == "__main__":

    # get the MovieLens 25M dataset as a RecDataset object
    sounds_dataset = BBCSoundsDataset()

    model = BBCSoundsP2VRecModel()
    model.train(sounds_dataset.x_train)

    rec_list = BBCSoundsSimilarItemRecList(
        model=model,
        dataset=sounds_dataset
    )

    rec_list(verbose=True)
