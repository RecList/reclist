#%%
import os
import argparse
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--movies_data_path", type=str, help="Path to movies.csv")

    parser.add_argument("--ratings_data_path", type=str, help="Path to ratings.csv")

    parser.add_argument("--output_folder_path", type=str, help="Path to output folder")

    parser.add_argument(
        "--random_seed", type=int, help="Seed for dataset split generation", default=0
    )
    parser.add_argument(
        "--train_frac",
        type=float,
        help="Fraction of data to use for training set",
        default=0.8,
    )

    args = parser.parse_args()

    MOVIES_DATA_PATH = args.movies_data_path
    RATINGS_DATA_PATH = args.ratings_data_path
    OUTPUT_FOLDER_PATH = args.output_folder_path
    RANDOM_SEED = args.random_seed
    DATA_FRAC = args.train_frac

    if not os.path.exists(OUTPUT_FOLDER_PATH):
        raise FileNotFoundError

    ### MOVIE DATA PROCESSING ###
    # read in raw data
    df = pd.read_csv(MOVIES_DATA_PATH)
    # convert generes to list of genres
    df["genres"] = df["genres"].apply(lambda _: _.split("|"))
    # get id for genres
    df_explode = df.explode("genres")
    df_explode["genres_id"] = pd.factorize(df_explode["genres"])[0]
    # convert genres_id back to list
    df["genres_id"] = (
        df_explode[["movieId", "genres_id"]]
        .groupby("movieId")["genres_id"]
        .apply(list)
        .values
    )
    # re-label columns
    df.columns = ["movie_id", "title", "genres", "genres_id"]
    df = df.set_index("movie_id")
    # save as .pk file
    df.to_parquet(os.path.join(OUTPUT_FOLDER_PATH, "movielens_catalog.pk"))

    ### RATING DATA PROCESSING ###
    df = pd.read_csv(RATINGS_DATA_PATH)
    df = df.reset_index()
    # re-label columns
    df.columns = ["rating_id", "user_id", "movie_id", "rating", "timestamp"]
    df = df.set_index("rating_id")
    # normalize ratings
    df["rating"] /= 5

    # create random data split
    df_train = df.sample(frac=DATA_FRAC, replace=False, random_state=RANDOM_SEED)
    df_test = df.loc[~df.index.isin(df_train.index)]

    df_train[["user_id", "movie_id", "timestamp"]].to_parquet(
        os.path.join(OUTPUT_FOLDER_PATH, "movielens_x_train.pk")
    )
    df_train[["rating"]].to_parquet(
        os.path.join(OUTPUT_FOLDER_PATH, "movielens_y_train.pk")
    )
    df_test[["user_id", "movie_id", "timestamp"]].to_parquet(
        os.path.join(OUTPUT_FOLDER_PATH, "movielens_x_test.pk")
    )
    df_test[["rating"]].to_parquet(
        os.path.join(OUTPUT_FOLDER_PATH, "movielens_y_test.pk")
    )
