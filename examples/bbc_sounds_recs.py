import os
import json
from itertools import groupby

from reclist.datasets import BBCSoundsSampledDataset
from reclist.recommenders.lightfm_simulator import BBCSoundsLightFMSimulatorModel
from reclist.reclist import BBCSoundsRecList
from reclist.utils.config import get_cache_directory, download_file, BBC_SOUNDS_PREDICTIONS, load_predictions


def format_predictions(predictions):
    predictions_list = []
    for user_id, resource_ids in predictions.items():
        user_predictions = []
        for resource_id in resource_ids[0:12]:
            user_predictions.append({'userId_enc': user_id, 'resourceId': resource_id})
        predictions_list.append(user_predictions)
    return predictions_list


if __name__ == "__main__":

    sounds_dataset = BBCSoundsSampledDataset()

    model = BBCSoundsLightFMSimulatorModel()

    # load predictions
    load_from_cache = False
    cache_dir = get_cache_directory()
    predictions_filepath = os.path.join(cache_dir, "predictions.ndjson")

    if load_from_cache:
        print('Loading predictions from cache...')
        with open(os.path.join(cache_dir, predictions_filepath)) as f:
            predictions = [json.loads(line) for line in f]
    else:
        print('Loading predictions from: ', BBC_SOUNDS_PREDICTIONS)
        # load from the bucket
        predictions, _ = load_predictions(filename=BBC_SOUNDS_PREDICTIONS)
        # and save it locally
        if not os.path.exists(predictions_filepath):
            download_file(BBC_SOUNDS_PREDICTIONS, predictions_filepath)
        print('Predictions saved locally.')

    formatted_predictions = format_predictions(predictions)

    rec_list = BBCSoundsRecList(
        model=model,
        dataset=sounds_dataset,
        y_preds=formatted_predictions
    )

    rec_list(verbose=True)
