import numpy as np
import matplotlib.pyplot as plt
import os
from reclist.current import current

def price_homogeneity_test(y_test, y_preds, product_data, bins=25, debug=True, key='PRICE'):
    """

    :param y_test: test data
    :param y_preds: predicted data
    :param product_data: catalog data
    :param bins: bins on which we split the data for the product comparison
    :param debug: shows plots
    :param key: key from the product data to find the product price
    @return:
    """
    abs_log_price_diff = []
    for idx, (_y, _y_pred) in enumerate(zip(y_test, y_preds)):
        # need >=1 predictions
        if not _y_pred:
            continue
        # item must be in product data
        if _y[0] not in product_data or _y_pred[0] not in product_data:
            continue
        if product_data[_y[0]][key] and product_data[_y_pred[0]][key]:
            pred_item_price = float(product_data[_y_pred[0]][key])
            y_item_price = float(product_data[_y[0]][key])
            if pred_item_price and y_item_price:
                abs_log_price_diff.append(np.abs(np.log10(pred_item_price)-(np.log10(y_item_price))))

    histogram = np.histogram(abs_log_price_diff, bins=bins, density=False)
    histogram = (histogram[0].tolist(), histogram[1].tolist())
    if debug:
        # debug / viz
        plt.hist(abs_log_price_diff, bins=25)
        plt.savefig(os.path.join(current.report_path,
                                 'plots',
                                 'price_homogeneity.png'))
        plt.clf()

    return {
        'mean': np.mean(abs_log_price_diff),
        'histogram': histogram
    }
