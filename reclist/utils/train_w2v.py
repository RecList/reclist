import gensim


def train_embeddings(
    sessions: list,
    min_c: int = 3,
    size: int = 48,
    window: int = 5,
    iterations: int = 15,
    ns_exponent: float = 0.75,
    is_debug: bool = True):
    """
    Train CBOW to get product embeddings with sensible defaults (https://arxiv.org/abs/2007.14906).

    :param sessions: list of lists, as user sessions are list of interactions
    :param min_c: minimum frequency of an event for it to be calculated for product embeddings
    :param size: output dimension
    :param window: window parameter for gensim word2vec
    :param iterations: number of training iterations
    :param ns_exponent: ns_exponent parameter for gensim word2vec
    :param is_debug: if true, be more verbose when training

    :return: trained product embedding model
    """
    model = gensim.models.Word2Vec(sentences=sessions,
                                   min_count=min_c,
                                   vector_size=size,
                                   window=window,
                                   epochs=iterations,
                                   ns_exponent=ns_exponent)

    if is_debug:
        print("# products in the space: {}".format(len(model.wv.index_to_key)))

    return model.wv
