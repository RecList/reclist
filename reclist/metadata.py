import json
import os
import time
from enum import Enum
from abc import ABC, abstractmethod
from pathlib import Path
from functools import wraps
import pandas as pd
import numpy as np

class METADATA_STORE(Enum):
    LOCAL = 1
    S3 = 2


class MetaStore(ABC):

    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def write_file(self, path, data, is_json=False):
        pass


def metadata_store_factory(label) -> MetaStore:
    if label == METADATA_STORE.S3:
        return S3MetaStore
    else:
        return LocalMetaStore


class LocalMetaStore(MetaStore):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        return

    def write_file(self, path, data, is_json=False):
        if is_json:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        else:
            with open(path, "w") as f:
                f.write(data)

        return


class S3MetaStore(MetaStore):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        return

    def write_file(self, path, data, is_json=False):
        """
        We use s3fs to write to s3 - note: credentials are the default one
        locally stored in ~/.aws/credentials

        https://s3fs.readthedocs.io/en/latest/
        """
        import s3fs

        s3 = s3fs.S3FileSystem(anon=False)
        if is_json:
            with s3.open(path, 'wb') as f:
                f.write(json.dumps(data).encode('utf-8'))
        else:
            with s3.open(path, 'wb') as f:
                f.write(data)

        return
