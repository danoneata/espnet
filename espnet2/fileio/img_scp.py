import collections.abc
from pathlib import Path
from typing import Union

import numpy as np
from typeguard import check_argument_types

from skimage import io
from PIL import Image

from espnet2.fileio.read_text import read_2column_text


class ImgScpReader(collections.abc.Mapping):
    """Reader class for a scp file of image files.

    Examples:
        key1 /some/path/a.jpg
        key2 /some/path/b.jpg
        key3 /some/path/c.jpg
        key4 /some/path/d.jpg
        ...

        >>> reader = ImgScpReader('img.scp')
        >>> array = reader['key1']

    """

    def __init__(self, fname: Union[Path, str]):
        assert check_argument_types()
        self.fname = Path(fname)
        self.data = read_2column_text(fname)

    def get_path(self, key):
        return self.data[key]

    def __getitem__(self, key) -> np.ndarray:
        p = self.data[key]
        return io.imread(p)
        # return Image.open(p)

    def __contains__(self, item):
        return item

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def keys(self):
        return self.data.keys()
