from PIL.Image import Image
from typing import TypedDict, List


class DataDict(TypedDict, total=False):
    fname: str
    datas: List[Image]