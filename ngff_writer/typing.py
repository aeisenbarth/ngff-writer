from typing import Dict, List, Literal, Tuple, Union

DimensionAxisType = Literal["t", "c", "z", "y", "x"]
DimensionSeparatorType = Literal["/", "."]
JsonType = Union[
    Dict[str, "JsonType"], List["JsonType"], Tuple["JsonType"], str, int, float, bool, None,
]
ZarrModeType = Literal["r", "r+", "a", "w", "w-"]
