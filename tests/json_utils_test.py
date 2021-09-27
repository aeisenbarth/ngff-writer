import pytest

from ngff_writer.json_utils import nested_dict_apply_fn_at_path


@pytest.fixture
def nested_dict():
    return {
        "a11": {"a21": "x", "a22": {}, "a23": {"a31": 1.0}},
        "b": {},
        "c": [],
        "d": "x",
        "e": 1.0,
        "f": None,
    }


def test_nested_dict_apply_fn_at_path(nested_dict):
    expected = [(nested_dict["a11"], "a23")]
    actual = []
    nested_dict_apply_fn_at_path(
        nested_dict,
        key_path=["a11", "a23"],
        fn=lambda node_dict, key: actual.append((node_dict, key)),
    )
    assert actual == expected
