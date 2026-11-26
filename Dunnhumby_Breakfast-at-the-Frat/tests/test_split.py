import pandas as pd
import pytest

from retail_multilevel.split import FIRST_TEST_WEEK, is_test, model_sample


def _panel():
    df = pd.DataFrame({"WEEK": list(range(8)) * 2, "IN_MODEL": True, "ID": range(16)})
    df.loc[[3, 15], "IN_MODEL"] = False
    return df


def test_first_test_week_constant():
    assert FIRST_TEST_WEEK == 104


def test_is_test_mask():
    df = _panel()
    assert is_test(df, 5).tolist() == (df["WEEK"] >= 5).tolist()


def test_train_test_partition_disjoint_and_complete():
    df = _panel()
    tr = model_sample(df, "train", first_test_week=5)
    te = model_sample(df, "test", first_test_week=5)
    al = model_sample(df, "all", first_test_week=5)
    assert set(tr["ID"]).isdisjoint(te["ID"])
    assert set(tr["ID"]) | set(te["ID"]) == set(al["ID"])
    assert (tr["WEEK"] < 5).all() and (te["WEEK"] >= 5).all()


def test_excludes_not_in_model():
    al = model_sample(_panel(), "all", first_test_week=5)
    assert len(al) == 14
    assert al["IN_MODEL"].all()
    assert not {3, 15} & set(al["ID"])


def test_invalid_part_raises():
    with pytest.raises(ValueError, match="part must be"):
        model_sample(_panel(), "validation")
