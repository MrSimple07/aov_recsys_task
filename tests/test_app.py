import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.data import clean_transactions
from fake_data import generate
from src.training import train


APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture(scope="module")
def art(tmp_path_factory):
    out = tmp_path_factory.mktemp("app_art")
    train(clean_transactions(generate(n_items=120, n_cats=10, n_invoices=3000, seed=2)),
          out, top_k=100, fx=16100.0)
    return str(out)


def test_app_runs_and_simulates(art):
    os.environ["AOV_ARTIFACTS"] = art
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception
    assert any("so'm" in str(m.value) for m in at.metric)          # headline metrics
    assert len(at.dataframe) >= 3                                  # models + revenue check + cart tables
    table = at.dataframe[0].value
    assert len(table) == 6                                         # every model is listed
    assert {"AOV (so'm)", "Change (%)", "Accept (%)", "Completed (%)"} <= set(table.columns)
    assert not {"± (%)", "Real revenue change (%)"} & set(table.columns)   # revenue is not in the main table
    assert [tab.label for tab in at.tabs][0].strip().endswith("Compare models")


def test_app_without_models(tmp_path):
    os.environ["AOV_ARTIFACTS"] = str(tmp_path / "missing")
    at = AppTest.from_file(APP).run()
    assert not at.exception and at.error
