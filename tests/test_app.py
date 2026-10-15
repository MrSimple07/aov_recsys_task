import os
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from src.synthetic import generate


def test_app_runs_end_to_end(tmp_path):
    csv = tmp_path / "retail.csv"
    generate(n_items=120, n_customers=300, n_invoices=3000, seed=2).to_csv(csv, index=False)
    os.environ["AOV_DATA"] = str(csv)
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=180).run()
    assert not at.exception, at.exception
    at.multiselect[0].select(at.multiselect[0].options[0]).run()   # build a cart
    assert not at.exception, at.exception
    at.button[0].click().run()                                      # run simulation
    assert not at.exception, at.exception
    os.environ.pop("AOV_DATA")
