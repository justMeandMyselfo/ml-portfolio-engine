import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

import pytest

from mlportfolio.data.loaders import make_synthetic_market_data


@pytest.fixture(scope="session")
def synthetic_data():
    return make_synthetic_market_data(n_days=1200, seed=7)
