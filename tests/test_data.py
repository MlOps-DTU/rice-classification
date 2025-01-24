import os

import pytest
from omegaconf import OmegaConf
from torch.utils.data import Subset

from src.rice_classification.data import get_rice_pictures


@pytest.fixture
def mock_cfg():
    cfg = OmegaConf.create(
        {
            "parameters": {
                "height": 50,
                "width": 50,
                "test_size": 0.2,
                "raw_dir": "data/raw_unzipped",
                "zip_file_path": "data/raw/Arborio.zip",
                "processed_dir": "data/processed",
            }
        }
    )
    return cfg


def test_get_rice_pictures(mock_cfg):
    """
    Test the `get_rice_pictures` function for loading preprocessed datasets.
    """
    loaded_train, loaded_test = get_rice_pictures(mock_cfg)

    assert isinstance(loaded_train, Subset), "Loaded train dataset is not a Subset."
    assert isinstance(loaded_test, Subset), "Loaded test dataset is not a Subset."

    assert len(loaded_train) == 59754, "Incorrect train dataset size in get_rice_pictures."
    assert len(loaded_test) == 14939, "Incorrect test dataset size in get_rice_pictures."


if __name__ == "__main__":
    pytest.main()

"""
COVERAGE REPORT

>> coverage run -m pytest tests/test_data.py
C:/Users/USUARIO/AppData/Local/Programs/Python/Python312/Lib/site-packages/pytest_asyncio/plugin.py:207: PytestDeprecationWarning: The configuration option "asyncio_default_fixture_loop_scope" is unset.
The event loop scope for asynchronous fixtures will default to the fixture caching scope. Future versions of pytest-asyncio will default the loop scope for asynchronous fixtures to function scope. Set the default fixture loop scope explicitly in order to avoid unexpected behavior in the future. Valid fixture loop scopes are: "function", "class", "module", "package", "session"

  warnings.warn(PytestDeprecationWarning(_DEFAULT_FIXTURE_LOOP_SCOPE_UNSET))
========================== test session starts ===========================
platform win32 -- Python 3.12.0, pytest-8.3.4, pluggy-1.5.0
rootdir: C:/Users/USUARIO/Documents/DTU/MLOps7/rice-classification
configfile: pyproject.toml
plugins: anyio-4.3.0, hydra-core-1.3.2, asyncio-0.25.2
asyncio: mode=Mode.STRICT, asyncio_default_fixture_loop_scope=None
collected 1 item

tests/test_data.py .                                                [100%]

============================ warnings summary ============================
../../../../AppData/Roaming/Python/Python312/site-packages/dateutil/tz/tz.py:37
  C:/Users/USUARIO/AppData/Roaming/Python/Python312/site-packages/dateutil/tz/tz.py:37: DeprecationWarning: datetime.datetime.utcfromtimestamp() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.fromtimestamp(timestamp, datetime.UTC).
    EPOCH = datetime.datetime.utcfromtimestamp(0)

<frozen importlib._bootstrap>:488
  <frozen importlib._bootstrap>:488: DeprecationWarning: Type google._upb._message.MessageMapContainer uses PyType_Spec with a metaclass that has custom tp_new. This is deprecated and will no longer be allowed in Python 3.14.

<frozen importlib._bootstrap>:488
  <frozen importlib._bootstrap>:488: DeprecationWarning: Type google._upb._message.ScalarMapContainer uses PyType_Spec with a metaclass that has custom tp_new. This is deprecated and will no longer be allowed in Python 3.14.

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
===================== 1 passed, 3 warnings in 13.78s =====================


>> coverage report
Name                                  Stmts   Miss  Cover
---------------------------------------------------------
src/rice_classification/__init__.py       0      0   100%
src/rice_classification/data.py         110     79    28%
---------------------------------------------------------
TOTAL                                   110     79    28%
"""
