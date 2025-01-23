import os

import pytest
from omegaconf import OmegaConf
from rice_classification.data import get_rice_pictures
from torch.utils.data import Subset


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
============================ test session starts =============================
platform win32 -- Python 3.12.0, pytest-8.3.4, pluggy-1.5.0
rootdir: C:/Users/USUARIO/Documents/DTU/MLOps5/rice-classification
configfile: pyproject.toml
plugins: anyio-4.3.0, hydra-core-1.3.2, asyncio-0.25.2
asyncio: mode=Mode.STRICT, asyncio_default_fixture_loop_scope=None
collected 2 items

tests/test_data.py ..                                                   [100%]

============================== warnings summary ==============================
tests/test_data.py::test_main
  C:/Users/USUARIO/Documents/DTU/MLOps5/rice-classification/tests/test_data.py:39: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
    train_set = torch.load(train_file)

tests/test_data.py::test_main
  C:/Users/USUARIO/Documents/DTU/MLOps5/rice-classification/tests/test_data.py:40: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
    test_set = torch.load(test_file)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
======================= 2 passed, 2 warnings in 10.47s =======================


>> coverage report
Name                                  Stmts   Miss  Cover
---------------------------------------------------------
src/rice_classification/__init__.py       0      0   100%
src/rice_classification/data.py          58     21    64%
---------------------------------------------------------
TOTAL                                    58     21    64%
"""
