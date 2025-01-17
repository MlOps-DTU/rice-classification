import os
import pytest
import torch
from torchvision import datasets
from torch.utils.data import Subset
from src.rice_classification.data import main, get_rice_pictures 



@pytest.fixture
def setup_data():
    RAW_DIR = os.path.join("data", "raw")
    PROCESSED_DIR = os.path.join("data", "processed")
    return {"raw_dir": RAW_DIR, "processed_dir": PROCESSED_DIR}

def test_main(setup_data):
    """
    Test the `main` function for processing and saving the datasets.
    """
    # Mock configuration
    class MockCfg:
        class Parameters:
            height = 128
            width = 128
            raw_dir, processed_dir = setup_data["raw_dir"], setup_data["processed_dir"]
            test_size = 0.5

        parameters = Parameters()

    main(MockCfg)

    train_file = os.path.join(setup_data["processed_dir"], "train.pt")
    test_file = os.path.join(setup_data["processed_dir"], "test.pt")

    assert os.path.exists(train_file), "Train dataset file was not created."
    assert os.path.exists(test_file), "Test dataset file was not created."

    train_set = torch.load(train_file)
    test_set = torch.load(test_file)

    assert isinstance(train_set, Subset), "Train dataset is not a Subset."
    assert isinstance(test_set, Subset), "Test dataset is not a Subset."

    assert len(train_set) == 75000//2, "Incorrect train dataset size."
    assert len(test_set) == 75000//2, "Incorrect test dataset size."

def test_get_rice_pictures(setup_data):
    """
    Test the `get_rice_pictures` function for loading preprocessed datasets.
    """

    train_set = Subset(datasets.ImageFolder(setup_data["raw_dir"]), [0, 2])
    test_set = Subset(datasets.ImageFolder(setup_data["raw_dir"]), [1, 3])

    train_file = os.path.join(setup_data["processed_dir"], "train.pt")
    test_file = os.path.join(setup_data["processed_dir"], "test.pt")

    torch.save(train_set, train_file)
    torch.save(test_set, test_file)

    loaded_train, loaded_test = get_rice_pictures()

    assert isinstance(loaded_train, Subset), "Loaded train dataset is not a Subset."
    assert isinstance(loaded_test, Subset), "Loaded test dataset is not a Subset."

    assert len(loaded_train.dataset) == 75000, "Incorrect train dataset size in get_rice_pictures."
    assert len(loaded_test.dataset) == 75000, "Incorrect test dataset size in get_rice_pictures."

if __name__ == "__main__":
    pytest.main()


"""
COVERAGE REPORT

>> coverage run -m pytest tests/test_data.py 
========================================================== test session starts ===========================================================
platform win32 -- Python 3.12.0, pytest-8.3.4, pluggy-1.5.0
rootdir: C:/Users/USUARIO/Documents/DTU/MLOps2/rice-classification
configfile: pyproject.toml
plugins: anyio-4.8.0, hydra-core-1.3.2
collected 2 items                                                                                                                         

tests/test_data.py ..                                                                                                               [100%]

============================================================ warnings summary ============================================================ 
tests/test_data.py::test_main
  C:/Users/USUARIO/Documents/DTU/MLOps2/rice-classification/tests/test_data.py:38: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
    train_set = torch.load(train_file)

tests/test_data.py::test_main
  C:/Users/USUARIO/Documents/DTU/MLOps2/rice-classification/tests/test_data.py:39: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
    test_set = torch.load(test_file/

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
===================================================== 2 passed, 2 warnings in 10.39s ===================================================== 


>> coverage report
Name                                  Stmts   Miss  Cover
---------------------------------------------------------
src/rice_classification/__init__.py       0      0   100%
src/rice_classification/data.py          30      1    97%
---------------------------------------------------------
TOTAL                                    30      1    97%
"""
