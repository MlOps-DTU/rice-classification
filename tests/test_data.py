import os
import pytest
import torch
from torchvision import datasets
from torch.utils.data import Subset
from src.rice_classification.data import main, get_rice_pictures 

@pytest.fixture
def setup_data(tmp_path):
    """
    Set up a basic directory structure with mock images for testing.
    """
    raw_dir = tmp_path / "raw_data"
    raw_dir.mkdir()

    processed_dir = tmp_path / "processed_data"
    processed_dir.mkdir()

    class_0 = raw_dir / "class_0"
    class_1 = raw_dir / "class_1"
    class_0.mkdir()
    class_1.mkdir()

    for i in range(5):
        (class_0 / f"image_{i}.png").touch() if i % 2 == 0 else (class_1 / f"image_{i}.png").touch()

    return {"raw_dir": raw_dir, "processed_dir": processed_dir}

def test_main(setup_data):
    """
    Test the `main` function for processing and saving the datasets.
    """
    class MockCfg:
        class Parameters:
            height = 128
            width = 128
            raw_dir = str(setup_data["raw_dir"])
            processed_dir = str(setup_data["processed_dir"])
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
    print(len(loaded_train), len(loaded_test))

    assert isinstance(loaded_train, Subset), "Loaded train dataset is not a Subset."
    assert isinstance(loaded_test, Subset), "Loaded test dataset is not a Subset."

    assert len(loaded_train) == 60000, "Incorrect train dataset size in get_rice_pictures."
    assert len(loaded_test) == 15000, "Incorrect test dataset size in get_rice_pictures."



if __name__ == "__main__":
    pytest.main()



"""
COVERAGE REPORT

>> coverage run -m pytest test_data.py 
======================================================= test session starts ========================================================
platform win32 -- Python 3.12.0, pytest-8.3.4, pluggy-1.5.0
rootdir: C:/Users/USUARIO/Documents/DTU/MLOps2/rice-classification
configfile: pyproject.toml
plugins: anyio-4.8.0, hydra-core-1.3.2
collected 2 items                                                                                                                   

test_data.py ..                                                                                                               [100%]

========================================================= warnings summary ========================================================= 
tests/test_data.py::test_main
  C:/Users/USUARIO/Documents/DTU/MLOps2/rice-classification/tests/test_data.py:56: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
    train_set = torch.load(train_file)

tests/test_data.py::test_main
  C:/Users/USUARIO/Documents/DTU/MLOps2/rice-classification/tests/test_data.py:57: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
    test_set = torch.load(test_file)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
================================================== 2 passed, 2 warnings in 6.57s =================================================== 


>> coverage report   
Name                                                                                            Stmts   Miss  Cover
-------------------------------------------------------------------------------------------------------------------
C:/Users/USUARIO/Documents/DTU/MLOps2/rice-classification/src/rice_classification/__init__.py       0      0   100%
C:/Users/USUARIO/Documents/DTU/MLOps2/rice-classification/src/rice_classification/data.py          23      1    96%
__init__.py                                                                                         4      0   100%
test_data.py                                                                                       55      1    98%
-------------------------------------------------------------------------------------------------------------------
TOTAL                                                                                              82      2    98%

"""
