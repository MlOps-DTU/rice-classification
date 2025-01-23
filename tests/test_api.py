import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
import torch
from src.rice_classification.api import app
import torchvision.transforms as transforms


# Mocking the model and transform
@pytest.fixture
def mock_model_and_transform():
    # Mock the model
    mock_model = MagicMock()
    
    # Mock the transform function
    mock_transform = MagicMock()
    
    # Setup mock output for the model (mocked as a tensor)
    mock_output = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])  # Mocked output for 5 classes
    mock_model.return_value = mock_output
    
    # Setup mock transform to return a fake tensor (mocked input image)
    mock_transform.return_value = torch.tensor([0.5]).unsqueeze(0)
    transform = transforms.Compose(
        [
            transforms.Resize((50, 50)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    # Attach the mocked model and transform to the app state
    app.state.model = mock_model
    app.state.transform = transform

    yield mock_model, mock_transform  # Return mocked objects for use in tests

    # Cleanup mocks (if needed)
    del app.state.model
    del app.state.transform

@pytest.fixture
def client():
    # Create a TestClient for the FastAPI app
    return TestClient(app)

def test_root(client):
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == ["Hello and welcome to the rice classification app!"]

def test_predict_no_file(client):
    """Test the /predict endpoint without a file."""
    response = client.post("/predict/")
    assert response.status_code == 422  # Unprocessable Entity

def test_predict_endpoint(client, mock_model_and_transform):
    """Test the /predict/ endpoint."""

    # Provide a sample test image (this could be any valid image path)
    test_image_path = "tests/sample_image.jpg"  
    
    # Make a POST request to the /predict/ endpoint with the test image
    with open(test_image_path, "rb") as img:
        files = {"data": img}
        response = client.post("/predict/", files=files)

    # Check that the response status code is 200 (OK)
    assert response.status_code == 200
    json_response = response.json()
    assert "Prediction of rice category" in json_response
    assert "Prediction int: " in json_response





"""

>> pytest -v tests/test_api.py  
C:/Users/USUARIO/AppData/Local/Programs/Python/Python312/Lib/site-packages/pytest_asyncio/plugin.py:207: PytestDeprecationWarning: The configuration option "asyncio_default_fixture_loop_scope" is unset.        
The event loop scope for asynchronous fixtures will default to the fixture caching scope. Future versions of pytest-asyncio will default the loop scope for asynchronous fixtures to function scope. Set the default fixture loop scope explicitly in order to avoid unexpected behavior in the future. Valid fixture loop scopes are: "function", "class", "module", "package", "session"

  warnings.warn(PytestDeprecationWarning(_DEFAULT_FIXTURE_LOOP_SCOPE_UNSET))
======================== test session starts ========================
platform win32 -- Python 3.12.0, pytest-8.3.4, pluggy-1.5.0 -- C:/Users/USUARIO/AppData/Local/Programs/Python/Python312/python.exe
cachedir: .pytest_cache
rootdir: C:/Users/USUARIO/Documents/DTU/MLOps5/rice-classification    
configfile: pyproject.toml
plugins: anyio-4.3.0, hydra-core-1.3.2, asyncio-0.25.2
asyncio: mode=Mode.STRICT, asyncio_default_fixture_loop_scope=None    
collected 3 items                                                    

tests/test_api.py::test_root PASSED                            [ 33%]
tests/test_api.py::test_predict_no_file PASSED                 [ 66%]
tests/test_api.py::test_predict_endpoint PASSED                [100%]

========================= warnings summary ========================== 
<frozen importlib._bootstrap>:488
  <frozen importlib._bootstrap>:488: DeprecationWarning: Type google._upb._message.MessageMapContainer uses PyType_Spec with a metaclass that has custom tp_new. This is deprecated and will no longer be allowed in Python 3.14.

<frozen importlib._bootstrap>:488
  <frozen importlib._bootstrap>:488: DeprecationWarning: Type google._upb._message.ScalarMapContainer uses PyType_Spec with a metaclass that has custom tp_new. This is deprecated and will no longer be allowed in Python 3.14.

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=================== 3 passed, 2 warnings in 7.72s =================== 

"""
