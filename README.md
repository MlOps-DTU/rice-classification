# Project Description

## Project Goal
The goal of this project is to apply the different open-source frameworks we have learned about during the course on a small project. To achieve this, our project will focus on the machine learning task of classifying diverse types of rice grains based on their images. For this purpose, we will utilize machine learning techniques to build a reliable classification model capable of distinguishing between various rice types.

## Framework
For this project, we will use the additional libary torchvision, and specificily the transform sub-library. This package is a framework commonly used for image processing. It includes functionality to reduce image sizes, normalize the pixel values and many other image augmentations that can be relevant when doing an image classification task. We plan to leverage this framework to preprocess and manipulate the dataset using various techniques, which will enhance the accuracy and robustness of our classification model. Some of the techniques we are going to implement are image resizing (to ensure compatibility with the models), normalization (scaling pixel values to a range of [0,1] for improved convergence during training), as well as other techniques we find useful during our model and training development.

## Dataset
The dataset provided is available on Kaggle https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset. The dataset consists of 75k images, and 5 types of rice grains where in each folder, there are around 15k images of that grain rice. The type of rice grains is Arborio, Basmati, Ipsala, Jasmine and Karacadag. In the kaggle dataset, there is also a second dataset consisting of features like shape, color and other features for a machine learning model, but we will not focus on that dataset due to the fact that we want to develop an image classifying model.

## Models
For this project, we are initially going to implement a CNN network for image classification, like the neural networks implemented during the exercises of the course. We may use ResNet, or another neural network we find interesting to work with.

## Project structure
The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```
Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
