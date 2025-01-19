# Use Google Cloud SDK image as the base for the build stage
FROM gcr.io/google-cloud-sdk/cloud-sdk:slim AS data

# Download only the data
RUN mkdir /data
RUN gsutil cp -r gs://emils_mlops_data_bucket/data/* /data/

# Main build stage
FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY src src/
COPY configs configs/
COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY README.md README.md
COPY pyproject.toml pyproject.toml

# Copy data from the data stage
COPY --from=data /data data/
RUN mkdir models
RUN mkdir -p reports/figures

RUN pip install -r requirements.txt --no-cache-dir --verbose
RUN pip install . --no-deps --no-cache-dir --verbose

ENTRYPOINT ["python", "-u", "src/rice_classification/train.py"]
