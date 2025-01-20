import torch
import gcsfs

# Define the GCS file path
bucket_path = "gs://emils_mlops_data_bucket/data/processed/train.pt"

# Create a GCS filesystem object
fs = gcsfs.GCSFileSystem()

# Open the file and load it with PyTorch
with fs.open(bucket_path, 'rb') as f:
    data = torch.load(f)

# Print or use the loaded data
print(data.shape)