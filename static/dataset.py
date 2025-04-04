import kagglehub

# Download latest version
path = kagglehub.dataset_download("mhskjelvareid/dagm-2007-competition-dataset-optical-inspection")

print("Path to dataset files:", path)