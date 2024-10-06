## Download and use C4 as a local dataset
C4 dataset may not be compatible with mirror sites, e.g., [HF-Mirror](https://hf-mirror.com/).

Thus, we provide tutorials for downloading and using it as follows.

### Step 1 Prepare the download path
Prepare the download path, e.g., `/root/c4`. 

### Step 2 Download C4 in [HF-Mirror](https://hf-mirror.com/)
Please modify `cache-dir` in the script according to your specific download path.
```bash
pip install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --repo-type dataset  --cache-dir /root/dataDisk/c4 --resume-download allenai/c4 --include "en/c4-validation*" --local-dir-use-symlinks False
```
If you get an error, it may be an environment variable problem, try this
```bash
# linux or mac
echo "export PATH=\"`python3 -m site --user-base`/bin:\$PATH\"" >> ~/.bashrc
source ~/.bashrc
```

### Step 2 Use C4 for pre-training
Please replace the C4 dataset loading code in `torchrun_main.py` accordingly.

Since the validation set and training dataset of the downloaded C4 dataset are in the same folder, we need to divide them manually.

#### Validation dataset
```python
val_data = datasets.load_dataset("c4", "en", split="validation", streaming=True, trust_remote_code=True) #DGX
```
Replace with the following (Please modify `/root/c4/`)
```python
# Set the path to the local dataset 
validation_data_dir = "/root/c4/datasets--allenai--c4/snapshots/1588ec454efa1a09f29cd18ddd04fe05fc8653a2/en"
# List all files in the directory
all_files = os.listdir(validation_data_dir)
# Filter the required files
validation_filtered_files = [f for f in all_files if f.startswith("c4-validation")]
# Load the specified files
val_data = datasets.load_dataset("json",
                                 data_files=[os.path.join(validation_data_dir, f) for f in validation_filtered_files],
                                 split='train', streaming=True) 
```

#### Training dataset
```python
data = datasets.load_dataset("allenai/c4", "en", split="train", streaming=True)
```
Replace with the following (Please modify `/root/c4/`)
```python
# Set the path to the local dataset
train_data_dir = "/root/c4/datasets--allenai--c4/snapshots/1588ec454efa1a09f29cd18ddd04fe05fc8653a2/en"
# List all files in the directory
all_files = os.listdir(train_data_dir)
# Filter the required files
train_filtered_files = [f for f in all_files if f.startswith("c4-train")]
# Load the specified files
data = datasets.load_dataset("json", data_files=[os.path.join(train_data_dir, f) for f in train_filtered_files],
                             split='train', streaming=True)
```