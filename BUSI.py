import os
from PIL import Image
import shutil
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms


# Define custom dataset class
class BUSI(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')  # 'L' mode for grayscale
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Function to prepare and split dataset into train/val
def prepare_dataset(root_dir, output_dir, val_size=0.2):
    image_paths = []
    labels = []
    label_mapping = {'benign': 0, 'malignant': 1, 'normal': 2}
    
    # Create output train and val directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Walk through each class folder (benign, malignant, normal)
    for label_name in label_mapping.keys():
        label_dir = os.path.join(root_dir, label_name)
        for file_name in os.listdir(label_dir):
            if '_mask' not in file_name:  # Exclude masked images
                image_path = os.path.join(label_dir, file_name)
                image_paths.append(image_path)
                labels.append(label_mapping[label_name])

    # Split dataset into train and validation
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=val_size, stratify=labels)

    # Move files to appropriate folders
    for path, label in zip(train_paths, train_labels):
        label_name = [k for k, v in label_mapping.items() if v == label][0]
        label_train_dir = os.path.join(train_dir, label_name)
        os.makedirs(label_train_dir, exist_ok=True)
        shutil.copy(path, label_train_dir)

    for path, label in zip(val_paths, val_labels):
        label_name = [k for k, v in label_mapping.items() if v == label][0]
        label_val_dir = os.path.join(val_dir, label_name)
        os.makedirs(label_val_dir, exist_ok=True)
        shutil.copy(path, label_val_dir)

    print("Train and validation datasets created successfully.")

# Define your transformations (similar to how you would for MNIST)
#transform = transforms.Compose([
 #   transforms.Resize((28, 28)),  # Resize to match MNIST input size if needed
 #  transforms.ToTensor(),
 #  transforms.Normalize((0.5,), (0.5,))  # Normalize (optional)
#])

#root_dir = '/Users/sarazatezalo/Documents/EPFL/semestral project/Dataset_BUSI_with_GT/'
#output_dir = '/Users/sarazatezalo/Documents/EPFL/semestral project/data/BUSI/'
#prepare_dataset(root_dir, output_dir) # TO BE RUN ONLY ONCE

