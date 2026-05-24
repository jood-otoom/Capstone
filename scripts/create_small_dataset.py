import os
import shutil
import random
from pathlib import Path

def create_subset_dataset(source_dir, target_dir, num_samples_per_split):
    """Create a smaller subset dataset for faster training."""

    # Create target directories
    for split in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(target_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(target_dir, split, 'labels'), exist_ok=True)

    for split in ['train', 'valid', 'test']:
        print(f"Processing {split} split...")

        # Get all image files
        img_source = Path(source_dir) / split / 'images'
        lbl_source = Path(source_dir) / split / 'labels'
        img_target = Path(target_dir) / split / 'images'
        lbl_target = Path(target_dir) / split / 'labels'

        if not img_source.exists():
            continue

        # Get all image files
        if split == 'test':
            # For test, recursively find all images in subdirectories
            all_images = []
            for root, dirs, files in os.walk(img_source):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        all_images.append(Path(root) / file)
        else:
            all_images = list(img_source.glob('*.jpg')) + list(img_source.glob('*.png')) + list(img_source.glob('*.jpeg'))

        # Randomly sample
        num_samples = min(num_samples_per_split[split], len(all_images))
        sampled_images = random.sample(all_images, num_samples)

        print(f"  Selected {len(sampled_images)} images out of {len(all_images)}")

        # Copy sampled images and their labels
        for img_path in sampled_images:
            # Copy image
            rel_path = img_path.relative_to(img_source)
            target_img_path = img_target / rel_path
            target_img_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, target_img_path)

            # Copy corresponding label (if exists)
            img_name = img_path.stem
            label_files = list(lbl_source.glob(f'{img_name}.txt'))
            if label_files:
                shutil.copy2(label_files[0], lbl_target / f'{img_name}.txt')

if __name__ == '__main__':
    # Create a smaller dataset for faster training
    source_dir = 'processed_dataset'
    target_dir = 'processed_dataset_small'

    # Sample sizes: adjust these for your desired training time
    num_samples = {
        'train': 1500,  # ~1.5K training images
        'valid': 300,   # ~300 validation images
        'test': 200     # ~200 test images
    }

    print("Creating smaller dataset for faster training...")
    create_subset_dataset(source_dir, target_dir, num_samples)

    # Create data.yaml for the small dataset
    data_yaml_content = f"""train: ../{target_dir}/train/images
val: ../{target_dir}/valid/images
test: ../{target_dir}/test/images

nc: 1
names:
  ["accident"]
"""

    with open(f'{target_dir}/data.yaml', 'w') as f:
        f.write(data_yaml_content)

    print(f"\nSmall dataset created in '{target_dir}'")
    print("Ready for fast training! (~30-60 minutes with GPU)")