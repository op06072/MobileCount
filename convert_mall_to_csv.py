"""Convert MALL dataset MAT files to CSV format for faster loading"""

import scipy.io as sio
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def mat_to_csv(mat_file_path, csv_file_path):
    """Convert a single MAT file to CSV"""
    try:
        # Load MAT file
        mat_data = sio.loadmat(str(mat_file_path))

        if "map" in mat_data:
            density_map = mat_data["map"]

            # Save as CSV
            df = pd.DataFrame(density_map)
            df.to_csv(csv_file_path, index=False, header=False)
            return True
        else:
            print(f"Warning: 'map' key not found in {mat_file_path}")
            return False
    except Exception as e:
        print(f"Error converting {mat_file_path}: {e}")
        return False


def convert_mall_dataset(base_path):
    """Convert all MALL MAT files to CSV (both train and test)"""
    base_path = Path(base_path)

    total_converted = 0
    total_failed = 0

    # Process both train and test folders
    for split in ["train", "test"]:
        print(f"\n{'=' * 60}")
        print(f"Processing {split.upper()} split...")
        print(f"{'=' * 60}")

        # Find all MAT files in split/den
        den_folder = base_path / split / "den"

        if not den_folder.exists():
            print(f"Warning: {den_folder} does not exist, skipping...")
            continue

        mat_files = list(den_folder.glob("*.mat"))

        if not mat_files:
            print(f"No MAT files found in {den_folder}")
            continue

        print(f"Found {len(mat_files)} MAT files in {split}/den...")

        # Create CSV output directory
        csv_dir = base_path / split / "den_csv"
        csv_dir.mkdir(exist_ok=True)

        # Convert each file
        converted = 0
        failed = 0

        for mat_file in tqdm(mat_files, desc=f"Converting {split}"):
            csv_file = csv_dir / (mat_file.stem + ".csv")

            if mat_to_csv(mat_file, csv_file):
                converted += 1
            else:
                failed += 1

        print(f"\n{split.upper()} Results:")
        print(f"  - Converted: {converted} files")
        print(f"  - Failed: {failed} files")
        print(f"  - Output: {csv_dir}")

        total_converted += converted
        total_failed += failed

    print(f"\n{'=' * 60}")
    print(f"✓ TOTAL CONVERSION COMPLETE!")
    print(f"{'=' * 60}")
    print(f"  - Total Converted: {total_converted} files")
    print(f"  - Total Failed: {total_failed} files")

    # Update loader to use CSV
    print(f"\nTo use CSV files, update MALL__gt_name_folder to 'den_csv'")
    print(f"and MALL__gt_format to '.csv' in your config.")


if __name__ == "__main__":
    # MALL dataset path
    mall_path = Path("exp/data/Mall")

    if not mall_path.exists():
        print(f"Error: MALL dataset not found at {mall_path}")
        print("Please update the path in this script.")
    else:
        convert_mall_dataset(mall_path)

        # Also create a CSV loader function
        print("\n" + "=" * 60)
        print("CSV Loader Function:")
        print("=" * 60)
        print("""
def load_gt_csv(filename):
    '''Load density map from CSV'''
    density_map = pd.read_csv(filename, header=None).values.astype(np.float32)
    return density_map
        
# In MALL.py, update load_gt():
elif Path(filename).suffix == ".csv":
    density_map = pd.read_csv(filename, header=None).values.astype(np.float32)
        """)
