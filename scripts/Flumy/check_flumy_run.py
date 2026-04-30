import os
import argparse
import shutil
import numpy as np

def check_flumy_files(data_dir, move_bad_files=False):
    data_dir = os.path.abspath(data_dir)
    bad_dir = os.path.join(data_dir, "corrupted_files")

    print(f"Scanning directory for .npz files: {data_dir}")
    npz_files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
    
    if not npz_files:
        print("Error: No .npz files found in the specified directory.")
        return

    total_files = len(npz_files)
    valid_count = 0
    bad_files = []

    print("Checking files. This might take a moment for 10k+ files...")

    for filename in npz_files:
        filepath = os.path.join(data_dir, filename)
        is_bad = False
        reason = ""

        # 1. OS-level check for completely empty files (0 bytes)
        if os.path.getsize(filepath) == 0:
            is_bad = True
            reason = "0 bytes (Empty file at OS level)"
        else:
            # 2. Numpy-level check for corrupted or empty arrays
            try:
                with np.load(filepath) as data:
                    if 'facies' not in data:
                        is_bad = True
                        reason = "Missing 'facies' array key"
                    elif data['facies'].size == 0 or data['facies'].shape[0] == 0:
                        is_bad = True
                        reason = "Empty 'facies' array (Size 0 inside .npz)"
            except Exception as e:
                is_bad = True
                # Truncating the exception message so it doesn't flood your terminal
                reason = f"Corrupted/Unreadable by numpy ({str(e)[:40]}...)"

        if is_bad:
            bad_files.append((filename, reason))
        else:
            valid_count += 1

    # Report findings
    print("\n" + "=" * 50)
    print(" SCAN RESULTS")
    print("=" * 50)
    print(f"Total files scanned: {total_files}")
    print(f"Valid files:         {valid_count}")
    print(f"Bad files found:     {len(bad_files)}")
    print("=" * 50)

    if bad_files:
        print("\nList of bad files:")
        for bf, reason in bad_files[:20]: # Only print the first 20 to avoid terminal spam
            print(f" - {bf}: {reason}")
        
        if len(bad_files) > 20:
            print(f" ... and {len(bad_files) - 20} more.")

        # Optional logic to quarantine the bad files
        if move_bad_files:
            os.makedirs(bad_dir, exist_ok=True)
            print(f"\nMoving {len(bad_files)} bad files to quarantine folder: {bad_dir}")
            for bf, _ in bad_files:
                src = os.path.join(data_dir, bf)
                dst = os.path.join(bad_dir, bf)
                try:
                    shutil.move(src, dst)
                except Exception as e:
                    print(f"Could not move {bf}: {e}")
            print("Move complete. Your dataset is now clean!")
        else:
            print("\nTip: Run this script again with the '--move_bad' flag to quarantine these files.")
    else:
        print("\nExcellent! No bad files found. You are good to go.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check Flumy .npz files for corruption or empty facies arrays.")
    parser.add_argument('--data_dir', type=str, default='.', help="Directory containing the .npz files")
    parser.add_argument('--move_bad', action='store_true', help="If set, automatically moves bad files to a 'corrupted_files' subfolder")
    
    args = parser.parse_args()
    check_flumy_files(args.data_dir, args.move_bad)