import os
import sys

def rename_files_in_hierarchy(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if not filenames:
            continue
        folder_name = os.path.basename(dirpath)
        for i, filename in enumerate(filenames, start=1):
            old_path = os.path.join(dirpath, filename)
            name, ext = os.path.splitext(filename)
            new_name = f"{folder_name}_{i}{ext}"
            new_path = os.path.join(dirpath, new_name)
            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} -> {new_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <root_directory>")
        sys.exit(1)
    root_dir = sys.argv[1]
    rename_files_in_hierarchy(root_dir)