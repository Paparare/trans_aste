import os
import shutil


def extract_results(source_dir, target_dir):
    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Walk through the directory
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file == 'result.txt':
                # Path of result.txt in its current location
                full_file_path = os.path.join(root, file)
                # Path where the result.txt will be copied to
                # This uses the subdirectory path within source_dir to avoid filename conflicts
                target_file_path = os.path.join(target_dir, os.path.relpath(full_file_path, start=source_dir))
                # Ensure the subdirectory in the target exists
                os.makedirs(os.path.dirname(target_file_path), exist_ok=True)
                # Copy file
                shutil.copy(full_file_path, target_file_path)
                print(f"Copied: {full_file_path} to {target_file_path}")


# Use the function
source_directory = 'saved_models'
target_directory = 'extracted_results'
extract_results(source_directory, target_directory)