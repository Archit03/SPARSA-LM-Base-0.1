import os
import hashlib

def calculate_hash(file_path):
    """Calculate MD5 hash of a file."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as file:
        while chunk := file.read(8192):  # Read file in chunks to handle large files
            hasher.update(chunk)
    return hasher.hexdigest()

def remove_duplicates(directory):
    """Scan a directory for duplicate files and remove them."""
    hashes = {}  # Store file hashes and paths
    duplicates = []  # Store duplicate file paths

    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_hash = calculate_hash(file_path)

            if file_hash in hashes:
                print(f"Duplicate found: {file_path}")
                duplicates.append(file_path)  # Add duplicate to the list
            else:
                hashes[file_hash] = file_path

    # Remove duplicate files
    for duplicate in duplicates:
        try:
            os.remove(duplicate)
            print(f"Deleted: {duplicate}")
        except Exception as e:
            print(f"Error deleting {duplicate}: {e}")

    print(f"Completed. Total duplicates removed: {len(duplicates)}")

# Specify the directory to scan

directory_to_scan = r"C:\Users\ASUS\Desktop\PreProcessed\cleaned"
remove_duplicates(directory_to_scan)
 