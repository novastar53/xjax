import os
import requests
import zipfile

def download_glove(url, output_dir="glove"):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    glove_zip = os.path.join(output_dir, "glove.6B.zip")
    glove_file = os.path.join(output_dir, "glove.6B.100d.txt")

    # Check if the file already exists
    if os.path.exists(glove_file):
        print(f"{glove_file} already exists. Skipping download.")
        return glove_file

    print(f"Downloading GloVe embeddings from {url}...")
    
    # Download the zip file
    response = requests.get(url, stream=True)
    with open(glove_zip, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print("Download completed. Extracting the GloVe embeddings...")

    # Extract the specific 100D file from the zip
    with zipfile.ZipFile(glove_zip, "r") as zip_ref:
        zip_ref.extract("glove.6B.100d.txt", output_dir)

    print("Extraction completed. Cleaning up...")
    os.remove(glove_zip)  # Optional: Delete the zip file

    print(f"GloVe embeddings are available at: {glove_file}")
    return glove_file

if __name__ == "__main__":
    glove_url = "http://nlp.stanford.edu/data/glove.6B.zip"
    download_glove(glove_url)

