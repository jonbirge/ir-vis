import os
import urllib.request
import zipfile
import shutil

def setup():
    print("⬇️  Downloading PlotNeuralNet core files...")
    url = "https://github.com/HarisIqbal88/PlotNeuralNet/archive/master.zip"
    file_name = "pnn.zip"
    
    try:
        urllib.request.urlretrieve(url, file_name)
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return

    # Extract
    with zipfile.ZipFile(file_name, 'r') as zip_ref:
        zip_ref.extractall(".")
    
    # Move 'pycore' and 'layers' to root for easy import
    source_dir = "PlotNeuralNet-master"
    target_items = ["pycore", "layers"]
    
    for item in target_items:
        src = os.path.join(source_dir, item)
        if os.path.exists(item):
            shutil.rmtree(item)
        shutil.move(src, ".")
    
    # Create init file to allow imports
    with open("pycore/__init__.py", "w") as f: f.write("")

    # Cleanup
    shutil.rmtree(source_dir)
    os.remove(file_name)
    print("✅ Setup complete. 'pycore/' and 'layers/' are ready.")

if __name__ == "__main__":
    setup()
    