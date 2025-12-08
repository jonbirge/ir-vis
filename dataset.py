"""
Dataset module for IR-to-Color Image Translation

This module handles:
1. Downloading the COCO dataset (or other supported datasets)
2. Creating simulated IR/visible image pairs for training
3. Data augmentation and preprocessing

The key insight for training data generation:
- Real IR images capture thermal radiation, which correlates with but differs from
  visible red channel. The red channel approximation works because:
  - Many natural materials have similar relative reflectance in red and near-IR
  - Vegetation, in particular, shows this correlation (red edge effect)
  - For a first approximation, this gives reasonable training signal
  
- To simulate perspective/FOV differences, we take a random crop of the original
  image as the "IR" view, while using the full image as the color reference.
  This forces the network to learn robust feature matching across viewpoints.

Supported datasets:
- Cityscapes: European urban street scenes (requires free registration)
- COCO: Diverse outdoor scenes (auto-downloads)
"""

import os
import sys
import random
import urllib.request
import zipfile
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List, Union

import torch # type: ignore
from torch.utils.data import Dataset, DataLoader # type: ignore
import torchvision.transforms as T # type: ignore
import torchvision.transforms.functional as TF # type: ignore
from PIL import Image # type: ignore
import numpy as np # type: ignore

from config import Config, DataConfig


class CityscapesLoader:
    """
    Utility class to load and optionally download Cityscapes dataset.
    
    Cityscapes contains 5000 finely annotated images from 50 different cities,
    with high-quality 2048x1024 resolution images of urban street scenes.
    
    Automatic download requires:
    1. Free registration at https://www.cityscapes-dataset.com/
    2. Setting environment variables:
       - CITYSCAPES_USERNAME (your email)
       - CITYSCAPES_PASSWORD (your password)
    
    Alternatively, download manually:
    - leftImg8bit_trainvaltest.zip (11GB) from the website
    - Extract to data_root/cityscapes/
    """
    
    DOWNLOAD_INSTRUCTIONS = """
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                    CITYSCAPES DATASET SETUP                         ║
    ╠══════════════════════════════════════════════════════════════════════╣
    ║ Cityscapes requires free registration to download.                  ║
    ║                                                                      ║
    ║ OPTION 1: Automatic Download (Recommended)                          ║
    ║ ──────────────────────────────────────────                          ║
    ║ 1. Register at: https://www.cityscapes-dataset.com/register/        ║
    ║ 2. Set environment variables:                                       ║
    ║    Windows (PowerShell):                                            ║
    ║      $env:CITYSCAPES_USERNAME = "your_email@example.com"            ║
    ║      $env:CITYSCAPES_PASSWORD = "your_password"                     ║
    ║    Windows (CMD):                                                   ║
    ║      set CITYSCAPES_USERNAME=your_email@example.com                 ║
    ║      set CITYSCAPES_PASSWORD=your_password                          ║
    ║    Linux/Mac:                                                       ║
    ║      export CITYSCAPES_USERNAME="your_email@example.com"            ║
    ║      export CITYSCAPES_PASSWORD="your_password"                     ║
    ║ 3. Run the training script again                                    ║
    ║                                                                      ║
    ║ OPTION 2: Manual Download                                           ║
    ║ ─────────────────────────                                           ║
    ║ 1. Register at: https://www.cityscapes-dataset.com/register/        ║
    ║ 2. Login and go to: https://www.cityscapes-dataset.com/downloads/  ║
    ║ 3. Download: leftImg8bit_trainvaltest.zip (11GB)                    ║
    ║ 4. Extract to: {data_root}/cityscapes/                              ║
    ║                                                                      ║
    ║ Expected structure after extraction:                                 ║
    ║   {data_root}/cityscapes/leftImg8bit/train/aachen/...               ║
    ║   {data_root}/cityscapes/leftImg8bit/val/frankfurt/...              ║
    ║                                                                      ║
    ║ OPTION 3: Use COCO instead (auto-downloads, no registration)        ║
    ║ ────────────────────────────────────────────────────────            ║
    ║   python train.py --dataset coco                                    ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """
    
    def __init__(self, data_root: str):
        """
        Initialize the Cityscapes loader.
        
        Args:
            data_root: Root directory where data is stored
        """
        self.data_root = Path(data_root)
        # Support both extraction folder names:
        # - "cityscapes" (expected standard location)
        # - "leftImg8bit_trainvaltest" (default extraction folder name)
        self.cityscapes_root = self.data_root / "cityscapes"
        if not (self.cityscapes_root / "leftImg8bit").exists():
            # Try alternative extraction folder name
            self.cityscapes_root = self.data_root / "leftImg8bit_trainvaltest"
        self.leftimg_root = self.cityscapes_root / "leftImg8bit"
        
    def check_exists(self) -> bool:
        """Check if Cityscapes dataset exists."""
        train_dir = self.leftimg_root / "train"
        val_dir = self.leftimg_root / "val"
        
        if not train_dir.exists() or not val_dir.exists():
            return False
        
        # Check that there are actually images
        train_images = list(train_dir.glob("*/*_leftImg8bit.png"))
        return len(train_images) > 0
    
    def download(self) -> bool:
        """
        Download Cityscapes dataset using cityscapesscripts.
        
        Requires CITYSCAPES_USERNAME and CITYSCAPES_PASSWORD environment variables.
        
        Returns:
            True if download successful, False otherwise
        """
        username = os.environ.get('CITYSCAPES_USERNAME')
        password = os.environ.get('CITYSCAPES_PASSWORD')
        
        if not username or not password:
            print("\nCityscapes credentials not found in environment variables.")
            print(self.DOWNLOAD_INSTRUCTIONS.format(data_root=self.data_root))
            return False
        
        try:
            # Try to import cityscapesscripts
            from cityscapesscripts.download import downloader # type: ignore
        except ImportError:
            print("\nInstalling cityscapesscripts package for automatic download...")
            import subprocess
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', 
                'cityscapesscripts', '--quiet'
            ])
            from cityscapesscripts.download import downloader # type: ignore
        
        # Create destination directory
        self.cityscapes_root.mkdir(parents=True, exist_ok=True)
        
        print(f"\nDownloading Cityscapes to {self.cityscapes_root}...")
        print("This may take a while (leftImg8bit is ~11GB)...")
        
        try:
            # The downloader expects to be run from the destination directory
            # or we can set the destination
            session = downloader.login(username, password)
            if session is None:
                print("ERROR: Failed to login to Cityscapes. Check your credentials.")
                return False
            
            # Download leftImg8bit package (the RGB images)
            # Package ID for leftImg8bit_trainvaltest.zip
            downloader.download_packages(
                session=session,
                package_names=['leftImg8bit_trainvaltest.zip'],
                destination_path=str(self.cityscapes_root)
            )
            
            print("Download complete! Extracting...")
            
            # Extract the zip file
            zip_path = self.cityscapes_root / "leftImg8bit_trainvaltest.zip"
            if zip_path.exists():
                import zipfile
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.cityscapes_root)
                print("Extraction complete!")
                
                # Optionally remove zip to save space
                # zip_path.unlink()
                
            return self.check_exists()
            
        except Exception as e:
            print(f"\nERROR during download: {e}")
            print("\nTrying alternative download method...")
            return self._download_alternative(username, password)
    
    def _download_alternative(self, username: str, password: str) -> bool:
        """
        Alternative download method using direct HTTP requests.
        
        Args:
            username: Cityscapes username (email)
            password: Cityscapes password
            
        Returns:
            True if successful, False otherwise
        """
        import requests # type: ignore
        from tqdm import tqdm # type: ignore
        
        print("\nUsing alternative download method...")
        
        # Cityscapes login and download URLs
        login_url = "https://www.cityscapes-dataset.com/login/"
        download_url = "https://www.cityscapes-dataset.com/file-handling/?packageID=3"
        
        session = requests.Session()
        
        try:
            # Get CSRF token
            response = session.get(login_url)
            
            # Try to find CSRF token in cookies or response
            csrf_token = session.cookies.get('csrftoken', '')
            
            # Login
            login_data = {
                'username': username,
                'password': password,
                'csrfmiddlewaretoken': csrf_token,
                'next': '/'
            }
            
            headers = {
                'Referer': login_url,
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = session.post(login_url, data=login_data, headers=headers)
            
            if 'logout' not in response.text.lower() and 'sign out' not in response.text.lower():
                print("Login may have failed. Attempting download anyway...")
            
            # Download the file
            print("Downloading leftImg8bit_trainvaltest.zip (~11GB)...")
            
            self.cityscapes_root.mkdir(parents=True, exist_ok=True)
            zip_path = self.cityscapes_root / "leftImg8bit_trainvaltest.zip"
            
            response = session.get(download_url, stream=True, headers=headers)
            
            if response.status_code != 200:
                print(f"Download failed with status code: {response.status_code}")
                return False
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(zip_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            print("Download complete! Extracting...")
            
            # Extract
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.cityscapes_root)
            
            print("Extraction complete!")
            return self.check_exists()
            
        except Exception as e:
            print(f"Alternative download failed: {e}")
            print(self.DOWNLOAD_INSTRUCTIONS.format(data_root=self.data_root))
            return False
    
    def get_image_paths(self, split: str = "train") -> List[Path]:
        """
        Get all image paths for a given split.
        
        Cityscapes organizes images by city, so we need to traverse subdirectories.
        
        Args:
            split: 'train', 'val', or 'test'
            
        Returns:
            List of paths to all images in the split
        """
        split_dir = self.leftimg_root / split
        
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Cityscapes {split} split not found at {split_dir}."
            )
        
        # Find all images across city subdirectories
        image_paths = []
        for city_dir in split_dir.iterdir():
            if city_dir.is_dir():
                for img_path in city_dir.glob("*_leftImg8bit.png"):
                    image_paths.append(img_path)
        
        image_paths.sort()
        print(f"Found {len(image_paths)} Cityscapes {split} images")
        return image_paths
    
    def get_train_paths(self) -> List[Path]:
        """Get training image paths."""
        return self.get_image_paths("train")
    
    def get_val_paths(self) -> List[Path]:
        """Get validation image paths."""
        return self.get_image_paths("val")


class COCODownloader:
    """
    Utility class to download COCO 2017 dataset.
    
    We use the train2017 split which contains ~118K images with good
    coverage of outdoor scenes. The validation split (val2017, ~5K images)
    is used for evaluation.
    """
    
    # URLs for COCO 2017 dataset
    TRAIN_URL = "http://images.cocodataset.org/zips/train2017.zip"
    VAL_URL = "http://images.cocodataset.org/zips/val2017.zip"
    
    def __init__(self, data_root: str):
        """
        Initialize the downloader.
        
        Args:
            data_root: Root directory where data will be stored
        """
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)
        
    def download_and_extract(self, url: str, extract_to: str) -> Path:
        """
        Download a zip file and extract it.
        
        Args:
            url: URL to download from
            extract_to: Subdirectory name for extraction
            
        Returns:
            Path to the extracted directory
        """
        zip_path = self.data_root / f"{extract_to}.zip"
        extract_path = self.data_root / "coco"
        
        # Check if already extracted
        final_path = extract_path / extract_to
        if final_path.exists() and any(final_path.iterdir()):
            print(f"Dataset already exists at {final_path}")
            return final_path
            
        # Download if zip doesn't exist
        if not zip_path.exists():
            print(f"Downloading {url}...")
            print("This may take a while (COCO train2017 is ~18GB)...")
            
            # Download with progress
            def report_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(100, downloaded * 100 / total_size)
                print(f"\rProgress: {percent:.1f}% ({downloaded / 1e9:.2f} GB / {total_size / 1e9:.2f} GB)", 
                      end='', flush=True)
            
            urllib.request.urlretrieve(url, zip_path, reporthook=report_progress)
            print("\nDownload complete!")
        
        # Extract
        print(f"Extracting to {extract_path}...")
        extract_path.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("Extraction complete!")
        
        # Optionally remove zip to save space
        # zip_path.unlink()
        
        return final_path
    
    def download_train(self) -> Path:
        """Download and extract training set."""
        return self.download_and_extract(self.TRAIN_URL, "train2017")
    
    def download_val(self) -> Path:
        """Download and extract validation set."""
        return self.download_and_extract(self.VAL_URL, "val2017")


class IRColorPairDataset(Dataset):
    """
    Dataset class for IR-to-Color image translation training.
    
    For each sample, this dataset provides:
    - ir_image: Simulated IR image (grayscale, from red channel of a crop)
    - ref_image: Full color reference image
    - target_image: Ground truth colorized version of the IR region
    - crop_coords: The crop coordinates used (for visualization/debugging)
    
    The simulation process:
    1. Load a full-resolution color image
    2. Select a random crop region (simulating different FOV)
    3. Extract the red channel from the crop and convert to grayscale (simulated IR)
    4. The full image serves as the color reference
    5. The crop region in full color is the ground truth target
    """
    
    def __init__(
        self,
        image_source: Union[str, List[Path]],
        config: DataConfig,
        is_training: bool = True,
        max_samples: Optional[int] = None,
        fixed_crop_seed: Optional[int] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            image_source: Either a directory path (str) or list of image paths
            config: Data configuration object
            is_training: Whether this is training (enables augmentation)
            max_samples: Optional limit on number of samples (for debugging)
            fixed_crop_seed: If set, uses deterministic cropping based on this seed
                           (useful for consistent visualization across epochs)
        """
        self.config = config
        self.is_training = is_training
        self.fixed_crop_seed = fixed_crop_seed
        
        # Handle both directory path and list of paths
        if isinstance(image_source, (str, Path)):
            image_dir = Path(image_source)
            # Get list of all image files
            valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
            self.image_paths = [
                p for p in image_dir.iterdir()
                if p.suffix.lower() in valid_extensions
            ]
        else:
            # Assume it's a list of paths
            self.image_paths = list(image_source)
        
        # Sort for reproducibility
        self.image_paths.sort()
        
        # Optionally limit samples
        if max_samples is not None:
            self.image_paths = self.image_paths[:max_samples]
            
        print(f"Dataset initialized with {len(self.image_paths)} images")
        
        # Setup transforms
        self._setup_transforms()
        
    def _setup_transforms(self):
        """
        Setup the image transformation pipelines.
        
        We use separate transforms for:
        - Reference image: Resize to ref_image_size, normalize
        - IR image: Will be created from crop, then resized and normalized
        - Target image: The colorized ground truth
        """
        # Normalization parameters (ImageNet stats, since we use pretrained features)
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # For grayscale IR image, we still normalize but repeat to 3 channels
        # This allows using the same pretrained encoder
        self.normalize_gray = T.Normalize(
            mean=[0.485, 0.485, 0.485],  # Same value for all channels
            std=[0.229, 0.229, 0.229]
        )
        
        # Color jitter for augmentation (only during training)
        if self.is_training and self.config.use_augmentation:
            self.color_jitter = T.ColorJitter(
                brightness=self.config.color_jitter_brightness,
                contrast=self.config.color_jitter_contrast,
                saturation=self.config.color_jitter_saturation,
                hue=self.config.color_jitter_hue
            )
        else:
            self.color_jitter = None
    
    def _apply_geometric_augmentation(self, image: Image.Image) -> Image.Image:
        """
        Apply geometric augmentations (rotation and perspective) to reference image only.
        
        These are applied only to the reference image to simulate it being
        captured from a different perspective/viewpoint than the IR input.
        This forces the network to learn robust feature matching across
        different viewpoints.
        
        Args:
            image: PIL Image (reference image)
            
        Returns:
            Augmented PIL Image
        """
        # Random rotation
        if self.config.random_rotation and random.random() < 0.5:
            angle = random.uniform(
                -self.config.max_rotation_angle, 
                self.config.max_rotation_angle
            )
            image = TF.rotate(image, angle, fill=0)
        
        # Random perspective warping
        if self.config.random_perspective and random.random() < 0.5:
            # Get image dimensions
            width, height = image.size
            
            # Generate random perspective distortion
            # startpoints and endpoints define the transformation
            distortion = self.config.perspective_distortion
            
            # Calculate random corner displacements
            # Each corner can move by up to distortion * dimension
            max_dx = int(width * distortion)
            max_dy = int(height * distortion)
            
            # Original corners
            startpoints = [
                (0, 0),
                (width, 0),
                (width, height),
                (0, height)
            ]
            
            # Randomly displaced corners
            endpoints = [
                (random.randint(0, max_dx), random.randint(0, max_dy)),
                (width - random.randint(0, max_dx), random.randint(0, max_dy)),
                (width - random.randint(0, max_dx), height - random.randint(0, max_dy)),
                (random.randint(0, max_dx), height - random.randint(0, max_dy))
            ]
            
            image = TF.perspective(image, startpoints, endpoints, fill=0)
        
        return image
            
    def _get_random_crop_params(
        self, 
        img_width: int, 
        img_height: int
    ) -> Tuple[int, int, int, int]:
        """
        Generate random crop parameters.
        
        The crop simulates the different field of view of the IR sensor.
        We ensure the crop is large enough to contain meaningful content
        but small enough to create a perspective difference.
        
        Args:
            img_width: Original image width
            img_height: Original image height
            
        Returns:
            Tuple of (top, left, crop_height, crop_width)
        """
        # Random crop ratio within configured range
        min_ratio, max_ratio = self.config.crop_ratio_range
        crop_ratio = random.uniform(min_ratio, max_ratio)
        
        # Calculate crop dimensions
        crop_height = int(img_height * crop_ratio)
        crop_width = int(img_width * crop_ratio)
        
        # Random position for the crop
        max_top = img_height - crop_height
        max_left = img_width - crop_width
        
        top = random.randint(0, max_top) if max_top > 0 else 0
        left = random.randint(0, max_left) if max_left > 0 else 0
        
        return top, left, crop_height, crop_width
    
    def _simulate_ir_from_crop(self, crop: Image.Image) -> Image.Image:
        """
        Create a simulated IR image from a color crop.
        
        The simulation extracts the red channel, which approximates near-IR
        reflectance for many natural materials (vegetation, soil, etc.).
        
        For more realistic simulation, you could:
        - Apply a gamma correction to simulate sensor response
        - Add noise to simulate sensor characteristics
        - Apply slight blur to simulate different optical properties
        
        Args:
            crop: Color PIL Image of the cropped region
            
        Returns:
            Grayscale PIL Image simulating IR capture
        """
        # Convert to numpy for channel manipulation
        crop_np = np.array(crop)
        
        # Extract red channel (index 0 in RGB)
        red_channel = crop_np[:, :, 0]
        
        # Optional: Add slight Gaussian noise to simulate sensor noise
        # This helps the network become robust to noise
        if self.is_training and random.random() < 0.5:
            noise_std = random.uniform(0, 5)  # Small noise, 0-5 intensity levels
            noise = np.random.normal(0, noise_std, red_channel.shape)
            red_channel = np.clip(red_channel + noise, 0, 255).astype(np.uint8)
        
        # Convert to PIL grayscale image
        ir_image = Image.fromarray(red_channel, mode='L')
        
        return ir_image
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single training sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing:
            - 'ir_image': Tensor [3, H, W] - Grayscale IR (repeated to 3 channels)
            - 'ref_image': Tensor [3, H, W] - Color reference image
            - 'target_image': Tensor [3, H, W] - Ground truth colorized crop
            - 'crop_coords': Tuple (top, left, height, width) - For visualization
            - 'image_path': str - Original image path (for debugging)
        """
        # Load the image
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a random other sample instead
            return self.__getitem__(random.randint(0, len(self) - 1))
        
        # Ensure minimum size for cropping
        min_size = 300  # Minimum dimension for useful crops
        if min(image.size) < min_size:
            # Resize up if too small
            scale = min_size / min(image.size)
            new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
            image = image.resize(new_size, Image.BILINEAR)
        
        img_width, img_height = image.size
        
        # Apply color jitter to the whole image (if training)
        # This augments the color distribution the network sees
        if self.color_jitter is not None:
            image = self.color_jitter(image)
        
        # Random horizontal flip (applied consistently to all derived images)
        # Skip if using fixed seed for deterministic visualization
        do_flip = self.is_training and self.config.random_horizontal_flip and random.random() < 0.5 and self.fixed_crop_seed is None
        if do_flip:
            image = TF.hflip(image)
        
        # Get crop parameters for simulated IR region
        # Use deterministic seed if provided (for consistent visualization)
        if self.fixed_crop_seed is not None:
            # Save current random state
            saved_state = random.getstate()
            # Set deterministic seed based on fixed_crop_seed + idx
            random.seed(self.fixed_crop_seed + idx)
            top, left, crop_h, crop_w = self._get_random_crop_params(img_width, img_height)
            # Restore random state
            random.setstate(saved_state)
        else:
            top, left, crop_h, crop_w = self._get_random_crop_params(img_width, img_height)
        
        # Extract the crop region
        crop = TF.crop(image, top, left, crop_h, crop_w)
        
        # Create simulated IR from the crop
        ir_image = self._simulate_ir_from_crop(crop)
        
        # Resize everything to target dimensions
        # Reference: full image resized to ref_image_size
        ref_image = TF.resize(image, self.config.ref_image_size)
        
        # Apply geometric augmentation ONLY to reference image (after resizing)
        # This simulates the reference being captured from a different perspective
        # than the IR input, forcing the network to learn robust feature matching
        # Skip if using fixed seed for deterministic visualization
        if self.is_training and self.config.use_augmentation and self.fixed_crop_seed is None:
            ref_image = self._apply_geometric_augmentation(ref_image)
        
        # IR: grayscale crop resized to ir_image_size
        ir_image = TF.resize(ir_image, self.config.ir_image_size)
        
        # Target: color crop resized to ir_image_size (same size as IR)
        target_image = TF.resize(crop, self.config.ir_image_size)
        
        # Convert to tensors
        ref_tensor = TF.to_tensor(ref_image)  # [3, H, W]
        target_tensor = TF.to_tensor(target_image)  # [3, H, W]
        
        # For IR, convert to tensor and repeat to 3 channels
        # This allows using the same encoder architecture
        ir_tensor = TF.to_tensor(ir_image)  # [1, H, W]
        ir_tensor = ir_tensor.repeat(3, 1, 1)  # [3, H, W]
        
        # Normalize all tensors
        ref_tensor = self.normalize(ref_tensor)
        target_tensor = self.normalize(target_tensor)
        ir_tensor = self.normalize_gray(ir_tensor)
        
        return {
            'ir_image': ir_tensor,
            'ref_image': ref_tensor,
            'target_image': target_tensor,
            'crop_coords': (top, left, crop_h, crop_w),
            'image_path': str(img_path)
        }


def get_dataloaders(config: Config) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders.
    
    This function handles:
    1. Loading/downloading the dataset based on config
    2. Creating train/val dataset objects
    3. Wrapping them in DataLoaders with appropriate settings
    
    Supported datasets:
    - 'cityscapes': European urban scenes (requires manual download)
    - 'coco': COCO 2017 (auto-downloads)
    
    Args:
        config: Complete configuration object
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    data_config = config.data
    training_config = config.training
    
    dataset_name = data_config.dataset_name.lower()
    
    if dataset_name == "cityscapes":
        # Load Cityscapes dataset
        loader = CityscapesLoader(data_config.data_root)
        
        if not loader.check_exists():
            print("\nCityscapes dataset not found. Attempting to download...")
            success = loader.download()
            
            if not success:
                raise FileNotFoundError(
                    "Could not download Cityscapes dataset. "
                    "Please follow the instructions above to set up credentials or download manually, "
                    "or use --dataset coco for automatic download without registration."
                )
        
        train_paths = loader.get_train_paths()
        val_paths = loader.get_val_paths()
        
        # Create datasets with path lists
        train_dataset = IRColorPairDataset(
            image_source=train_paths,
            config=data_config,
            is_training=True,
            max_samples=data_config.max_train_samples
        )
        
        val_dataset = IRColorPairDataset(
            image_source=val_paths,
            config=data_config,
            is_training=False
        )
        
    elif dataset_name == "coco":
        # Download and load COCO dataset
        downloader = COCODownloader(data_config.data_root)
        train_dir = downloader.download_train()
        val_dir = downloader.download_val()
        
        # Create datasets with directory paths
        train_dataset = IRColorPairDataset(
            image_source=str(train_dir),
            config=data_config,
            is_training=True,
            max_samples=data_config.max_train_samples
        )
        
        val_dataset = IRColorPairDataset(
            image_source=str(val_dir),
            config=data_config,
            is_training=False
        )
    
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Supported datasets: 'cityscapes', 'coco'"
        )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=data_config.num_workers,
        pin_memory=True,
        drop_last=True  # Drop incomplete batches for consistent batch norm
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader


def denormalize(tensor: torch.Tensor, gray: bool = False) -> torch.Tensor:
    """
    Denormalize a tensor for visualization.
    
    Reverses the ImageNet normalization applied during preprocessing.
    
    Args:
        tensor: Normalized tensor [B, 3, H, W] or [3, H, W]
        gray: Whether this is a grayscale image (use gray normalization)
        
    Returns:
        Denormalized tensor in [0, 1] range
    """
    if gray:
        mean = torch.tensor([0.485, 0.485, 0.485]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.229, 0.229]).view(3, 1, 1)
    else:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    if tensor.device.type != 'cpu':
        mean = mean.to(tensor.device)
        std = std.to(tensor.device)
    
    # Handle batched input
    if tensor.dim() == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    
    denorm = tensor * std + mean
    return torch.clamp(denorm, 0, 1)


# Quick test when run directly
if __name__ == "__main__":
    from config import get_config
    
    config = get_config()
    
    # Test with a small subset
    print("Testing dataset creation...")
    
    # This will download if needed (can take a while!)
    train_loader, val_loader = get_dataloaders(config)
    
    # Get one batch
    batch = next(iter(train_loader))
    
    print(f"\nBatch contents:")
    print(f"  IR image shape: {batch['ir_image'].shape}")
    print(f"  Reference image shape: {batch['ref_image'].shape}")
    print(f"  Target image shape: {batch['target_image'].shape}")
    print(f"  IR value range: [{batch['ir_image'].min():.3f}, {batch['ir_image'].max():.3f}]")
    print(f"  Target value range: [{batch['target_image'].min():.3f}, {batch['target_image'].max():.3f}]")
