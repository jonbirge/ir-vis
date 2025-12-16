"""
Component Tests for IR-to-Color Image Translation

This module provides comprehensive tests for all major components:
- Model architecture and forward passes
- Dataset loading and preprocessing
- Loss functions
- Utility functions

Run all tests:
    python tests.py

Run specific test:
    python tests.py TestModel
    python tests.py TestLosses.test_combined_loss
"""

import sys
import unittest
import tempfile
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from PIL import Image


class TestConfig(unittest.TestCase):
    """Tests for configuration module."""
    
    def test_config_creation(self):
        """Test that config can be created with defaults."""
        from config import get_config, Config
        
        config = get_config()
        self.assertIsInstance(config, Config)
        
    def test_config_attributes(self):
        """Test that all expected config attributes exist."""
        from config import get_config
        
        config = get_config()
        
        # Check sub-configs exist
        self.assertTrue(hasattr(config, 'data'))
        self.assertTrue(hasattr(config, 'model'))
        self.assertTrue(hasattr(config, 'loss'))
        self.assertTrue(hasattr(config, 'training'))
        
        # Check some specific attributes
        self.assertIsInstance(config.training.batch_size, int)
        self.assertIsInstance(config.training.learning_rate, float)
        self.assertIsInstance(config.model.encoder_backbone, str)
        
    def test_config_directories_created(self):
        """Test that output directories are created."""
        from config import get_config
        
        config = get_config()
        
        # Check directories exist
        self.assertTrue(Path(config.training.output_dir).exists())

    def test_curriculum_default_creation(self):
        """Test that the default curriculum can be created and has expected structure."""
        from config import get_curriculum, Curriculum

        curriculum = get_curriculum()
        self.assertIsInstance(curriculum, Curriculum)
        self.assertEqual(len(curriculum.stages), 3)
        self.assertEqual(curriculum.total_epochs, 300)

        # Stage semantics from TODO.md
        self.assertEqual(curriculum.stages[0].data.dataset_name, 'coco')
        self.assertEqual(curriculum.stages[1].data.dataset_name, 'coco')
        self.assertEqual(curriculum.stages[2].data.dataset_name, 'cityscapes')

        self.assertEqual(curriculum.stages[0].training.num_epochs, 75)
        self.assertEqual(curriculum.stages[1].training.num_epochs, 75)
        self.assertEqual(curriculum.stages[2].training.num_epochs, 150)

        self.assertEqual(curriculum.stages[0].loss.l1_weight, 0.0)
        self.assertGreater(curriculum.stages[1].loss.l1_weight, 0.0)

        # Object reuse: model and training should be shared across all stages
        self.assertIs(curriculum.stages[0].model, curriculum.stages[1].model)
        self.assertIs(curriculum.stages[1].model, curriculum.stages[2].model)
        self.assertIs(curriculum.stages[0].training, curriculum.stages[1].training)
        self.assertIsNot(curriculum.stages[1].training, curriculum.stages[2].training)

    def test_curriculum_yaml_round_trip(self):
        """Test curriculum save/load via load_experiment_config dispatcher."""
        from config import get_curriculum, load_experiment_config, Curriculum

        curriculum = get_curriculum()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'curriculum.yaml'
            curriculum.save_yaml(path)
            loaded = load_experiment_config(path)

            self.assertIsInstance(loaded, Curriculum)
            self.assertEqual(len(loaded.stages), 3)
            self.assertEqual(loaded.total_epochs, 300)

    def test_config_copy_reuses_sections(self):
        """Test Config.copy defaults to reusing unchanged section objects."""
        from config import get_config

        cfg = get_config()
        cfg2 = cfg.copy(loss={'l1_weight': 0.0})

        # Changed section is new, unchanged sections reused
        self.assertIs(cfg2.data, cfg.data)
        self.assertIs(cfg2.model, cfg.model)
        self.assertIs(cfg2.training, cfg.training)
        self.assertIsNot(cfg2.loss, cfg.loss)
        self.assertEqual(cfg2.loss.l1_weight, 0.0)


class TestModel(unittest.TestCase):
    """Tests for model architecture."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        from config import get_config
        cls.config = get_config()
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def test_model_creation(self):
        """Test that model can be created."""
        from model import create_model
        
        model = create_model(self.config.model)
        self.assertIsNotNone(model)
        
    def test_model_forward_pass(self):
        """Test forward pass with random inputs."""
        from model import create_model
        
        model = create_model(self.config.model).to(self.device)
        model.eval()
        
        # Create random inputs
        batch_size = 2
        ir_image = torch.randn(batch_size, 3, 256, 256).to(self.device)
        ref_image = torch.randn(batch_size, 3, 256, 256).to(self.device)
        
        with torch.no_grad():
            outputs = model(ir_image, ref_image)
        
        # Check output structure
        self.assertIn('output', outputs)
        self.assertIn('content_features', outputs)
        self.assertIn('ref_features', outputs)
        
        # Check output shape
        self.assertEqual(outputs['output'].shape, (batch_size, 3, 256, 256))
        
    def test_model_output_range(self):
        """Test that model output is in expected range [-1, 1]."""
        from model import create_model
        
        model = create_model(self.config.model).to(self.device)
        model.eval()
        
        ir_image = torch.randn(2, 3, 256, 256).to(self.device)
        ref_image = torch.randn(2, 3, 256, 256).to(self.device)
        
        with torch.no_grad():
            outputs = model(ir_image, ref_image)
        
        # Output should be in [-1, 1] due to tanh activation
        self.assertLessEqual(outputs['output'].max().item(), 1.0)
        self.assertGreaterEqual(outputs['output'].min().item(), -1.0)
        
    def test_model_different_input_sizes(self):
        """Test model with different input sizes."""
        from model import create_model
        
        model = create_model(self.config.model).to(self.device)
        model.eval()
        
        for size in [128, 256, 512]:
            ir_image = torch.randn(1, 3, size, size).to(self.device)
            ref_image = torch.randn(1, 3, size, size).to(self.device)
            
            with torch.no_grad():
                outputs = model(ir_image, ref_image)
            
            self.assertEqual(outputs['output'].shape, (1, 3, size, size),
                           f"Failed for size {size}")
            
    def test_encoder_features(self):
        """Test encoder feature extraction."""
        from model import ResNetEncoder
        
        encoder = ResNetEncoder(backbone='resnet34', pretrained=False).to(self.device)
        
        x = torch.randn(2, 3, 256, 256).to(self.device)
        features, skip_features = encoder(x)
        
        # Check number of skip features
        self.assertEqual(len(skip_features), 5)
        
        # Check feature dimensions decrease spatially
        prev_size = 256
        for i, skip in enumerate(skip_features):
            self.assertLessEqual(skip.shape[2], prev_size)
            prev_size = skip.shape[2]
            
    def test_cross_attention(self):
        """Test cross-attention module."""
        from model import CrossAttention
        
        embed_dim = 256
        attn = CrossAttention(embed_dim=embed_dim, num_heads=8).to(self.device)
        
        query = torch.randn(2, 64, embed_dim).to(self.device)
        key = torch.randn(2, 100, embed_dim).to(self.device)
        value = torch.randn(2, 100, embed_dim).to(self.device)
        
        output, attn_weights = attn(query, key, value, return_attention=True)
        
        self.assertEqual(output.shape, query.shape)
        self.assertEqual(attn_weights.shape, (2, 8, 64, 100))
        
    def test_decoder_block(self):
        """Test decoder block upsampling."""
        from model import DecoderBlock
        
        block = DecoderBlock(
            in_channels=256,
            skip_channels=128,
            out_channels=128
        ).to(self.device)
        
        x = torch.randn(2, 256, 8, 8).to(self.device)
        skip = torch.randn(2, 128, 16, 16).to(self.device)
        
        output = block(x, skip)
        
        # Should upsample by 2x
        self.assertEqual(output.shape, (2, 128, 16, 16))


class TestLosses(unittest.TestCase):
    """Tests for loss functions."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        from config import get_config
        cls.config = get_config()
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def test_vgg_features(self):
        """Test VGG feature extraction."""
        from losses import VGGFeatures
        
        layers = ['relu1_2', 'relu2_2', 'relu3_4']
        vgg = VGGFeatures(layers).to(self.device)
        
        x = torch.randn(2, 3, 256, 256).clamp(-1, 1).to(self.device)
        features = vgg(x)
        
        self.assertEqual(len(features), len(layers))
        for layer in layers:
            self.assertIn(layer, features)
            
    def test_gram_matrix(self):
        """Test Gram matrix computation."""
        from losses import gram_matrix
        
        features = torch.randn(2, 64, 32, 32).to(self.device)
        gram = gram_matrix(features)
        
        self.assertEqual(gram.shape, (2, 64, 64))
        
        # Gram matrix should be symmetric
        diff = torch.abs(gram - gram.transpose(1, 2))
        self.assertLess(diff.max().item(), 1e-5)
        
    def test_perceptual_loss(self):
        """Test perceptual loss computation."""
        from losses import PerceptualLoss
        
        layers = ['relu2_2', 'relu3_4']
        loss_fn = PerceptualLoss(layers).to(self.device)
        
        pred = torch.randn(2, 3, 256, 256).clamp(-1, 1).to(self.device)
        target = torch.randn(2, 3, 256, 256).clamp(-1, 1).to(self.device)
        
        loss, layer_losses = loss_fn(pred, target)
        
        self.assertGreater(loss.item(), 0)
        self.assertEqual(len(layer_losses), len(layers))
        
    def test_style_loss(self):
        """Test style loss computation."""
        from losses import StyleLoss
        
        layers = ['relu1_2', 'relu2_2']
        loss_fn = StyleLoss(layers).to(self.device)
        
        pred = torch.randn(2, 3, 256, 256).clamp(-1, 1).to(self.device)
        target = torch.randn(2, 3, 256, 256).clamp(-1, 1).to(self.device)
        
        loss, layer_losses = loss_fn(pred, target)
        
        self.assertGreater(loss.item(), 0)
        
    def test_histogram_loss(self):
        """Test color histogram loss."""
        from losses import ColorHistogramLoss
        
        loss_fn = ColorHistogramLoss(num_bins=64).to(self.device)
        
        pred = torch.randn(2, 3, 256, 256).clamp(-1, 1).to(self.device)
        target = torch.randn(2, 3, 256, 256).clamp(-1, 1).to(self.device)
        
        loss = loss_fn(pred, target)
        
        self.assertGreater(loss.item(), 0)
        
        # Same image should have zero loss
        loss_same = loss_fn(pred, pred)
        self.assertLess(loss_same.item(), 0.01)
        
    def test_combined_loss(self):
        """Test combined loss function."""
        from losses import CombinedLoss
        
        loss_fn = CombinedLoss(self.config.loss).to(self.device)
        
        pred = torch.randn(2, 3, 256, 256).clamp(-1, 1).to(self.device)
        target = torch.randn(2, 3, 256, 256).clamp(-1, 1).to(self.device)
        
        total_loss, loss_dict = loss_fn(pred, target)
        
        # Check all expected losses are present
        self.assertIn('total', loss_dict)
        self.assertIn('l1', loss_dict)
        self.assertIn('perceptual', loss_dict)
        self.assertIn('style', loss_dict)
        self.assertIn('histogram', loss_dict)
        
        # Check no NaN
        self.assertFalse(torch.isnan(total_loss))
        
    def test_loss_gradients(self):
        """Test that gradients flow through loss."""
        from losses import CombinedLoss
        
        loss_fn = CombinedLoss(self.config.loss).to(self.device)
        
        # Create a simple tensor that requires grad
        # Use a small size for speed
        pred = torch.randn(2, 3, 64, 64, device=self.device, requires_grad=True)
        
        # Apply tanh to get [-1, 1] range while preserving gradients
        pred_clamped = torch.tanh(pred)
        
        # Target doesn't need gradients
        target = torch.tanh(torch.randn(2, 3, 64, 64, device=self.device))
        
        total_loss, _ = loss_fn(pred_clamped, target)
        total_loss.backward()
        
        self.assertIsNotNone(pred.grad, "Gradient should not be None")
        self.assertFalse(torch.isnan(pred.grad).any(), "Gradient should not contain NaN")
        self.assertTrue((pred.grad.abs() > 0).any(), "Gradients should be non-zero somewhere")


class TestUtils(unittest.TestCase):
    """Tests for utility functions."""
    
    def test_set_seed(self):
        """Test reproducibility with seed setting."""
        from utils import set_seed
        import random
        
        set_seed(42)
        val1 = random.random()
        
        set_seed(42)
        val2 = random.random()
        
        self.assertEqual(val1, val2)
        
    def test_denormalize(self):
        """Test tensor denormalization."""
        from utils import denormalize
        
        # Create normalized tensor
        tensor = torch.randn(1, 3, 64, 64)
        denorm = denormalize(tensor)
        
        # Should be in [0, 1] range
        self.assertGreaterEqual(denorm.min().item(), 0.0)
        self.assertLessEqual(denorm.max().item(), 1.0)
        
    def test_tensor_to_image(self):
        """Test tensor to image conversion."""
        from utils import tensor_to_image
        
        tensor = torch.rand(3, 64, 64)  # [0, 1] range
        image = tensor_to_image(tensor)
        
        self.assertEqual(image.shape, (64, 64, 3))
        self.assertEqual(image.dtype, np.uint8)
        self.assertGreaterEqual(image.min(), 0)
        self.assertLessEqual(image.max(), 255)
        
    def test_format_time(self):
        """Test time formatting."""
        from utils import format_time
        
        self.assertEqual(format_time(30), "30s")
        self.assertEqual(format_time(90), "1m 30s")
        self.assertEqual(format_time(3661), "1h 1m 1s")
        
    def test_count_parameters(self):
        """Test parameter counting."""
        from utils import count_parameters
        
        model = nn.Sequential(
            nn.Linear(10, 20),  # 10*20 + 20 = 220 params
            nn.Linear(20, 5)   # 20*5 + 5 = 105 params
        )
        
        counts = count_parameters(model)
        
        self.assertEqual(counts['total'], 325)
        self.assertEqual(counts['trainable'], 325)
        self.assertEqual(counts['frozen'], 0)
        
    def test_metrics_logger(self):
        """Test metrics logging."""
        from utils import MetricsLogger
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(tmpdir)
            
            # Log some metrics
            for epoch in range(3):
                for i in range(5):
                    logger.log_iteration({'loss': 0.5 - epoch * 0.1})
                avg = logger.end_epoch(epoch)
                self.assertIn('train_loss', avg)
                
            logger.save()
            
            # Check file was created
            self.assertTrue(Path(tmpdir, 'metrics.json').exists())
            
    def test_checkpoint_save_load(self):
        """Test checkpoint saving and loading."""
        from utils import save_checkpoint, load_checkpoint
        from config import get_config
        
        config = get_config()
        
        # Create simple model
        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "test_checkpoint.pt"
            
            # Save
            save_checkpoint(
                model, optimizer, None, 5, 0.123, config, str(checkpoint_path)
            )
            
            # Verify file exists
            self.assertTrue(checkpoint_path.exists())
            
            # Load into new model
            model2 = nn.Linear(10, 5)
            optimizer2 = torch.optim.Adam(model2.parameters())
            
            metadata = load_checkpoint(
                str(checkpoint_path), model2, optimizer2
            )
            
            self.assertEqual(metadata['epoch'], 5)
            self.assertAlmostEqual(metadata['loss'], 0.123, places=5)


class TestDataset(unittest.TestCase):
    """Tests for dataset functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures - create dummy images."""
        from config import get_config
        cls.config = get_config()
        
        # Create temporary directory with dummy images
        cls.tmpdir = tempfile.mkdtemp()
        cls.img_dir = Path(cls.tmpdir) / "images"
        cls.img_dir.mkdir()
        
        # Create dummy images
        for i in range(5):
            img = Image.fromarray(
                np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            )
            img.save(cls.img_dir / f"image_{i}.jpg")
            
    @classmethod
    def tearDownClass(cls):
        """Clean up temporary files."""
        shutil.rmtree(cls.tmpdir)
        
    def test_dataset_creation(self):
        """Test dataset can be created from directory."""
        from dataset import IRColorPairDataset
        
        dataset = IRColorPairDataset(
            image_source=str(self.img_dir),
            config=self.config.data,
            is_training=True
        )
        
        self.assertEqual(len(dataset), 5)
        
    def test_dataset_getitem(self):
        """Test dataset item retrieval."""
        from dataset import IRColorPairDataset
        
        dataset = IRColorPairDataset(
            image_source=str(self.img_dir),
            config=self.config.data,
            is_training=True
        )
        
        sample = dataset[0]
        
        # Check all expected keys
        self.assertIn('ir_image', sample)
        self.assertIn('ref_image', sample)
        self.assertIn('target_image', sample)
        self.assertIn('crop_coords', sample)
        self.assertIn('image_path', sample)
        
        # Check shapes
        self.assertEqual(sample['ir_image'].shape[0], 3)
        self.assertEqual(sample['ref_image'].shape[0], 3)
        self.assertEqual(sample['target_image'].shape[0], 3)
        
    def test_dataset_normalization(self):
        """Test that images are normalized."""
        from dataset import IRColorPairDataset
        
        dataset = IRColorPairDataset(
            image_source=str(self.img_dir),
            config=self.config.data,
            is_training=False  # No augmentation
        )
        
        sample = dataset[0]
        
        # Normalized images should have values outside [0, 1]
        # (due to ImageNet normalization with negative values possible)
        ir_range = sample['ir_image'].max() - sample['ir_image'].min()
        self.assertGreater(ir_range.item(), 0)
        
    def test_denormalize_function(self):
        """Test denormalization function."""
        from dataset import denormalize
        
        # Create a normalized tensor
        tensor = torch.randn(1, 3, 64, 64)
        denorm = denormalize(tensor)
        
        # Should be clamped to [0, 1]
        self.assertGreaterEqual(denorm.min().item(), 0.0)
        self.assertLessEqual(denorm.max().item(), 1.0)


class TestIntegration(unittest.TestCase):
    """Integration tests for full pipeline."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        from config import get_config
        cls.config = get_config()
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create temporary directory with dummy images
        cls.tmpdir = tempfile.mkdtemp()
        cls.img_dir = Path(cls.tmpdir) / "images"
        cls.img_dir.mkdir()
        
        for i in range(4):
            img = Image.fromarray(
                np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            )
            img.save(cls.img_dir / f"image_{i}.jpg")
            
    @classmethod
    def tearDownClass(cls):
        """Clean up temporary files."""
        shutil.rmtree(cls.tmpdir)
        
    def test_forward_backward_pass(self):
        """Test complete forward and backward pass."""
        from model import create_model
        from losses import CombinedLoss
        from dataset import IRColorPairDataset
        from torch.utils.data import DataLoader
        
        # Create model and loss
        model = create_model(self.config.model).to(self.device)
        criterion = CombinedLoss(self.config.loss).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # Create dataset and loader
        dataset = IRColorPairDataset(
            image_source=str(self.img_dir),
            config=self.config.data,
            is_training=True
        )
        loader = DataLoader(dataset, batch_size=2, shuffle=True)
        
        # Get a batch
        batch = next(iter(loader))
        ir_image = batch['ir_image'].to(self.device)
        ref_image = batch['ref_image'].to(self.device)
        target_image = batch['target_image'].to(self.device)
        
        # Forward pass
        model.train()
        outputs = model(ir_image, ref_image)
        
        # Compute loss (convert target to [-1, 1])
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        target_denorm = target_image * std + mean
        target_for_loss = target_denorm * 2 - 1
        
        loss, loss_dict = criterion(outputs['output'], target_for_loss)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check gradients exist and are finite
        for param in model.parameters():
            if param.grad is not None:
                self.assertFalse(torch.isnan(param.grad).any())
                self.assertFalse(torch.isinf(param.grad).any())
                
    def test_visualization_generation(self):
        """Test that visualizations can be generated."""
        from model import create_model
        from utils import save_batch_visualization
        from dataset import IRColorPairDataset
        from torch.utils.data import DataLoader
        
        model = create_model(self.config.model).to(self.device)
        model.eval()
        
        dataset = IRColorPairDataset(
            image_source=str(self.img_dir),
            config=self.config.data,
            is_training=False
        )
        loader = DataLoader(dataset, batch_size=2)
        
        batch = next(iter(loader))
        ir_image = batch['ir_image'].to(self.device)
        ref_image = batch['ref_image'].to(self.device)
        target_image = batch['target_image']
        
        with torch.no_grad():
            outputs = model(ir_image, ref_image)
        
        viz_path = Path(self.tmpdir) / "test_viz.png"
        save_batch_visualization(
            ir_image.cpu(),
            ref_image.cpu(),
            outputs['output'].cpu(),
            target_image,
            str(viz_path),
            max_samples=2
        )
        
        self.assertTrue(viz_path.exists())


def run_tests(test_name=None):
    """Run tests with optional filtering."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    if test_name:
        # Run specific test class or method
        if '.' in test_name:
            # Test method specified
            class_name, method_name = test_name.split('.')
            test_class = globals().get(class_name)
            if test_class:
                suite.addTest(test_class(method_name))
        else:
            # Test class specified
            test_class = globals().get(test_name)
            if test_class:
                suite.addTests(loader.loadTestsFromTestCase(test_class))
    else:
        # Run all tests
        for name, obj in globals().items():
            if isinstance(obj, type) and issubclass(obj, unittest.TestCase):
                if obj != unittest.TestCase:
                    suite.addTests(loader.loadTestsFromTestCase(obj))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    test_name = sys.argv[1] if len(sys.argv) > 1 else None
    success = run_tests(test_name)
    sys.exit(0 if success else 1)
