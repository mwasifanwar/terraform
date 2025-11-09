import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from src.computer_vision.mineral_detector import MineralDetectionModel
from src.computer_vision.formation_analyzer import FormationModel

class TestComputerVision(unittest.TestCase):
    def test_mineral_detection(self):
        detector = MineralDetectionModel()
        test_image = np.random.rand(224, 224, 3).astype(np.float32)
        spectral_bands = np.random.rand(13, 224, 224).astype(np.float32)
        
        detection = detector.detect_minerals(test_image, spectral_bands)
        self.assertIn('mineral', detection)
        self.assertIn('confidence', detection)
        self.assertGreaterEqual(detection['confidence'], 0)
        self.assertLessEqual(detection['confidence'], 1)
    
    def test_formation_analysis(self):
        analyzer = FormationModel()
        test_image = np.random.rand(512, 512, 3).astype(np.float32)
        
        analysis = analyzer.analyze_formation(test_image)
        self.assertIn('formation_type', analysis)
        self.assertIn('segmentation_map', analysis)
        self.assertIn('confidence', analysis)

if __name__ == '__main__':
    unittest.main()