import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_processing.satellite_processor import SatelliteProcessor
from src.computer_vision.mineral_detector import MineralDetectionModel
from src.computer_vision.formation_analyzer import FormationModel
from src.computer_vision.disaster_predictor import DisasterModel
from src.geological_models.mineral_predictor import MineralPredictionModel
from src.geological_models.resource_estimator import ResourceEstimator
from src.visualization.map_generator import MapGenerator
from src.api.server import app
import uvicorn

def run_demo():
    print("Running TerraForm Demo...")
    
    processor = SatelliteProcessor()
    satellite_data = processor.load_satellite_image("data/satellite_images/sample.tif")
    
    print("Satellite data loaded successfully")
    print(f"Bands shape: {satellite_data['bands'].shape}")
    
    mineral_detector = MineralDetectionModel()
    rgb_image = processor.extract_rgb_composite(satellite_data['bands'])
    mineral_detections = mineral_detector.detect_minerals_in_region({
        'rgb': rgb_image,
        'bands': satellite_data['bands']
    })
    
    print(f"Detected {len(mineral_detections)} mineral occurrences")
    
    formation_analyzer = FormationModel()
    formation_analysis = formation_analyzer.analyze_formation(rgb_image)
    print(f"Formation type: {formation_analysis['formation_type']}")
    
    disaster_predictor = DisasterModel()
    risk_assessment = disaster_predictor.predict_disaster_risk({}, {}, {})
    print(f"Overall disaster risk: {risk_assessment['overall_risk']:.2f}")
    
    map_generator = MapGenerator()
    geological_map = map_generator.create_interactive_map(40.0, -100.0, zoom_start=6)
    geological_map = map_generator.add_mineral_locations(geological_map, mineral_detections)
    
    geological_map.save("geological_analysis_map.html")
    print("Interactive map saved as geological_analysis_map.html")
    
    return {
        'mineral_detections': mineral_detections,
        'formation_analysis': formation_analysis,
        'risk_assessment': risk_assessment
    }

def run_api():
    from src.utils.config import Config
    config = Config()
    print(f"Starting TerraForm API server on {config.get('api.host')}:{config.get('api.port')}")
    uvicorn.run(app, host=config.get('api.host'), port=config.get('api.port'))

def main():
    parser = argparse.ArgumentParser(description='TerraForm: AI for Geological Exploration')
    parser.add_argument('--mode', choices=['api', 'demo', 'analyze'], default='demo', help='Operation mode')
    parser.add_argument('--image', type=str, help='Path to satellite image for analysis')
    parser.add_argument('--coordinates', nargs=2, type=float, help='Latitude and longitude for analysis')
    
    args = parser.parse_args()
    
    if args.mode == 'api':
        run_api()
    elif args.mode == 'demo':
        run_demo()
    elif args.mode == 'analyze':
        if args.image:
            processor = SatelliteProcessor()
            satellite_data = processor.load_satellite_image(args.image)
            
            mineral_detector = MineralDetectionModel()
            rgb_image = processor.extract_rgb_composite(satellite_data['bands'])
            mineral_detections = mineral_detector.detect_minerals_in_region({
                'rgb': rgb_image,
                'bands': satellite_data['bands']
            })
            
            print(f"Analysis completed - mwasifanwar")
            print(f"Found {len(mineral_detections)} mineral detections")
        else:
            print("Please provide an image path with --image argument")
    else:
        print("TerraForm system ready - mwasifanwar")

if __name__ == "__main__":
    main()