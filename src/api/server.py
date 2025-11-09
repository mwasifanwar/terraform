from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import tempfile
import os

app = FastAPI(title="TerraForm API", version="1.0.0")

class SatelliteAnalysisRequest(BaseModel):
    image_path: str
    coordinates: Optional[List[float]] = None
    analysis_types: List[str] = ["mineral", "formation", "disaster"]

class MineralDetectionRequest(BaseModel):
    image_path: str
    spectral_bands: Optional[List[List[float]]] = None
    confidence_threshold: float = 0.7

class DisasterPredictionRequest(BaseModel):
    geological_data: Dict[str, Any]
    terrain_data: Dict[str, Any]
    historical_data: Optional[Dict[str, Any]] = None

class ResourceEstimationRequest(BaseModel):
    mineral_potentials: Dict[str, float]
    area_data: Dict[str, float]
    geological_data: Dict[str, Any]

@app.post("/analyze_satellite_imagery")
async def analyze_satellite_imagery(request: SatelliteAnalysisRequest):
    try:
        from src.data_processing.satellite_processor import SatelliteProcessor
        from src.computer_vision.mineral_detector import MineralDetectionModel
        from src.computer_vision.formation_analyzer import FormationModel
        from src.computer_vision.disaster_predictor import DisasterModel
        
        processor = SatelliteProcessor()
        satellite_data = processor.load_satellite_image(request.image_path)
        
        results = {
            'satellite_data': {
                'bands_shape': satellite_data['bands'].shape if satellite_data['bands'] is not None else None,
                'bounds': satellite_data['bounds'],
                'crs': str(satellite_data['crs'])
            },
            'analyses': {}
        }
        
        if "mineral" in request.analysis_types:
            mineral_detector = MineralDetectionModel()
            rgb_image = processor.extract_rgb_composite(satellite_data['bands'])
            mineral_detections = mineral_detector.detect_minerals_in_region({
                'rgb': rgb_image,
                'bands': satellite_data['bands']
            })
            results['analyses']['mineral_detection'] = mineral_detections
        
        if "formation" in request.analysis_types:
            formation_analyzer = FormationModel()
            rgb_image = processor.extract_rgb_composite(satellite_data['bands'])
            formation_analysis = formation_analyzer.analyze_formation(rgb_image)
            results['analyses']['formation_analysis'] = formation_analysis
        
        if "disaster" in request.analysis_types and request.coordinates:
            disaster_predictor = DisasterModel()
            risk_assessment = disaster_predictor.predict_disaster_risk({}, {}, {})
            results['analyses']['disaster_risk'] = risk_assessment
        
        return {
            "status": "success",
            "results": results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect_minerals")
async def detect_minerals(request: MineralDetectionRequest):
    try:
        from src.computer_vision.mineral_detector import MineralDetectionModel
        
        detector = MineralDetectionModel()
        
        if request.spectral_bands:
            import numpy as np
            spectral_bands = np.array(request.spectral_bands)
        else:
            spectral_bands = np.random.rand(13, 224, 224)
        
        import cv2
        image = cv2.imread(request.image_path)
        if image is None:
            raise HTTPException(status_code=400, detail="Could not load image")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        detection = detector.detect_minerals(image, spectral_bands)
        
        return {
            "status": "success",
            "detection": detection,
            "confidence_threshold": request.confidence_threshold
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_disaster_risk")
async def predict_disaster_risk(request: DisasterPredictionRequest):
    try:
        from src.computer_vision.disaster_predictor import DisasterModel
        
        predictor = DisasterModel()
        risk_assessment = predictor.predict_disaster_risk(
            request.geological_data,
            request.terrain_data,
            request.historical_data
        )
        
        return {
            "status": "success",
            "risk_assessment": risk_assessment
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/estimate_resources")
async def estimate_resources(request: ResourceEstimationRequest):
    try:
        from src.geological_models.resource_estimator import ResourceEstimator
        
        estimator = ResourceEstimator()
        estimates = estimator.estimate_resources(
            request.mineral_potentials,
            request.geological_data,
            request.area_data
        )
        
        reserves = estimator.calculate_reserve_categories(estimates)
        
        return {
            "status": "success",
            "resource_estimates": estimates,
            "reserve_categories": reserves
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_geological_data")
async def upload_geological_data(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.shp') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        from src.data_processing.geological_mapper import GeologicalMapper
        mapper = GeologicalMapper()
        geological_data = mapper.load_geological_map(temp_path)
        
        os.unlink(temp_path)
        
        return {
            "status": "success",
            "geological_data": {
                "feature_count": len(geological_data['geodataframe']) if geological_data['geodataframe'] is not None else 0,
                "bounds": geological_data['bounds'].tolist() if geological_data['bounds'] is not None else None,
                "crs": str(geological_data['crs'])
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "TerraForm"}

if __name__ == "__main__":
    import uvicorn
    from src.utils.config import Config
    config = Config()
    uvicorn.run(app, host=config.get('api.host'), port=config.get('api.port'))