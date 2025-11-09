import numpy as np
from sklearn.ensemble import RandomForestRegressor

class ResourceEstimator:
    def __init__(self):
        self.config = Config()
        self.estimation_model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    def estimate_resources(self, mineral_potentials, geological_data, area_data):
        estimates = {}
        
        for mineral, potential in mineral_potentials.items():
            estimate = self._estimate_single_resource(mineral, potential, geological_data, area_data)
            estimates[mineral] = estimate
        
        return estimates
    
    def _estimate_single_resource(self, mineral, potential, geological_data, area_data):
        base_potential = potential.get('potential', 0)
        
        area = area_data.get('area', 1)
        depth = area_data.get('depth', 100)
        
        concentration_factors = {
            'copper': 0.01,
            'gold': 0.0001,
            'iron': 0.1,
            'silver': 0.001,
            'zinc': 0.02,
            'lead': 0.015,
            'nickel': 0.005,
            'uranium': 0.0005
        }
        
        density_factors = {
            'copper': 8.9,
            'gold': 19.3,
            'iron': 7.8,
            'silver': 10.5,
            'zinc': 7.1,
            'lead': 11.3,
            'nickel': 8.9,
            'uranium': 19.1
        }
        
        concentration = base_potential * concentration_factors.get(mineral, 0.01)
        density = density_factors.get(mineral, 5.0)
        
        volume = area * depth
        mass = volume * density * concentration
        
        ore_grade = concentration * 100
        
        confidence = self._calculate_estimation_confidence(base_potential, area, geological_data)
        
        return {
            'estimated_mass_kg': mass,
            'estimated_volume_m3': volume,
            'average_grade_percent': ore_grade,
            'concentration': concentration,
            'confidence': confidence,
            'economic_viability': self._assess_economic_viability(mineral, ore_grade, mass)
        }
    
    def _calculate_estimation_confidence(self, potential, area, geological_data):
        base_confidence = potential
        
        if geological_data and 'geodataframe' in geological_data:
            gdf = geological_data['geodataframe']
            data_points = len(gdf)
            base_confidence *= min(data_points / 100, 1.0)
        
        area_factor = min(area / 1000000, 1.0)
        base_confidence *= area_factor
        
        return base_confidence
    
    def _assess_economic_viability(self, mineral, grade, mass):
        grade_thresholds = {
            'copper': 0.5,
            'gold': 0.0001,
            'iron': 20.0,
            'silver': 0.01,
            'zinc': 4.0,
            'lead': 3.0,
            'nickel': 1.0,
            'uranium': 0.05
        }
        
        mass_thresholds = {
            'copper': 1000000,
            'gold': 1000,
            'iron': 10000000,
            'silver': 10000,
            'zinc': 500000,
            'lead': 500000,
            'nickel': 100000,
            'uranium': 10000
        }
        
        grade_viable = grade >= grade_thresholds.get(mineral, 1.0)
        mass_viable = mass >= mass_thresholds.get(mineral, 1000000)
        
        if grade_viable and mass_viable:
            return 'highly_viable'
        elif grade_viable or mass_viable:
            return 'potentially_viable'
        else:
            return 'not_viable'
    
    def calculate_reserve_categories(self, estimates, confidence_threshold=0.7):
        reserves = {
            'proven': {},
            'probable': {},
            'possible': {}
        }
        
        for mineral, estimate in estimates.items():
            confidence = estimate['confidence']
            mass = estimate['estimated_mass_kg']
            
            if confidence >= confidence_threshold:
                reserves['proven'][mineral] = mass * 0.7
                reserves['probable'][mineral] = mass * 0.2
                reserves['possible'][mineral] = mass * 0.1
            elif confidence >= 0.5:
                reserves['probable'][mineral] = mass * 0.6
                reserves['possible'][mineral] = mass * 0.4
            else:
                reserves['possible'][mineral] = mass
        
        return reserves
    
    def optimize_mining_plan(self, estimates, terrain_data, constraints=None):
        if constraints is None:
            constraints = {}
        
        mining_plan = {}
        
        for mineral, estimate in estimates.items():
            if estimate['economic_viability'] in ['highly_viable', 'potentially_viable']:
                plan = self._create_mineral_mining_plan(mineral, estimate, terrain_data, constraints)
                mining_plan[mineral] = plan
        
        return mining_plan
    
    def _create_mineral_mining_plan(self, mineral, estimate, terrain_data, constraints):
        mass = estimate['estimated_mass_kg']
        grade = estimate['average_grade_percent']
        
        daily_production = min(mass * 0.001, 100000)
        
        mine_life_years = mass / (daily_production * 365)
        
        slope_constraint = terrain_data.get('mean_slope', 15)
        if slope_constraint > 30:
            mining_method = 'underground'
        else:
            mining_method = 'open_pit'
        
        infrastructure_requirements = self._determine_infrastructure(mining_method, daily_production)
        
        return {
            'mining_method': mining_method,
            'estimated_mine_life_years': mine_life_years,
            'daily_production_kg': daily_production,
            'infrastructure_requirements': infrastructure_requirements,
            'capital_cost_estimate': self._estimate_capital_cost(mining_method, daily_production),
            'operating_cost_estimate': self._estimate_operating_cost(mining_method, daily_production, grade)
        }
    
    def _determine_infrastructure(self, mining_method, production):
        infrastructure = []
        
        if mining_method == 'open_pit':
            infrastructure.extend(['excavators', 'haul_trucks', 'crushing_plant'])
        else:
            infrastructure.extend(['shaft', 'underground_equipment', 'ventilation'])
        
        if production > 50000:
            infrastructure.extend(['processing_plant', 'tailings_storage', 'power_supply'])
        
        return infrastructure
    
    def _estimate_capital_cost(self, mining_method, production):
        base_cost = production * 10
        
        if mining_method == 'underground':
            base_cost *= 1.5
        
        return base_cost
    
    def _estimate_operating_cost(self, mining_method, production, grade):
        base_cost = production * 0.1
        
        if mining_method == 'underground':
            base_cost *= 1.3
        
        if grade < 1.0:
            base_cost /= grade
        
        return base_cost