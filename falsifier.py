from verifai.features.features import *
from verifai.samplers.feature_sampler import *
from verifai.falsifier import generic_falsifier
from verifai.monitor import specification_monitor
from dotmap import DotMap

CONFIDENCE_THRESHOLD = 0.8
MAX_ITERS = 1000
PORT = 8888
MAXREQS = 5
BUFSIZE = 4096

class confidence_spec(specification_monitor):
    def __init__(self):
        def specification(traj):
            # return bool(traj['yTrue'] == traj['yPred'])
            return traj['confidence'] - CONFIDENCE_THRESHOLD
        super().__init__(specification)

car1_domain = Struct({
    'distance': Box([8, 16]),  
    'lane_order': Categorical(*np.arange(0, 6)), 
    'carID': Categorical(*np.arange(0, 27))
})

car2_domain = Struct({
    'distance': Box([8, 16]), 
    'lane_order': Categorical(*np.arange(0, 2)), 
    'carID': Categorical(*np.arange(0, 27))
})

camera_domain = Struct({
    'x': Box([0.5, 1.5]),   
    'z': Box([1.4, 2.5]),   
    'pitch': Box([-20, 0])   
})

space = FeatureSpace({
    'weatherID': Feature(Categorical(*np.arange(0, 24))),
    
    'numCars': Feature(Categorical(*np.arange(0, 3))),

    'townLocation': Feature(Box([0, 1])),

    'car1': Feature(car1_domain),
    'car2': Feature(car2_domain),
    'camera': Feature(camera_domain),
    
    'brightness': Feature(Box([0.5, 1])),
    'sharpness': Feature(Box([0, 1])),
    'contrast': Feature(Box([0.5, 1.5])),
    'color': Feature(Box([0, 1]))
})

sampler = FeatureSampler.crossEntropySamplerFor(space)

falsifier_params = DotMap(n_iters=MAX_ITERS,
                          compute_error_table=True,
                          fal_thres=0.5,
                          verbosity=1)

server_options = DotMap(port=PORT, bufsize=BUFSIZE, maxreqs=MAXREQS)

falsifier = generic_falsifier(sampler=sampler, server_options=server_options,
                             monitor=confidence_spec(), falsifier_params=falsifier_params)

falsifier.run_falsifier()

analysis_params = DotMap()
falsifier.analyze_error_table(analysis_params=analysis_params)

print("Error table")
print(falsifier.error_table.table)
