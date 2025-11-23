import carla
import queue
import numpy as np
import itertools
from PIL import Image, ImageEnhance

# --- CONFIGURATION ---
WEATHERS = {
    "Default": carla.WeatherParameters.Default,
    "ClearNoon": carla.WeatherParameters.ClearNoon,
    "CloudyNoon": carla.WeatherParameters.CloudyNoon,
    "WetNoon": carla.WeatherParameters.WetNoon,
    "WetCloudyNoon": carla.WeatherParameters.WetCloudyNoon,
    "MidRainyNoon": carla.WeatherParameters.MidRainyNoon,
    "HardRainNoon": carla.WeatherParameters.HardRainNoon,
    "SoftRainNoon": carla.WeatherParameters.SoftRainNoon,
    "ClearSunset": carla.WeatherParameters.ClearSunset,
    "CloudySunset": carla.WeatherParameters.CloudySunset,
    "WetSunset": carla.WeatherParameters.WetSunset,
    "WetCloudySunset": carla.WeatherParameters.WetCloudySunset,
    "MidRainSunset": carla.WeatherParameters.MidRainSunset,
    "HardRainSunset": carla.WeatherParameters.HardRainSunset,
    "SoftRainSunset": carla.WeatherParameters.SoftRainSunset,
    "ClearNight": carla.WeatherParameters.ClearNight,
    "CloudyNight": carla.WeatherParameters.CloudyNight,
    "WetNight": carla.WeatherParameters.WetNight,
    "WetCloudyNight": carla.WeatherParameters.WetCloudyNight,
    "SoftRainNight": carla.WeatherParameters.SoftRainNight,
    "MidRainyNight": carla.WeatherParameters.MidRainyNight,
    "HardRainNight": carla.WeatherParameters.HardRainNight,
    "DustStorm": carla.WeatherParameters.DustStorm,
}
WEATHER_LIST = list(WEATHERS.values())
IMAGE_SIZE = (1920, 1080)
FOV = 90

# --- CONSTANTS ---
CAR1_LANE_ORDERS = list(itertools.permutations([-1, 0, 1])) 
CAR2_LANE_ORDERS = list(itertools.permutations([-1, 1]))

# --- HELPERS ---

def get_scalar(val):
    if hasattr(val, '__len__') and not isinstance(val, (str, bytes)):
        if len(val) > 0: return val[0]
    return val

def cleanup_actors(actor_list):
    """Cleans up the actors created by THIS script execution."""
    for actor in actor_list:
        if actor and actor.is_alive: actor.destroy()

def clean_world_ghosts(world):
    """
    CRITICAL FIX: Destroys ALL vehicles/sensors in the world.
    This prevents 'ghost' cars from previous runs from cluttering the image.
    """
    # clear vehicles
    for actor in world.get_actors().filter('vehicle.*'):
        if actor.is_alive: actor.destroy()
    # clear sensors (cameras)
    for actor in world.get_actors().filter('sensor.*'):
        if actor.is_alive: actor.destroy()

def is_same_direction(wp1, wp2):
    if wp1 is None or wp2 is None: return False
    return (wp1.lane_id * wp2.lane_id) > 0

def check_alignment(transform, waypoint):
    if not waypoint: return False
    tf_yaw = transform.rotation.yaw
    wp_yaw = waypoint.transform.rotation.yaw
    diff = abs(tf_yaw - wp_yaw) % 360
    return not (90 < diff < 270)

def get_lane_waypoint(start_wp, lane_offset):
    if lane_offset == 0: return start_wp
    
    target_lane = None
    if lane_offset == -1: target_lane = start_wp.get_left_lane()
    elif lane_offset == 1: target_lane = start_wp.get_right_lane()
        
    if (target_lane and 
        target_lane.lane_type == carla.LaneType.Driving and 
        is_same_direction(start_wp, target_lane)):
        return target_lane
    return None

def check_location_suitability(world, location, required_lanes=1):
    wp = world.get_map().get_waypoint(location)
    if required_lanes < 2: return True
    
    l = wp.get_left_lane()
    if l and l.lane_type == carla.LaneType.Driving and is_same_direction(wp, l): return True
    r = wp.get_right_lane()
    if r and r.lane_type == carla.LaneType.Driving and is_same_direction(wp, r): return True
    return False

def try_spawn_lowest_z(world, blueprint, transform, start_z=0.10, max_z=0.5, step=0.05):
    """
    Attempts to spawn an actor starting at start_z. 
    Increased start_z to 0.10 to be safer against ground collision.
    """
    base_z = transform.location.z
    
    for z_offset in np.arange(start_z, max_z, step):
        transform.location.z = base_z + z_offset
        actor = world.try_spawn_actor(blueprint, transform)
        if actor is not None:
            return actor
    return None

# --- GENERATOR ---

def genImage(sample):
    
    client = carla.Client('localhost', 2000)
    client.set_timeout(20.0)
    world = client.get_world()
    
    if 'Town04' not in world.get_map().name:
        world = client.load_world('Town04')
    
    clean_world_ghosts(world)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    spawn_points.sort(key=lambda p: (p.location.x, p.location.y))

    traffic_bps = bp_lib.filter('vehicle.*')
    traffic_bps = [x for x in traffic_bps if int(x.get_attribute('number_of_wheels')) == 4]
    excluded_keywords = [
                    'truck', 'bus', 'van', 'sprinter', 'carlacola', 
                    'firetruck', 'ambulance', 'cybertruck'
                                ] 
    traffic_bps = [x for x in traffic_bps if not any(keyword in x.id for keyword in excluded_keywords)]
    traffic_bps = sorted(traffic_bps, key=lambda x: x.id)

    actor_list = []
    generated_pil_image = None

    try:
        # 1. Setup
        weather_id = int(get_scalar(sample.weatherID))
        num_cars = int(get_scalar(sample.numCars))
        
        if 0 <= weather_id < len(WEATHER_LIST):
            world.set_weather(WEATHER_LIST[weather_id])
        else:
            world.set_weather(carla.WeatherParameters.Default)
        world.tick()

        try:
            town_loc_val = float(get_scalar(sample.townLocation))
            start_idx = int(town_loc_val * (len(spawn_points) - 1))
        except:
            start_idx = 0

        success = False

        # 2. Search for valid geometry
        for offset in range(100):
            current_idx = (start_idx + offset) % len(spawn_points)
            cleanup_actors(actor_list)
            actor_list = []

            try:
                raw_tf = spawn_points[current_idx]
                ego_tf = carla.Transform(
                    carla.Location(raw_tf.location.x, raw_tf.location.y, raw_tf.location.z),
                    carla.Rotation(raw_tf.rotation.pitch, raw_tf.rotation.yaw, raw_tf.rotation.roll)
                )

                ego_wp = world.get_map().get_waypoint(ego_tf.location)
                
                if not check_alignment(ego_tf, ego_wp):
                    found_correct = False
                    for lane_check in [ego_wp.get_left_lane(), ego_wp.get_right_lane()]:
                        if lane_check and check_alignment(ego_tf, lane_check):
                            ego_wp = lane_check
                            found_correct = True
                            break
                    if not found_correct: continue

                # Multi-lane check
                if num_cars == 2:
                    if not check_location_suitability(world, ego_tf.location, 2):
                        continue

                # --- A. Spawn Ego ---
                ego_bp = bp_lib.find('vehicle.tesla.model3')
                
                ego_vehicle = try_spawn_lowest_z(world, ego_bp, ego_tf)
                
                if not ego_vehicle: continue

                ego_vehicle.set_simulate_physics(False) 
                actor_list.append(ego_vehicle)
                
                car1_actual_offset = None

                # --- B. Spawn Car 1 ---
                if num_cars >= 1:
                    c1_data = sample.car1
                    dist_c1 = float(get_scalar(c1_data.distance))
                    order_id = int(get_scalar(c1_data.lane_order))
                    pref_list = CAR1_LANE_ORDERS[order_id % len(CAR1_LANE_ORDERS)]

                    spawned_c1 = False
                    for lane_off in pref_list:
                        target_lane_wp = get_lane_waypoint(ego_wp, lane_off)
                        if not target_lane_wp: continue
                        
                        next_wps = target_lane_wp.next(dist_c1)
                        if not next_wps: continue

                        bp = traffic_bps[int(get_scalar(c1_data.carID)) % len(traffic_bps)]
                        tf = next_wps[0].transform
                        
                        car = try_spawn_lowest_z(world, bp, tf)
                        if car:
                            car.set_simulate_physics(False)
                            car.set_light_state(carla.VehicleLightState.All)
                            actor_list.append(car)
                            spawned_c1 = True
                            car1_actual_offset = lane_off
                            break
                    
                    if not spawned_c1: 
                        raise RuntimeError("Car 1 fail")

                # --- C. Spawn Car 2 ---
                if num_cars == 2:
                    if car1_actual_offset is None: raise RuntimeError("Car 1 missing")

                    c2_data = sample.car2
                    dist_c2 = float(get_scalar(c2_data.distance))
                    order_id_2 = int(get_scalar(c2_data.lane_order))
                    pref_list_c2 = CAR2_LANE_ORDERS[order_id_2 % len(CAR2_LANE_ORDERS)]
                    
                    spawned_c2 = False
                    for rel_offset in pref_list_c2:
                        abs_offset = car1_actual_offset + rel_offset
                        
                        target_lane_wp = get_lane_waypoint(ego_wp, abs_offset)
                        if not target_lane_wp: continue
                        
                        next_wps = target_lane_wp.next(dist_c2)
                        if not next_wps: continue
                        
                        bp = traffic_bps[int(get_scalar(c2_data.carID)) % len(traffic_bps)]
                        tf = next_wps[0].transform
                        
                        # Use helper to spawn
                        car = try_spawn_lowest_z(world, bp, tf)
                        if car:
                            car.set_simulate_physics(False)
                            car.set_light_state(carla.VehicleLightState.All)
                            actor_list.append(car)
                            spawned_c2 = True
                            break
                    
                    if not spawned_c2: raise RuntimeError("Car 2 fail")

                # --- D. Capture ---
                camera_bp = bp_lib.find('sensor.camera.rgb')
                camera_bp.set_attribute('image_size_x', str(IMAGE_SIZE[0]))
                camera_bp.set_attribute('image_size_y', str(IMAGE_SIZE[1]))
                camera_bp.set_attribute('fov', str(FOV))
                
                cam_data = sample.camera
                cam_tf = carla.Transform(
                    carla.Location(x=float(get_scalar(cam_data.x)), z=float(get_scalar(cam_data.z))),
                    carla.Rotation(pitch=float(get_scalar(cam_data.pitch)))
                )
                camera = world.spawn_actor(camera_bp, cam_tf, attach_to=ego_vehicle)
                actor_list.append(camera)
                
                q = queue.Queue()
                camera.listen(q.put)
                
                for _ in range(15): world.tick()

                carla_image = q.get(timeout=2.0)
                array = np.frombuffer(carla_image.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (carla_image.height, carla_image.width, 4))
                generated_pil_image = Image.fromarray(array[:, :, :3][:, :, ::-1])
                
                success = True
                break

            except RuntimeError as e:
                continue

        if not success:
            print("Failed generation.")
            return None

    finally:
        settings.synchronous_mode = False
        world.apply_settings(settings)
        cleanup_actors(actor_list)

    if generated_pil_image:
        try:
            generated_pil_image = ImageEnhance.Brightness(generated_pil_image).enhance(get_scalar(sample.brightness))
            generated_pil_image = ImageEnhance.Contrast(generated_pil_image).enhance(get_scalar(sample.contrast))
            generated_pil_image = ImageEnhance.Sharpness(generated_pil_image).enhance(get_scalar(sample.sharpness))
            generated_pil_image = ImageEnhance.Color(generated_pil_image).enhance(get_scalar(sample.color))
        except: pass

    return generated_pil_image