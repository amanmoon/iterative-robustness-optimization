import carla
import random
import time
import os
import queue
import math

# --- CONFIGURATION ---
OUTPUT_DIR = "dataset_highway_multilane_test"
COUNTS = [2]           
MAP_NAME = 'Town04'          

IMAGES_PER_WEATHER = 500
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
IMAGE_SIZE = (1920, 1080)
FOV = 90                     

def is_same_direction(wp1, wp2):
    """
    Check if two waypoints are in lanes going the same direction.
    In OpenDRIVE, same sign of lane_id means same direction.
    """
    if wp1 is None or wp2 is None:
        return False
    return (wp1.lane_id * wp2.lane_id) > 0

def is_location_multilane(world, location):
    """
    Checks if the location has at least one valid adjacent lane 
    going in the SAME direction.
    """
    carla_map = world.get_map()
    wp = carla_map.get_waypoint(location)

    r_lane = wp.get_right_lane()
    if r_lane and r_lane.lane_type == carla.LaneType.Driving:
        if is_same_direction(wp, r_lane):
            return True
    
    l_lane = wp.get_left_lane()
    if l_lane and l_lane.lane_type == carla.LaneType.Driving:
        if is_same_direction(wp, l_lane):
            return True
        
    return False

def get_spawn_transforms(world, start_location, target_count):
    carla_map = world.get_map()
    start_wp = carla_map.get_waypoint(start_location)
    spawn_transforms = []

    if target_count >= 1:
        next_wps = start_wp.next(random.randint(10, 16))
        if next_wps:
            spawn_transforms.append(next_wps[0].transform)

    if target_count >= 2:
        valid_side_lane = None

        r_lane = start_wp.get_right_lane()
        if r_lane and r_lane.lane_type == carla.LaneType.Driving and is_same_direction(start_wp, r_lane):
            valid_side_lane = r_lane

        if valid_side_lane is None:
            l_lane = start_wp.get_left_lane()
            if l_lane and l_lane.lane_type == carla.LaneType.Driving and is_same_direction(start_wp, l_lane):
                valid_side_lane = l_lane

        if valid_side_lane:
            next_side_wps = valid_side_lane.next(random.randint(10, 18))
            if next_side_wps:
                spawn_transforms.append(next_side_wps[0].transform)
        else:
            return [] 

    return spawn_transforms

def cleanup_actors(actor_list):
    for actor in actor_list:
        if actor and actor.is_alive:
            actor.destroy()

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(20.0)

    print(f"!!! RELOADING MAP: {MAP_NAME} (Highway) !!!")
    world = client.load_world(MAP_NAME) 
    
    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("--- Starting Generation with Weather Variations ---")

    try:
        for target_count in COUNTS:
            for weather_name, weather_param in WEATHERS.items():
                
                save_folder = os.path.join(OUTPUT_DIR, f"{target_count}_cars", weather_name)
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                
                print(f"\n>>> Configuration: {target_count} CARS | Weather: {weather_name}")
                world.set_weather(weather_param)
                
                for _ in range(10): world.tick()

                count_generated = 0
                
                while count_generated < IMAGES_PER_WEATHER:
                    actor_list = []
                    try:
                        # --- Pre-Check & Spawn Ego Vehicle ---
                        ego_bp = bp_lib.find('vehicle.tesla.model3')
                        ego_transform = random.choice(spawn_points)

                        if target_count == 2:
                            if not is_location_multilane(world, ego_transform.location):
                                continue 

                        ego_vehicle = world.try_spawn_actor(ego_bp, ego_transform)
                        if not ego_vehicle: continue 
                        actor_list.append(ego_vehicle)

                        # --- Traffic Logic ---
                        spawn_locations = []
                        if target_count > 0:
                            spawn_locations = get_spawn_transforms(world, ego_transform.location, target_count)
                            
                            if len(spawn_locations) < target_count:
                                cleanup_actors(actor_list)
                                continue 

                        # --- Spawn Traffic (WITH FILTER) ---
                        traffic_bps = bp_lib.filter('vehicle.*')
                        
                        if target_count == 2:
                            traffic_bps = [x for x in traffic_bps if int(x.get_attribute('number_of_wheels')) == 4]
                            
                            excluded_keywords = [
                                'truck', 'bus', 'van', 'sprinter', 'carlacola', 
                                'firetruck', 'ambulance', 'cybertruck'
                            ]

                            traffic_bps = [x for x in traffic_bps if not any(keyword in x.id for keyword in excluded_keywords)]

                        for tf in spawn_locations:
                            bp = random.choice(traffic_bps)
                            tf.location.z += 0.2 
                            car_actor = world.try_spawn_actor(bp, tf)
                            if car_actor:
                                car_actor.set_light_state(carla.VehicleLightState.All) 
                                actor_list.append(car_actor)
                            else:
                                raise RuntimeError("Failed to spawn.")

                        camera_bp = bp_lib.find('sensor.camera.rgb')
                        camera_bp.set_attribute('image_size_x', str(IMAGE_SIZE[0]))
                        camera_bp.set_attribute('image_size_y', str(IMAGE_SIZE[1]))
                        camera_bp.set_attribute('fov', str(FOV))
                        
                        cam_loc = carla.Location(x=1.5, z=1.4)
                        cam_rot = carla.Rotation(pitch=-5.0) 
                        cam_tf = carla.Transform(cam_loc, cam_rot)
                        
                        camera = world.spawn_actor(camera_bp, cam_tf, attach_to=ego_vehicle)
                        actor_list.append(camera)
                        
                        q = queue.Queue()
                        camera.listen(q.put)

                        for _ in range(10): world.tick()
                        while not q.empty(): q.get()
                        
                        world.tick()
                        image = q.get()
                        
                        filename = f"img_{count_generated:04d}.png"
                        full_path = os.path.join(save_folder, filename)
                        image.save_to_disk(full_path)
                        
                        if count_generated % 50 == 0:
                            print(f"   Saved [{count_generated}/{IMAGES_PER_WEATHER}] to {weather_name}")
                        
                        count_generated += 1

                    except RuntimeError:
                        continue
                    finally:
                        cleanup_actors(actor_list)

    finally:
        settings.synchronous_mode = False
        world.apply_settings(settings)
        print("Done.")

if __name__ == "__main__":
    main()