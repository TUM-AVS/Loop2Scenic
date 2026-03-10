## 1. SET MAP AND MODEL
# Point Scenic to your custom OpenDRIVE file
param map = 'data/map/hockbruck.xodr'

# Setting carla_map to None tells CARLA to auto-generate the 3D road mesh
param carla_map = None 

# Import the CARLA driving domain
model scenic.simulators.carla.model

## 2. SPAWN ACTORS
# Spawn an ego vehicle and tell it to simply drive forward following the lane
ego = new Car with behavior FollowLaneBehavior()

# Spawn another car 20 meters ahead of the ego vehicle to make it a "traffic" scene
lead_car = new Car ahead of ego by 20, 
    with behavior FollowLaneBehavior()

## 3. TERMINATION CONDITION
# End the simulation after 30 seconds
terminate after 30 seconds