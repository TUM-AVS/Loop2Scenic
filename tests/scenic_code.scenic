description = "Ego vehicle performs a lane change to avoid a stationary obstacle blocking its lane."
param map = localPath('../maps/Town05.xodr')
param carla_map = 'Town05'
model scenic.simulators.carla.model
MODEL = 'vehicle.lincoln.mkz_2017'
param weather = 'ClearNoon'

egoLane = Uniform(*network.lanes)
egoSection = Uniform(*egoLane.sections)
egoSpawnPt = new OrientedPoint in egoSection.centerline

leftSection = egoSection._laneToLeft
rightSection = egoSection._laneToRight
targetSection = Uniform(*filter(lambda s: s is not None, [egoSection, leftSection, rightSection]))

obstacleSpawnPt = new OrientedPoint in targetSection.centerline, ahead of egoSpawnPt by Range(20, 40)