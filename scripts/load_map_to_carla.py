import carla

# 1. Connect to your remote CARLA server
client = carla.Client('10.243.27.4', 2000)
client.set_timeout(10.0) # Set to 10 seconds just for this test

# ADD THIS LINE: Ask the server what version it is running
print(f"Connected to CARLA Server Version: {client.get_server_version()}")

# 2. Read your generated OpenDRIVE file
xodr_path = r"data\map\small.xodr"
with open(xodr_path, 'r', encoding='utf-8') as f:
    xodr_xml = f.read()

print("Sending map to CARLA server. This may take a minute for Munich...")

# 3. Use the CARLA C++ engine to generate and smooth the world
world = client.generate_opendrive_world(
    xodr_xml, 
    carla.OpendriveGenerationParameters(
        vertex_distance=2.0,
        max_road_length=500.0,
        wall_height=0.0,
        additional_width=0.6,
        smooth_junctions=True, # This is the magic fix!
        enable_mesh_visibility=True
    )
)

print("Map successfully loaded into CARLA!")