import carla
import io

def convert_osm_to_xodr(osm_file_path, output_xodr_path):
    with io.open(osm_file_path, mode="r", encoding="utf-8") as f:
        osm_data = f.read()

    settings = carla.Osm2OdrSettings()
    
    # FIX 1: Shift the massive Earth coordinates so the map center is (0,0)
    # This prevents the scipy math solver in Scenic from breaking!
    settings.center_map = True 
    
    # FIX 2: Disable traffic lights for now to prevent broken stop-line geometries
    settings.generate_traffic_lights = False
    
    settings.set_osm_way_types([
        "motorway", "motorway_link", "trunk", "trunk_link", 
        "primary", "primary_link", "secondary", "secondary_link", 
        "tertiary", "tertiary_link", "unclassified", "residential"
    ])

    print(f"Converting {osm_file_path} to OpenDRIVE...")
    xodr_data = carla.Osm2Odr.convert(osm_data, settings)

    with open(output_xodr_path, "w", encoding="utf-8") as f:
        f.write(xodr_data)
        
    print(f"Conversion complete! Saved to {output_xodr_path}")

# FIXED: Added 'r' before the string to handle Windows backslashes properly
convert_osm_to_xodr(r"data\map\hockbruck.osm", r"data\map\hockbruck.xodr")