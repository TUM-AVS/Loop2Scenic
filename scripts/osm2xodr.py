import carla
import io

def convert_osm_to_xodr(osm_file_path, output_xodr_path):
    # 1. Read the .osm data (ensure utf-8 encoding)
    with io.open(osm_file_path, mode="r", encoding="utf-8") as f:
        osm_data = f.read()

    # 2. Define the conversion settings
    settings = carla.Osm2OdrSettings()
    settings.generate_traffic_lights = True
    settings.set_osm_way_types([
        "motorway", "motorway_link", "trunk", "trunk_link", 
        "primary", "primary_link", "secondary", "secondary_link", 
        "tertiary", "tertiary_link", "unclassified", "residential"
    ])

    # 3. Convert to OpenDRIVE string
    print(f"Converting {osm_file_path} to OpenDRIVE...")
    xodr_data = carla.Osm2Odr.convert(osm_data, settings)

    # 4. Save the OpenDRIVE file (FIXED: Added encoding="utf-8" here)
    with open(output_xodr_path, "w", encoding="utf-8") as f:
        f.write(xodr_data)
        
    print(f"Conversion complete! Saved to {output_xodr_path}")

# FIXED: Added 'r' before the string to handle Windows backslashes properly
convert_osm_to_xodr(r"data\map\hockbruck.osm", r"data\map\hockbruck.xodr")