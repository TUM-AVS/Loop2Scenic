import os
from crdesigner.map_conversion.commonroad_to_opendrive.converter import cr2xodr
from commonroad.common.file_reader import CommonRoadFileReader

def convert_commonroad_to_xodr(input_xml_path, output_xodr_path):
    """
    Converts a CommonRoad .xml map into an OpenDRIVE .xodr file
    using the Scenario Designer conversion engine.
    """
    if not os.path.exists(input_xml_path):
        print(f"Error: Input file not found at {input_xml_path}")
        return

    print(f"--- Starting Conversion ---")
    print(f"Input: {input_xml_path}")

    try:
        # 1. Load the CommonRoad scenario using the standard File Reader
        # This preserves the Lanelet IDs and geometric relations
        scenario, _ = CommonRoadFileReader(input_xml_path).open()
        print(f"Successfully loaded Scenario ID: {scenario.scenario_id}")

        # 2. Convert Lanelets to OpenDRIVE Reference Lines
        # This step uses the Scenario Designer's internal math to create 
        # the smooth C2-continuous curves Scenic requires.
        print("Calculating parametric road geometry (this may take a moment)...")
        xodr_xml = cr2xodr(scenario)

        # 3. Write the output file
        with open(output_xodr_path, "w", encoding="utf-8") as f:
            f.write(xodr_xml)
        
        print(f"--- Conversion Successful ---")
        print(f"Output saved to: {output_xodr_path}")

    except Exception as e:
        print(f"An error occurred during conversion: {e}")

def main():
    # --- CONFIGURATION AREA ---
    # Update these paths to match your Munich data locations
    input_path = "data/map/munich_tum_area.xml"
    output_path = "data/map/munich_cleaned.xodr"
    # ---------------------------

    # Execute the conversion
    convert_commonroad_to_xodr(input_path, output_path)

if __name__ == "__main__":
    main()