"""
Helper utility functions.
"""

import logging
from pathlib import Path
from typing import Optional

import tiktoken

logger = logging.getLogger(__name__)


def ensure_directory(directory: str) -> Path:
    """
    Ensure a directory exists, create if it doesn't.
    
    Args:
        directory: Directory path
        
    Returns:
        Path object
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Count the number of tokens in a text string.
    
    Args:
        text: Text to count tokens for
        model: Model name for tokenizer
        
    Returns:
        Number of tokens
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    num_tokens = len(encoding.encode(text))
    return num_tokens


def truncate_text(
    text: str,
    max_tokens: int,
    model: str = "gpt-3.5-turbo",
    suffix: str = "..."
) -> str:
    """
    Truncate text to a maximum number of tokens.
    
    Args:
        text: Text to truncate
        max_tokens: Maximum number of tokens
        model: Model name for tokenizer
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    tokens = encoding.encode(text)
    
    if len(tokens) <= max_tokens:
        return text
    
    # Truncate and decode
    truncated_tokens = tokens[:max_tokens - len(encoding.encode(suffix))]
    truncated_text = encoding.decode(truncated_tokens) + suffix
    
    return truncated_text


def format_metadata(metadata: dict, indent: int = 2) -> str:
    """
    Format metadata dictionary as readable string.
    
    Args:
        metadata: Metadata dictionary
        indent: Indentation spaces
        
    Returns:
        Formatted string
    """
    lines = []
    for key, value in metadata.items():
        lines.append(f"{' ' * indent}{key}: {value}")
    return "\n".join(lines)

import os
import time
import subprocess
import tempfile
import shutil
import carla
import scenic
from scenic.simulators.carla.simulator import CarlaSimulator

def run_scenic_and_record_mp4(scenic_input: str, carla_exe_path: str, output_dir: str) -> bool:
    """
    Starts CARLA, loads a Scenic scenario, records a video stream via a chase camera, 
    and compiles it into an .mp4 using ffmpeg.
    """
    # Prepare directories
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, "temp_frames")
    os.makedirs(frames_dir, exist_ok=True)
    video_path = os.path.join(output_dir, "simulation_video.mp4")
    
    is_file = os.path.isfile(scenic_input)
    scenic_file_path = scenic_input if is_file else None
    
    carla_process = None
    temp_file = None
    camera = None
    success = False
    
    try:
        # 1. Start CARLA Server
        print("Starting CARLA Server...")
        carla_process = subprocess.Popen([carla_exe_path])
        
        client = carla.Client('localhost', 2000)
        client.set_timeout(2.0)
        
        # Poll server
        connected = False
        for _ in range(15):
            try:
                client.get_world()
                connected = True
                print("Connected to CARLA.")
                break
            except RuntimeError:
                time.sleep(2)
                
        if not connected:
            print("Error: Could not connect to CARLA server in time.")
            return False

        # 2. Handle Scenic input
        if not is_file:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".scenic", mode='w')
            temp_file.write(scenic_input)
            temp_file.close()
            scenic_file_path = temp_file.name
            
        # 3. Parse Scenario and Create Simulation
        print("Parsing Scenic scenario and spawning actors...")
        scenario = scenic.scenarioFromFile(scenic_file_path)
        simulator = CarlaSimulator()
        
        scene, _ = scenario.generate()
        # We use createSimulation instead of simulate so we can inject the camera!
        simulation = simulator.createSimulation(scene)
        
        # 4. Set up the Chase Camera
        world = client.get_world()
        
        # Try to find the primary vehicle to follow
        vehicles = world.get_actors().filter('vehicle.*')
        if not vehicles:
            print("Warning: No vehicles found in the scene to attach the camera to.")
            ego_vehicle = None
        else:
            # We assume the first vehicle is our main actor
            ego_vehicle = vehicles[0] 
        
        if ego_vehicle:
            cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
            cam_bp.set_attribute('image_size_x', '1280')
            cam_bp.set_attribute('image_size_y', '720')
            cam_bp.set_attribute('sensor_tick', '0.033') # Target ~30 FPS
            
            # Position camera behind and slightly above the car
            cam_transform = carla.Transform(carla.Location(x=-6.5, z=3.5), carla.Rotation(pitch=-15.0))
            camera = world.spawn_actor(cam_bp, cam_transform, attach_to=ego_vehicle)
            
            # We use a mutable dictionary to keep a sequential frame count for ffmpeg
            frame_tracker = {"count": 0}
            def save_frame(image):
                image.save_to_disk(os.path.join(frames_dir, f"{frame_tracker['count']:06d}.png"))
                frame_tracker['count'] += 1
                
            camera.listen(save_frame)
            print("Camera attached and recording started.")

        # 5. Run the Simulation
        print("Running simulation...")
        simulation.run()
        print("Simulation finished. Stopping camera...")
        
        if camera:
            camera.stop()
            camera.destroy()
            camera = None

        # 6. Stitch images to MP4 using ffmpeg
        if shutil.which("ffmpeg") is None:
            print("Error: 'ffmpeg' is not installed or not in PATH. Leaving raw frames in directory.")
        else:
            print(f"Stitching frames into {video_path}...")
            subprocess.run([
                "ffmpeg", "-y", "-framerate", "30", 
                "-i", os.path.join(frames_dir, "%06d.png"), 
                "-c:v", "libx264", "-pix_fmt", "yuv420p", 
                video_path
            ], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            
            # Clean up the thousands of .png files to save hard drive space
            shutil.rmtree(frames_dir, ignore_errors=True)
            print("Video generated successfully!")
            
        success = True
        
    except Exception as e:
        print(f"Simulation failed with error: {e}")
        success = False
        
    finally:
        # 7. Safe cleanup
        print("Cleaning up resources...")
        if camera and camera.is_alive:
            camera.stop()
            camera.destroy()
            
        if carla_process:
            carla_process.terminate()
            carla_process.wait()
            
        if temp_file and os.path.exists(temp_file.name):
            os.remove(temp_file.name)
            
    return success