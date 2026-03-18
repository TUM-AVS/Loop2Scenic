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
import carla
import scenic
from scenic.simulators.carla.simulator import CarlaSimulator

def run_scenic_in_carla(scenic_input: str, carla_exe_path: str, output_dir: str) -> bool:
    """
    Starts CARLA, loads a Scenic scenario, runs it, and saves a recording.
    
    Args:
        scenic_input: Either a file path to a .scenic file, or raw Scenic code.
        carla_exe_path: Absolute path to the CarlaUE4.exe or CarlaUE4.sh executable.
        output_dir: Directory where the CARLA recording (.log) will be saved.
        
    Returns:
        bool: True if the simulation ran and completed successfully, False otherwise.
    """
    # Prepare the output directory and recording path
    os.makedirs(output_dir, exist_ok=True)
    # CARLA's recorder requires an absolute path because the server writes the file
    recording_path = os.path.abspath(os.path.join(output_dir, "simulation_record.log"))
    
    is_file = os.path.isfile(scenic_input)
    scenic_file_path = scenic_input if is_file else None
    
    carla_process = None
    temp_file = None
    success = False
    
    try:
        # 1. Start the CARLA server
        print("Starting CARLA Server...")
        carla_process = subprocess.Popen([carla_exe_path])
        
        # 2. Poll the server until it is ready
        client = carla.Client('localhost', 2000)
        client.set_timeout(2.0)
        connected = False
        
        for attempt in range(15):  # Poll for up to 30 seconds
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

        # 3. Handle the Scenic input (create a temp file if it's raw text)
        if not is_file:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".scenic", mode='w')
            temp_file.write(scenic_input)
            temp_file.close()
            scenic_file_path = temp_file.name
            
        # 4. Start the CARLA recorder
        client.start_recorder(recording_path)
        print(f"Recording started: {recording_path}")
        
        # 5. Parse and run the Scenic scenario
        print("Parsing Scenic scenario...")
        scenario = scenic.scenarioFromFile(scenic_file_path)
        
        print("Running simulation...")
        simulator = CarlaSimulator()
        
        # Generate a specific scene from the probabilistic scenario and simulate it
        scene, _ = scenario.generate()
        simulation_result = simulator.simulate(scene)
        
        # If we reach this point without exceptions, the run was successful
        success = True
        print("Simulation completed successfully.")
        
    except Exception as e:
        print(f"Simulation failed with error: {e}")
        success = False
        
    finally:
        # 6. Safe cleanup of all resources
        print("Cleaning up resources...")
        try:
            client.stop_recorder()
        except NameError:
            pass # Client was never initialized
            
        if carla_process:
            carla_process.terminate()
            carla_process.wait() # Ensure the process actually dies
            
        if temp_file and os.path.exists(temp_file.name):
            os.remove(temp_file.name)
            
    return success