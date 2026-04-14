import json
import logging
import os
import subprocess

from src.agents.scenic_coder_agent import ScenicCoderAgent
from src.schema import HeaderSetting
from src.services import get_embedder, get_llm_service
from src.utils import setup_logging


def ingest_dsl(folder_path: str = "data/prompt_opt"):
    """
    Ingest DSL from the folder path.
    """
    dsl_list = []
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    for subfolder in subfolders:
        with open(os.path.join(subfolder, "new_description.json"), "r") as f:
            dsl = json.load(f)
            scenario_id = subfolder.split("/")[-1]
            dsl_list.append({"scenario_id": scenario_id, "dsl": dsl})
    return dsl_list

def code_generation(dsl_list: list, output_folder_path: str = "data/prompt_opt_code"):
    """
    Generate code from the DSL list.
    """
    from src.services import MilvusVectorStore
    from src.config import get_config
    config = get_config()
    logger = logging.getLogger(__name__)
    setup_logging(
        level=config.logging.level,
        log_file=config.logging.file,
        log_format=config.logging.format,
    )
    logger.info("Running ScenicCoderAgent standalone module.")

    llm_service = get_llm_service(provider=config.llm.provider, model=config.llm.model, temperature=config.llm.temperature, max_tokens=config.llm.max_tokens)
    connection_args = {
        "host": config.vector_db.host,
        "port": config.vector_db.port,
    }
    index_params = {
        "metric_type": config.vector_db.distance_metric.upper(),
        "index_type": config.vector_db.index_type,
        "params": {"nlist": config.vector_db.nlist},
    }
    search_params = {
        "metric_type": config.vector_db.distance_metric.upper(),
        "params": {"nprobe": config.vector_db.nprobe},
    }

    vector_db = MilvusVectorStore(
        connection_args=connection_args,
        index_params=index_params,
        search_params=search_params,
    )
    snippets_embedder = get_embedder(provider="huggingface", model_name="sentence-transformers/all-MiniLM-L6-v2", device="cuda")
    agent = ScenicCoderAgent(llm_service, vector_db, snippets_embedder)
    os.makedirs(output_folder_path, exist_ok=True)

    header_settings = HeaderSetting(
        map_file_path="../maps/Town05.xodr",
        carla_map="Town05",
        weather="ClearNoon",
        blueprint="vehicle.lincoln.mkz_2017"
    )

    for dsl in dsl_list:
        code = agent.adapt_code(
            original_scenic_code="", 
            evaluation_result={"Adversarials": False, "Ego": False, "Requirement and restrictions": False, "Scenario": False, "Spatial Relation": False}, 
            aim_dsl=dsl["dsl"], 
            header_settings=header_settings
        )
        
        with open(os.path.join(output_folder_path, f"{dsl['scenario_id']}.scenic"), "w") as f:
            f.write(code)

def run_simulation(code_path: str = "data/prompt_opt_code", output_folder_path: str = "data/prompt_opt_simulation_result"):
    result = subprocess.run([
        'src/utils/run_scenic_batch.sh', 
        code_path, 
        '--outdir', f"{output_folder_path}/video", 
        '--logdir', f"{output_folder_path}/logs"
    ], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to run simulation: {result.stderr}")
    return result.stdout

if __name__ == "__main__":
    dsl_list = ingest_dsl()
    code_generation(dsl_list)
    run_simulation()