"""
Main application entry point for ADS-MRAG system.

This module initializes the entire system using configuration from config.yaml.
"""

import logging
from typing import List, Optional, Tuple
import uuid
import gradio as gr

from src.config import get_config
from src.schema import MultimodalQuery
from src.utils.logger import setup_logging

# Import Services
from src.services import (
    get_llm_service,
    get_vlm_service,
    Retriever,
    get_reranker,
    MilvusVectorStore,
    get_embedder,
)

# Import Agents
from src.agents import InterpreterAgent, ScenicCoderAgent, CriticAgent

# Import Workflow / schema
from src.workflow import ScenarioWorkflow

def _initialize_logging(config) -> logging.Logger:
    """Configure logging and return a module logger."""
    setup_logging(
        level=config.logging.level,
        log_format=config.logging.format,
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting system initialization...")
    logger.info(
        f"Using configuration: "
        f"LLM={config.llm.provider}, "
        f"VLM={config.vlm.provider}, "
        f"Embedding={config.embedding.provider}"
    )
    return logger


def _initialize_llm_service(config, logger: logging.Logger):
    """Initialize shared LLM service."""
    logger.info(f"Initializing LLM service: {config.llm.provider}")
    llm_kwargs = {
        "provider": config.llm.provider,
        "model": config.llm.model,
        "temperature": config.llm.temperature,
        "max_tokens": config.llm.max_tokens,
        "api_key": config.llm.api_key,
    }
    if getattr(config.llm, "base_url", None):
        llm_kwargs["base_url"] = config.llm.base_url
    llm_service = get_llm_service(**llm_kwargs)
    logger.info("LLM service initialized.")
    return llm_service


def _initialize_vlm_service(config, logger: logging.Logger):
    """Initialize VLM service."""
    logger.info(f"Initializing VLM service: {config.vlm.provider}")
    vlm_kwargs = {
        "provider": config.vlm.provider,
        "model": config.vlm.model,
        "temperature": config.vlm.temperature,
        "max_tokens": config.vlm.max_tokens,
        "api_key": config.vlm.api_key,
    }
    if config.vlm.model_path:
        vlm_kwargs["model_path"] = config.vlm.model_path
    if getattr(config.vlm, "base_url", None):
        vlm_kwargs["base_url"] = config.vlm.base_url
    vlm_service = get_vlm_service(**vlm_kwargs)
    logger.info("VLM service initialized.")
    return vlm_service


def _initialize_embedder(config, logger: logging.Logger):
    """Initialize embedding model."""
    logger.info(f"Initializing Embedder: {config.embedding.provider}")
    embedder_kwargs = {
        "provider": config.embedding.provider,
        "model_name": config.embedding.model_name,
        # "device": config.embedding.device,
        # "batch_size": config.embedding.batch_size,
    }
    if config.embedding.model_path:
        embedder_kwargs["model_path"] = config.embedding.model_path
    embedder = get_embedder(**embedder_kwargs)
    logger.info("Embedder initialized.")
    return embedder

def _initialize_snippet_embedder(logger: logging.Logger):
    """Initialize snippet embedder."""
    logger.info("Initializing Snippet Embedder...")
    snippet_embedder = get_embedder(provider="huggingface", model_name="sentence-transformers/all-MiniLM-L6-v2", device="cuda")
    logger.info("Snippet Embedder initialized.")
    return snippet_embedder

def _initialize_vector_store(config, embedder, logger: logging.Logger) -> MilvusVectorStore:
    """Initialize vector database (Milvus) using the embedder dimension."""
    logger.info(
        f"Initializing Vector Store (Milvus "
    )
    embedding_dim = embedder.dimension
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
        embedding_dim=embedding_dim,
        collection_name=config.vector_db.collection_name,
        snippets_collection_name=config.vector_db.snippets_collection_name,
        connection_args=connection_args,
        index_params=index_params,
        search_params=search_params,
    )
    logger.info("Vector Store initialized.")
    return vector_db


def _initialize_reranker(config, logger: logging.Logger):
    """Initialize reranker if enabled in config."""
    if not config.retrieval.enable_reranking:
        logger.info("Reranking disabled in configuration.")
        return None

    logger.info("Initializing Reranker...")
    reranker_kwargs = {
        "provider": config.reranking.provider,  # Currently only qwen is supported
        "device": config.reranking.device,
        "model_name": config.reranking.model_name,
    }
    if config.reranking.model_path:
        reranker_kwargs["model_path"] = config.reranking.model_path

    reranker = get_reranker(**reranker_kwargs)
    logger.info("Reranker initialized.")
    return reranker


def _initialize_retriever(
    config,
    vector_db: MilvusVectorStore,
    reranker,
    logger: logging.Logger,
) -> Retriever:
    """Initialize retrieval pipeline."""
    logger.info("Initializing Retrieval Pipeline...")
    retrieval_pipeline = Retriever(
        vectorstore=vector_db,
        reranker=reranker,
        top_k=config.retrieval.top_k,
        similarity_threshold=config.retrieval.similarity_threshold,
    )
    logger.info("Retrieval Pipeline initialized.")
    return retrieval_pipeline


def _initialize_agents(
    shared_llm_service,
    vlm_service,
    vector_db,
    snippets_embedder,
    logger: logging.Logger,
    config=None,
) -> Tuple[InterpreterAgent, ScenicCoderAgent, CriticAgent]:
    """Initialize all agents."""
    logger.info("Initializing Agents...")

    interpreter_agent = InterpreterAgent(vlm_service=vlm_service)
    scenic_coder_agent = ScenicCoderAgent(llm_service=shared_llm_service, vector_store=vector_db, snippets_embedder=snippets_embedder)
    critic_kwargs = {}
    if config is not None and getattr(config, "critic", None) is not None:
        critic_kwargs = {
            "prompt_name": config.critic.prompt_name,
            "include_bev_video": config.critic.include_bev_video,
            "include_scenic_code": config.critic.include_scenic_code,
        }
    critic_agent = CriticAgent(vlm_service=vlm_service, **critic_kwargs)

    logger.info("Agents initialized.")
    return interpreter_agent, scenic_coder_agent, critic_agent


def _initialize_workflow(
    interpreter_agent: InterpreterAgent,
    scenic_coder_agent: ScenicCoderAgent,
    critic_agent: CriticAgent,
    retrieval_pipeline: Retriever,
    embedder,
    logger: logging.Logger,
) -> ScenarioWorkflow:
    """Wire up and return the main ScenarioWorkflow."""
    logger.info("Wiring up the LangGraph Workflow...")
    workflow = ScenarioWorkflow(
        interpreter=interpreter_agent,
        coder=scenic_coder_agent,
        critic=critic_agent,
        retriever=retrieval_pipeline,
        embedder=embedder,
        logger=logger,
    )
    logger.info("Workflow initialized.")
    return workflow

class ChatbotWorkflow:
    def __init__(self):
        self.workflow = None

    def initialize_system(self, config_path: Optional[str] = None) -> ScenarioWorkflow:
        """
        Builds the entire ADS-MRAG system from the bottom up using configuration.
        
        Args:
            config_path: Optional path to YAML configuration file.
                        If None, uses default config/config.yaml
        
        Returns:
            Initialized ScenarioWorkflow instance
        """
        # Load configuration
        config = get_config(config_path)
        
        logger = _initialize_logging(config)

        logger.info("Initializing Services...")
        shared_llm_service = _initialize_llm_service(config, logger)
        vlm_service = _initialize_vlm_service(config, logger)
        embedder = _initialize_embedder(config, logger)
        snippets_embedder = _initialize_snippet_embedder(logger)
        vector_db = _initialize_vector_store(config, embedder, logger)
        reranker = _initialize_reranker(config, logger)
        retrieval_pipeline = _initialize_retriever(config, vector_db, reranker, logger)

        interpreter_agent, scenic_coder_agent, critic_agent = _initialize_agents(
            shared_llm_service=shared_llm_service,
            vlm_service=vlm_service,
            vector_db=vector_db,
            snippets_embedder=snippets_embedder,
            logger=logger,
            config=config,
        )
        workflow = _initialize_workflow(
            interpreter_agent=interpreter_agent,
            scenic_coder_agent=scenic_coder_agent,
            critic_agent=critic_agent,
            retrieval_pipeline=retrieval_pipeline,
            embedder=embedder,
            logger=logger,
        )

        logger.info("✅ System initialized successfully!")
        return workflow

    def process_user_input(self, user_text: str, image_path: Optional[str], video_path: Optional[str], history: list, session_id: str):
        """
        Take user input (text, image, video), history and session_id, run the workflow and return the response.
        """
        # 1. Initialization
        if not session_id:
            session_id = str(uuid.uuid4())
        if not self.workflow:
            self.workflow = self.initialize_system()
            
        config = {"configurable": {"thread_id": session_id}}

        # 1. Append initial messages in the NEW Gradio format
        if image_path:
            # Media goes inside a tuple, inside the 'content' key
            history.append({"role": "user", "content": gr.FileData(path=image_path)})
        if video_path:
            history.append({"role": "user", "content": gr.FileData(path=video_path)})
            
        # Add the text
        history.append({"role": "user", "content": user_text})
        
        # Add the assistant's loading message
        history.append({"role": "assistant", "content": "⏳ Processing..."})
        yield history, session_id

        # 2. Check if the workflow is paused waiting for human review
        current_state = self.workflow.app.get_state(config)
        is_paused = len(current_state.next) > 0  # If 'next' has nodes, it is paused!

        # Bundle the text and optional media into a dictionary matching your schema
        multimodal_payload = MultimodalQuery(
            text=user_text,
            image_path=image_path,
            video_path=video_path
        )
        
        # A simple string representation for the standard messages history
        new_message = {"role": "user", "content": multimodal_payload.model_dump_json()}

        if is_paused:
            # It was paused! Inject the user's feedback into the state
            self.workflow.app.update_state(config, {
                "user_satisfied": False if user_text.lower() != "accept" else True,
                "user_modification": multimodal_payload, # Use the payload here!
                "messages": [new_message]
            })
            stream_input = None 
        else:
            # Brand new request! Pass the initial multimodal query
            stream_input = {
                "user_query": multimodal_payload, # Use the payload here!
                "messages": [new_message]
            }

        # 3. Get the messages from the LangGraph stream
        for event in self.workflow.app.stream(stream_input, config=config):
            for node_name, state_update in event.items():
                if isinstance(state_update, dict) and "messages" in state_update: # state_update might be None
                    for message in state_update["messages"]:
                        if message.get("role") == "assistant":
                            new_text = message.get("content", "")
                            formatted_text = f"**[{node_name.replace('_', ' ').title()}]**\n{new_text}"
                            history.append({"role": "assistant", "content": formatted_text})
                            yield history, session_id
        

        # 4. Check if the graph paused again after this run
        new_state = self.workflow.app.get_state(config)
        if len(new_state.next) > 0:
            final_ai_text = "\n\n**🛑 Graph Paused:** Please review the DSL above. Type your modifications, or type 'accept' to finish."
            history.append({"role": "assistant", "content": final_ai_text})
            yield history, session_id

def mock_workflow():
# Example test runner using real initialized components
    workflow = ChatbotWorkflow().initialize_system()

    # Construct a sample user query (adjust paths/text as needed)
    user_query = MultimodalQuery(
        text="Generate me a highway scenario looks like the one in this video",
        image_path=None,
        video_path="/home/dellpro2/chenli/ads-mrag/ads-mrag/data/processed/test_data/testvideo.mp4",  # e.g. "data/processed/test_data/testvideo.mp4"
    )

    initial_state = {"user_query": user_query, "max_count": 3}
    config = {"configurable": {"thread_id": "app_real_components_test"}}

    logger = logging.getLogger(__name__)
    logger.info("🚀 STARTING WORKFLOW RUN WITH REAL COMPONENTS...")

    # Initial run
    for event in workflow.app.stream(initial_state, config=config):
        pass

    logger.info("🛑 GRAPH PAUSED. Pretending user clicked 'Reject'...")
    user_feedback = MultimodalQuery(
        text="Please add another car in the scenario which turns left at the intersection behind the ego vehicle as I marked with a red box in the image.", 
        image_path="/home/dellpro2/chenli/ads-mrag/ads-mrag/data/processed/test_data/testimage.png", 
        video_path=None
    )
    workflow.app.update_state(
        config, 
        {"user_satisfied": False,
        "user_modification": user_feedback
        }
    )

    logger.info("🚀 RESUMING WITH HUMAN FEEDBACK...")
    
    for event in workflow.app.stream(None, config=config):
        pass
    
    logger.info("🛑 GRAPH PAUSED. Pretending user clicked 'Accept'...")
    workflow.app.update_state(config, {"user_satisfied": True})

    logger.info("✅ WORKFLOW COMPLETED SUCCESSFULLY.")

def build_ui():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🚗 Autonomous Driving Scenario Builder")
        gr.Markdown("Describe a scenario, upload references, review the generated DSL, and provide feedback to modify it!")

        session_thread_id = gr.State(None)
        chatbot = gr.Chatbot(height=500)

        # Create a layout for inputs
        with gr.Row():
            with gr.Column(scale=4):
                txt_input = gr.Textbox(
                    show_label=False,
                    placeholder="E.g., Generate a highway scenario with a truck merging...",
                    lines=2
                )
                with gr.Row():
                    # type="filepath" forces Gradio to pass the path string instead of numpy arrays
                    img_input = gr.Image(type="filepath", label="Upload Image (Optional)", height=150)
                    vid_input = gr.Video(label="Upload Video (Optional)", height=150)
                    
            with gr.Column(scale=1, min_width=100):
                submit_btn = gr.Button("Send", variant="primary", size="lg")

        # Wire the inputs to the function
        workflow = ChatbotWorkflow()
        
        # Note the updated `inputs` list! We pass the text, image, video, chatbot history, and session ID
        input_components = [txt_input, img_input, vid_input, chatbot, session_thread_id]
        output_components = [chatbot, session_thread_id]

        submit_event = txt_input.submit(
            workflow.process_user_input,
            inputs=input_components,
            outputs=output_components
        )
        
        submit_btn.click(
            workflow.process_user_input,
            inputs=input_components,
            outputs=output_components
        )

        # Clear the inputs instantly after the user hits submit
        def clear_inputs():
            return "", None, None
            
        submit_event.then(clear_inputs, outputs=[txt_input, img_input, vid_input])
        submit_btn.click(clear_inputs, outputs=[txt_input, img_input, vid_input])

    return demo

if __name__ == "__main__":
    app = build_ui()
    app.launch()
    
