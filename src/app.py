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
    get_embedder
)

# Import Agents
from src.agents import InterpreterAgent, ScenicCoderAgent, CriticAgent

# Import Workflow
from src.workflow import ScenarioWorkflow

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
        
        # Setup logging using config
        setup_logging(
            level=config.logging.level,
            log_file=config.logging.file,
            log_format=config.logging.format,
            run_name="ads_mrag_startup"
        )
        logger = logging.getLogger(__name__)
        logger.info("Starting system initialization...")
        logger.info(f"Using configuration: LLM={config.llm.provider}, VLM={config.vlm.provider}, Embedding={config.embedding.provider}")

        # ==========================================
        # LAYER 3: INITIALIZE SHARED SERVICES (Singletons)
        # ==========================================
        logger.info("Initializing Services...")
        
        # Initialize LLM service
        logger.info(f"Initializing LLM service: {config.llm.provider}")
        shared_llm_service = get_llm_service(
            provider=config.llm.provider,
            model=config.llm.model,
            api_key=config.llm.api_key,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens
        )
        
        # Initialize VLM service
        logger.info(f"Initializing VLM service: {config.vlm.provider}")
        vlm_kwargs = {
            "provider": config.vlm.provider,
            "model": config.vlm.model,
            "api_key": config.vlm.api_key,
            "device": config.vlm.device,
            "temperature": config.vlm.temperature,
            "max_tokens": config.vlm.max_tokens,
            "fps": config.vlm.fps,
            "max_frames": config.vlm.max_frames,
            "default_instruction": config.vlm.default_instruction
        }
        # Add model_path if provided (required for Qwen3VL)
        if config.vlm.model_path:
            vlm_kwargs["model_path"] = config.vlm.model_path
        vlm_service = get_vlm_service(**vlm_kwargs)
        
        # Initialize Embedder
        logger.info(f"Initializing Embedder: {config.embedding.provider}")
        embedder_kwargs = {
            "provider": config.embedding.provider,
            "model_name": config.embedding.model_name,
            "device": config.embedding.device,
            "batch_size": config.embedding.batch_size
        }
        # Add model_path if provided (required for Qwen models)
        if config.embedding.model_path:
            embedder_kwargs["model_path"] = config.embedding.model_path
        embedder = get_embedder(**embedder_kwargs)
        
        # Initialize Vector Store (Milvus)
        logger.info(f"Initializing Vector Store (Milvus {'Lite' if config.vector_db.use_lite else 'Server'})...")
        embedding_dim = embedder.dimension
        
        # Setup connection args based on mode
        if config.vector_db.use_lite:
            connection_args = {"uri": config.vector_db.lite_db_path}
            index_params = None
            search_params = None
        else:
            connection_args = {
                "host": config.vector_db.host,
                "port": config.vector_db.port
            }
            index_params = {
                "metric_type": config.vector_db.distance_metric.upper(),
                "index_type": config.vector_db.index_type,
                "params": {"nlist": config.vector_db.nlist}
            }
            search_params = {
                "metric_type": config.vector_db.distance_metric.upper(),
                "params": {"nprobe": config.vector_db.nprobe}
            }
        
        vector_db = MilvusVectorStore(
            embedding_dim=embedding_dim,
            collection_name=config.vector_db.collection_name,
            connection_args=connection_args,
            index_params=index_params,
            search_params=search_params
        )
        
        # Initialize Reranker (if enabled)
        reranker = None
        if config.retrieval.enable_reranking:
            logger.info("Initializing Reranker...")
            reranker_kwargs = {
                "provider": config.reranking.provider,  # Currently only qwen is supported
                "device": config.reranking.device,
                "torch_dtype": config.reranking.torch_dtype,
                "instruction": config.reranking.instruction,
                "fps": config.reranking.fps
            }
            if config.reranking.model_path:
                reranker_kwargs["model_path"] = config.reranking.model_path
            if config.reranking.model_name:
                reranker_kwargs["model_name"] = config.reranking.model_name
            reranker = get_reranker(**reranker_kwargs)
        
        # Initialize Retrieval Pipeline
        logger.info("Initializing Retrieval Pipeline...")
        retrieval_pipeline = Retriever(
            vectorstore=vector_db,
            reranker=reranker,
            top_k=config.retrieval.top_k,
            similarity_threshold=config.retrieval.similarity_threshold
        )

        # ==========================================
        # LAYER 2: INITIALIZE AGENTS (Injecting Services)
        # ==========================================
        logger.info("Initializing Agents...")
        # Notice how both agents get the EXACT SAME shared_llm_service!
        interpreter_agent = InterpreterAgent(llm_service=shared_llm_service)
        scenic_coder_agent = ScenicCoderAgent(llm_service=shared_llm_service)
        critic_agent = CriticAgent(vlm_service=vlm_service)

        # ==========================================
        # LAYER 1: INITIALIZE WORKFLOW (Injecting Agents)
        # ==========================================
        logger.info("Wiring up the LangGraph Workflow...")
        workflow = ScenarioWorkflow(
            interpreter=interpreter_agent,
            coder=scenic_coder_agent,
            critic=critic_agent,
            retriever=retrieval_pipeline,
            embedder=embedder
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