import logging
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
from dataclasses import dataclass

import torch
import gradio as gr
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


@dataclass
class DeploymentConfig:
    """
    Configuration for model deployment
    """
    model_path: str
    tokenizer_path: str
    device: str = "cuda"
    precision: str = "fp16"
    max_length: int = 100
    num_beams: int = 4
    temperature: float = 1.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    port: int = 7860
    share: bool = False
    auth: Optional[Tuple[str, str]] = None


def load_model_for_inference(
    config: DeploymentConfig,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load model and tokenizer for inference
    
    Args:
        config: Deployment configuration
        
    Returns:
        Tuple[PreTrainedModel, PreTrainedTokenizer]: Loaded model and tokenizer
    """
    try:
        # Set device
        device = torch.device(config.device)
        
        # Load tokenizer
        tokenizer = PreTrainedTokenizer.from_pretrained(config.tokenizer_path)
        
        # Load model
        model = PreTrainedModel.from_pretrained(
            config.model_path,
            torch_dtype=torch.float16 if config.precision == "fp16" else torch.float32,
            device_map="auto",
        )
        
        logger.info(f"Loaded model from {config.model_path}")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Error loading model for inference: {str(e)}")
        raise


def create_gradio_interface(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    config: DeploymentConfig,
) -> gr.Interface:
    """
    Create Gradio interface for model inference
    
    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        config: Deployment configuration
        
    Returns:
        gr.Interface: Gradio interface
    """
    try:
        def generate_text(prompt: str) -> str:
            # Encode prompt
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=config.max_length,
                    num_beams=config.num_beams,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    repetition_penalty=config.repetition_penalty,
                    early_stopping=True,
                )
                
            # Decode output
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return generated_text
            
        # Create interface
        interface = gr.Interface(
            fn=generate_text,
            inputs=gr.Textbox(lines=5, placeholder="Enter your prompt here..."),
            outputs=gr.Textbox(lines=5),
            title="Text Generation",
            description="Generate text using the model",
            examples=[
                ["Write a function to sort a list in Python."],
                ["Explain how neural networks work."],
                ["Write a SQL query to join two tables."],
            ],
        )
        
        logger.info("Created Gradio interface")
        return interface
        
    except Exception as e:
        logger.error(f"Error creating Gradio interface: {str(e)}")
        raise


def deploy_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    config: DeploymentConfig,
) -> None:
    """
    Deploy model using Gradio
    
    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        config: Deployment configuration
    """
    try:
        # Create interface
        interface = create_gradio_interface(model, tokenizer, config)
        
        # Launch interface
        interface.launch(
            server_name="0.0.0.0",
            server_port=config.port,
            share=config.share,
            auth=config.auth,
        )
        
        logger.info(f"Deployed model on port {config.port}")
        
    except Exception as e:
        logger.error(f"Error deploying model: {str(e)}")
        raise


def create_fastapi_app(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    config: DeploymentConfig,
) -> Any:
    """
    Create FastAPI app for model inference
    
    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        config: Deployment configuration
        
    Returns:
        Any: FastAPI app
    """
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
        
        app = FastAPI()
        
        class GenerateRequest(BaseModel):
            prompt: str
            
        class GenerateResponse(BaseModel):
            generated_text: str
            
        @app.post("/generate", response_model=GenerateResponse)
        async def generate(request: GenerateRequest) -> GenerateResponse:
            try:
                # Encode prompt
                inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)
                
                # Generate
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_length=config.max_length,
                        num_beams=config.num_beams,
                        temperature=config.temperature,
                        top_p=config.top_p,
                        repetition_penalty=config.repetition_penalty,
                        early_stopping=True,
                    )
                    
                # Decode output
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                return GenerateResponse(generated_text=generated_text)
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
                
        logger.info("Created FastAPI app")
        return app
        
    except Exception as e:
        logger.error(f"Error creating FastAPI app: {str(e)}")
        raise


def deploy_fastapi_app(
    app: Any,
    config: DeploymentConfig,
) -> None:
    """
    Deploy FastAPI app
    
    Args:
        app: FastAPI app
        config: Deployment configuration
    """
    try:
        import uvicorn
        
        # Run app
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=config.port,
        )
        
        logger.info(f"Deployed FastAPI app on port {config.port}")
        
    except Exception as e:
        logger.error(f"Error deploying FastAPI app: {str(e)}")
        raise


def create_dockerfile(
    config: DeploymentConfig,
    output_path: str,
) -> None:
    """
    Create Dockerfile for model deployment
    
    Args:
        config: Deployment configuration
        output_path: Path to save Dockerfile
    """
    try:
        dockerfile_content = f"""
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /app

# Install dependencies
RUN pip install transformers torch gradio fastapi uvicorn

# Copy model files
COPY {config.model_path} /app/model
COPY {config.tokenizer_path} /app/tokenizer

# Copy app code
COPY . /app/

# Expose port
EXPOSE {config.port}

# Run app
CMD ["python", "app.py"]
"""
        
        # Save Dockerfile
        with open(output_path, "w") as f:
            f.write(dockerfile_content)
            
        logger.info(f"Created Dockerfile at {output_path}")
        
    except Exception as e:
        logger.error(f"Error creating Dockerfile: {str(e)}")
        raise 