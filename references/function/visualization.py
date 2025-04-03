import logging
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
from dataclasses import dataclass

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


@dataclass
class VisualizationConfig:
    """
    Configuration for visualization
    """
    output_dir: str
    figure_size: Tuple[int, int] = (10, 6)
    dpi: int = 100
    style: str = "seaborn"
    palette: str = "viridis"
    save_format: str = "png"


def plot_training_curves(
    metrics: Dict[str, List[float]],
    config: VisualizationConfig,
) -> None:
    """
    Plot training curves
    
    Args:
        metrics: Dictionary of metric names and values
        config: Visualization configuration
    """
    try:
        plt.style.use(config.style)
        
        # Create figure
        plt.figure(figsize=config.figure_size, dpi=config.dpi)
        
        # Plot each metric
        for metric_name, values in metrics.items():
            plt.plot(values, label=metric_name)
            
        # Customize plot
        plt.title("Training Curves")
        plt.xlabel("Step")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plt.savefig(
            f"{config.output_dir}/training_curves.{config.save_format}",
            bbox_inches="tight",
            dpi=config.dpi,
        )
        plt.close()
        
        logger.info("Saved training curves plot")
        
    except Exception as e:
        logger.error(f"Error plotting training curves: {str(e)}")
        raise


def plot_attention_heatmap(
    attention_weights: torch.Tensor,
    config: VisualizationConfig,
    title: str = "Attention Heatmap",
) -> None:
    """
    Plot attention heatmap
    
    Args:
        attention_weights: Attention weights tensor
        config: Visualization configuration
        title: Plot title
    """
    try:
        plt.style.use(config.style)
        
        # Convert to numpy array
        attention_weights = attention_weights.detach().cpu().numpy()
        
        # Create figure
        plt.figure(figsize=config.figure_size, dpi=config.dpi)
        
        # Plot heatmap
        sns.heatmap(
            attention_weights,
            cmap=config.palette,
            annot=True,
            fmt=".2f",
            square=True,
        )
        
        # Customize plot
        plt.title(title)
        plt.xlabel("Key")
        plt.ylabel("Query")
        
        # Save plot
        plt.savefig(
            f"{config.output_dir}/attention_heatmap.{config.save_format}",
            bbox_inches="tight",
            dpi=config.dpi,
        )
        plt.close()
        
        logger.info("Saved attention heatmap plot")
        
    except Exception as e:
        logger.error(f"Error plotting attention heatmap: {str(e)}")
        raise


def plot_token_importance(
    token_importance: torch.Tensor,
    tokens: List[str],
    config: VisualizationConfig,
    title: str = "Token Importance",
) -> None:
    """
    Plot token importance
    
    Args:
        token_importance: Token importance scores
        tokens: List of tokens
        config: Visualization configuration
        title: Plot title
    """
    try:
        plt.style.use(config.style)
        
        # Convert to numpy array
        token_importance = token_importance.detach().cpu().numpy()
        
        # Create figure
        plt.figure(figsize=config.figure_size, dpi=config.dpi)
        
        # Plot bar chart
        plt.bar(
            range(len(tokens)),
            token_importance,
            color=plt.cm.get_cmap(config.palette)(0.5),
        )
        
        # Customize plot
        plt.title(title)
        plt.xlabel("Token")
        plt.ylabel("Importance")
        plt.xticks(range(len(tokens)), tokens, rotation=45, ha="right")
        
        # Save plot
        plt.savefig(
            f"{config.output_dir}/token_importance.{config.save_format}",
            bbox_inches="tight",
            dpi=config.dpi,
        )
        plt.close()
        
        logger.info("Saved token importance plot")
        
    except Exception as e:
        logger.error(f"Error plotting token importance: {str(e)}")
        raise


def plot_gradient_flow(
    model: PreTrainedModel,
    config: VisualizationConfig,
) -> None:
    """
    Plot gradient flow
    
    Args:
        model: Model to visualize
        config: Visualization configuration
    """
    try:
        plt.style.use(config.style)
        
        # Get gradients
        gradients = []
        names = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradients.append(param.grad.detach().cpu().numpy().flatten())
                names.append(name)
                
        # Create figure
        plt.figure(figsize=config.figure_size, dpi=config.dpi)
        
        # Plot box plot
        plt.boxplot(gradients, labels=names)
        
        # Customize plot
        plt.title("Gradient Flow")
        plt.xlabel("Layer")
        plt.ylabel("Gradient Value")
        plt.xticks(rotation=45, ha="right")
        
        # Save plot
        plt.savefig(
            f"{config.output_dir}/gradient_flow.{config.save_format}",
            bbox_inches="tight",
            dpi=config.dpi,
        )
        plt.close()
        
        logger.info("Saved gradient flow plot")
        
    except Exception as e:
        logger.error(f"Error plotting gradient flow: {str(e)}")
        raise


def plot_parameter_distribution(
    model: PreTrainedModel,
    config: VisualizationConfig,
) -> None:
    """
    Plot parameter distribution
    
    Args:
        model: Model to visualize
        config: Visualization configuration
    """
    try:
        plt.style.use(config.style)
        
        # Get parameters
        parameters = []
        names = []
        
        for name, param in model.named_parameters():
            parameters.append(param.detach().cpu().numpy().flatten())
            names.append(name)
            
        # Create figure
        plt.figure(figsize=config.figure_size, dpi=config.dpi)
        
        # Plot violin plot
        plt.violinplot(parameters, showmeans=True, showmedians=True)
        
        # Customize plot
        plt.title("Parameter Distribution")
        plt.xlabel("Layer")
        plt.ylabel("Parameter Value")
        plt.xticks(range(1, len(names) + 1), names, rotation=45, ha="right")
        
        # Save plot
        plt.savefig(
            f"{config.output_dir}/parameter_distribution.{config.save_format}",
            bbox_inches="tight",
            dpi=config.dpi,
        )
        plt.close()
        
        logger.info("Saved parameter distribution plot")
        
    except Exception as e:
        logger.error(f"Error plotting parameter distribution: {str(e)}")
        raise


def visualize_model_architecture(
    model: PreTrainedModel,
    config: VisualizationConfig,
) -> None:
    """
    Visualize model architecture
    
    Args:
        model: Model to visualize
        config: Visualization configuration
    """
    try:
        from torchviz import make_dot
        
        # Create dummy input
        dummy_input = torch.randn(1, 512)
        
        # Create graph
        dot = make_dot(
            model(dummy_input),
            params=dict(model.named_parameters()),
            show_attrs=True,
            show_saved=True,
        )
        
        # Save graph
        dot.render(
            f"{config.output_dir}/model_architecture",
            format=config.save_format,
            cleanup=True,
        )
        
        logger.info("Saved model architecture visualization")
        
    except Exception as e:
        logger.error(f"Error visualizing model architecture: {str(e)}")
        raise 