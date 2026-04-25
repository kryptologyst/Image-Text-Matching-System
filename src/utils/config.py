"""Configuration management utilities."""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from omegaconf import DictConfig, OmegaConf


def load_config(config_path: Union[str, Path]) -> DictConfig:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        DictConfig: Loaded configuration
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        config = OmegaConf.load(config_path)
        return config
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML in config file {config_path}: {e}")


def save_config(config: DictConfig, output_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration to save
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    OmegaConf.save(config, output_path)


def merge_configs(base_config: DictConfig, override_config: DictConfig) -> DictConfig:
    """
    Merge two configurations, with override_config taking precedence.
    
    Args:
        base_config: Base configuration
        override_config: Override configuration
        
    Returns:
        DictConfig: Merged configuration
    """
    return OmegaConf.merge(base_config, override_config)


def resolve_config_paths(config: DictConfig, base_dir: Union[str, Path]) -> DictConfig:
    """
    Resolve relative paths in configuration to absolute paths.
    
    Args:
        config: Configuration to resolve
        base_dir: Base directory for relative paths
        
    Returns:
        DictConfig: Configuration with resolved paths
    """
    base_dir = Path(base_dir)
    resolved_config = OmegaConf.create(OmegaConf.to_yaml(config))
    
    # Common path fields that should be resolved
    path_fields = [
        "data_dir",
        "output_dir", 
        "checkpoint_dir",
        "assets_dir",
        "log_file",
        "model_path",
        "config_path",
    ]
    
    for field in path_fields:
        if field in resolved_config and resolved_config[field]:
            path_value = resolved_config[field]
            if isinstance(path_value, str) and not Path(path_value).is_absolute():
                resolved_config[field] = str(base_dir / path_value)
    
    return resolved_config


def create_output_dirs(config: DictConfig) -> None:
    """
    Create output directories specified in configuration.
    
    Args:
        config: Configuration containing directory paths
    """
    dirs_to_create = [
        config.get("output_dir"),
        config.get("checkpoint_dir"),
        config.get("assets_dir"),
        config.get("log_file", "").rsplit("/", 1)[0] if config.get("log_file") else None,
    ]
    
    for dir_path in dirs_to_create:
        if dir_path:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


def get_config_value(config: DictConfig, key: str, default: Any = None) -> Any:
    """
    Get configuration value with dot notation support.
    
    Args:
        config: Configuration object
        key: Key in dot notation (e.g., "model.temperature")
        default: Default value if key not found
        
    Returns:
        Configuration value or default
    """
    try:
        return OmegaConf.select(config, key)
    except KeyError:
        return default


def set_config_value(config: DictConfig, key: str, value: Any) -> None:
    """
    Set configuration value with dot notation support.
    
    Args:
        config: Configuration object
        key: Key in dot notation (e.g., "model.temperature")
        value: Value to set
    """
    OmegaConf.set(config, key, value)


def validate_config(config: DictConfig, required_fields: list) -> None:
    """
    Validate that required fields are present in configuration.
    
    Args:
        config: Configuration to validate
        required_fields: List of required field names
        
    Raises:
        ValueError: If required fields are missing
    """
    missing_fields = []
    
    for field in required_fields:
        if not OmegaConf.select(config, field):
            missing_fields.append(field)
    
    if missing_fields:
        raise ValueError(f"Missing required configuration fields: {missing_fields}")


def print_config(config: DictConfig, title: str = "Configuration") -> None:
    """
    Print configuration in a formatted way.
    
    Args:
        config: Configuration to print
        title: Title for the configuration section
    """
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")
    print(OmegaConf.to_yaml(config))
    print(f"{'='*50}\n")
