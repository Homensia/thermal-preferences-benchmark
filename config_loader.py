"""
Configuration Loader Module
===========================

This module provides a simple utility function to load experiment
and model configuration parameters from a YAML file. Centralizing
the configuration ensures reproducibility, easier experiment 
management, and decoupling between code and settings.

Functions
---------
load_config(path="config.yaml"):
    Load and parse a YAML configuration file, returning its content
    as a Python dictionary. The function uses `yaml.safe_load` to
    prevent execution of arbitrary YAML tags.
"""


import yaml

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)
