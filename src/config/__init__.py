import yaml
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "default.yaml")

with open(CONFIG_PATH, "r") as file:
    CONFIG = yaml.safe_load(file)
