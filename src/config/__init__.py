import yaml

with open("./src/config/default.yaml", "r") as file:
    CONFIG = yaml.safe_load(file)
