import yaml

with open("./config/default.yaml", "r") as file:
    CONFIG = yaml.safe_load(file)
