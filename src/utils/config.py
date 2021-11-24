import os
import json


class Config():
    def __init__(self, config_path, output_path):
        self.config_path = config_path
        self.output_path = output_path

    def save_config(self, args):
        with open(os.path.join(self.output_path, "config.json"), "w") as fp:
            json.dump(args, fp, indent=4)

    def load_config(self, config_name):
        pass
