import os

class Config:
    INPUT_DIR  = os.environ.get("INPUT_DIR",  "/inputs")
    OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/outputs")