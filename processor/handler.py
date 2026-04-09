import os
import logging

log = logging.getLogger(__name__)

# Lambda event keys → environment variable mappings
EVENT_KEY_MAP = {
    "inputDir": "INPUT_DIR",
    "outputDir": "OUTPUT_DIR",
    "initialDownsample": "INITIAL_DOWNSAMPLE",
    "tileSize": "TILE_SIZE",
    "compression": "COMPRESSION",
    "compressionLevel": "COMPRESSION_LEVEL",
    "maxLevels": "MAX_LEVELS",
    "minDimension": "MIN_DIMENSION",
    "chunkSlices": "CHUNK_SLICES",
    "maxWorkers": "MAX_WORKERS",
}


def handler(event, context):
    """AWS Lambda handler — maps camelCase event keys to env vars, then runs."""
    for event_key, env_var in EVENT_KEY_MAP.items():
        if event_key in event:
            os.environ[env_var] = str(event[event_key])

    from processor.main import run
    run()

    return {"statusCode": 200, "body": "Conversion complete"}
