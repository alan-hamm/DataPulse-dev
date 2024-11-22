# Developed traditionally with the addition of AI assistance
# Author: Alan Hamm
# Date: April 2024

import os
from yaml import SafeLoader, nodes
from datetime import datetime

def getenv(loader, node):
    # If the node is a sequence, construct each item as an environment variable
    if isinstance(node, nodes.SequenceNode):
        env_values = [os.getenv(loader.construct_scalar(item)) for item in node.value]
        # Handle None values by providing a default or raising an error
        if None in env_values:
            raise ValueError("One or more environment variables are not set.")
        return env_values
    else:
        # If it's a scalar, treat it as a single environment variable
        value = os.getenv(loader.construct_scalar(node))
        if value is None:
            raise ValueError(f"The environment variable {loader.construct_scalar(node)} is not set.")
        return value

def join(loader, node):
    seq = loader.construct_sequence(node)
    try:
        return os.path.join(*seq)
    except TypeError as e:
        raise ValueError(f"Error joining path components: {seq}. Make sure all elements are strings.") from e

def get_current_time(loader, node):
    fmt = loader.construct_scalar(node)
    return datetime.now().strftime(fmt)

# Register constructors with SafeLoader using custom tags
SafeLoader.add_constructor('!join', join)
SafeLoader.add_constructor('!getenv', getenv)
SafeLoader.add_constructor('!current_time', get_current_time)  # Match this with the tag used in YAML

# Now you can use these constructors when loading YAML files.