import sys
import importlib

# List of libraries to check
libraries = [
    'argparse', 'bokeh', 'dask', 'datetime', 'decimal', 'distributed', 'filelock', 'gc',
    'gensim', 'hashlib', 'itertools', 'json', 'logging', 'math', 'matplotlib', 
    'multiprocessing', 'numpy', 'os', 'pandas', 'pickle', 'pyLDAvis', 'random', 
    'socket', 'sqlalchemy', 'sys', 'time', 'tornado', 'tqdm', 'warnings', 'yaml', 'zipfile'
]

# Print Python version
print(f"Python version: {sys.version}\n")

# Check and print version of each library
for lib in libraries:
    try:
        module = importlib.import_module(lib)
        version = getattr(module, '__version__', 'Version info not available')
        print(f"{lib}: {version}")
    except ImportError:
        print(f"{lib}: Not installed")
