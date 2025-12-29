"""
Execute all example notebooks to populate them with results.
"""
import sys
from pathlib import Path
from nbconvert.preprocessors import ExecutePreprocessor
import nbformat

def execute_notebook(notebook_path):
    """Execute a notebook and save it with outputs."""
    print(f"\n{'='*60}")
    print(f"Executing: {notebook_path.name}")
    print(f"{'='*60}")
    
    # Read notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    # Execute
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    try:
        ep.preprocess(nb, {'metadata': {'path': notebook_path.parent}})
        print(f"✓ Successfully executed {notebook_path.name}")
    except Exception as e:
        print(f"⚠ Error executing {notebook_path.name}: {e}")
        print("  Continuing with partial results...")
    
    # Save
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    
    print(f"✓ Saved {notebook_path.name} with outputs\n")

if __name__ == "__main__":
    examples_dir = Path(__file__).parent
    
    notebooks = [
        examples_dir / "01_basic_analysis.ipynb",
        examples_dir / "02_network_metrics.ipynb",
        examples_dir / "03_community_detection.ipynb",
        examples_dir / "04_visualization.ipynb",
    ]
    
    print("Executing example notebooks...")
    print("This may take a few minutes...")
    
    for nb_path in notebooks:
        if nb_path.exists():
            execute_notebook(nb_path)
        else:
            print(f"⚠ Notebook not found: {nb_path}")
    
    print("\n" + "="*60)
    print("✓ All notebooks executed!")
    print("="*60)

