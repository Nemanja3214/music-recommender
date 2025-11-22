import nbformat
from nbconvert import PythonExporter
import os

def export_notebook_to_python(notebook_path, output_path=None):

    """
    Export a Jupyter notebook to a Python script (.py).

    Args:
        notebook_path (str): Path to the .ipynb notebook.
        output_path (str, optional): Path to save the .py file. Defaults to same directory.
    """
    if not os.path.exists(notebook_path):
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")

    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # Convert to Python
    exporter = PythonExporter()
    script_body, _ = exporter.from_notebook_node(nb)

    # Define output path
    if output_path is None:
        base_name = os.path.splitext(notebook_path)[0]
        output_path = f"{base_name}.py"

    # Save Python script
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(script_body)

    print(f"✅ Notebook exported successfully to: {output_path}")

# Example usage inside a Jupyter cell:
export_notebook_to_python("main.ipynb")