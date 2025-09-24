# notebook_setup.py
"""
Setup utilities for notebooks to use local pywhyllm development version.
Import this at the beginning of your notebooks instead of manually adding sys.path.
"""
import sys
import os

def setup_local_pywhyllm():
    """Add the local pywhyllm source to Python path for development."""
    # Print current sys.path before modifications
    print("📋 Current sys.path before adding project root:")
    for i, path in enumerate(sys.path):
        print(f"  {i}: {path}")
    
    # Get the project root (two levels up from this file)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    
    print(f"\n🎯 Project root to add: {project_root}")
    
    # Add to sys.path if not already there
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"✅ Added local pywhyllm source to Python path: {project_root}")
    else:
        print(f"✅ Local pywhyllm source already in Python path")
    
    # Print sys.path after modifications
    print("\n📋 Updated sys.path after adding project root:")
    for i, path in enumerate(sys.path):
        marker = " ⭐" if path == project_root else ""
        print(f"  {i}: {path}{marker}")
    
    # Clear any cached pywhyllm modules to force reload from local source
    modules_to_remove = [key for key in sys.modules.keys() if key.startswith('pywhyllm')]
    for module_name in modules_to_remove:
        del sys.modules[module_name]
        print(f"🔄 Cleared cached module: {module_name}")
    
    return project_root


# Auto-setup when imported (but not when run directly)
if __name__ != "__main__":
    setup_local_pywhyllm()
