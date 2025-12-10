#!/usr/bin/env python3
"""
Run all DeFi swapper analysis notebooks in order.

This script executes:
1. butterswap_data_analyzer.ipynb
2. chainflip_volume_analyzer.ipynb
3. thorchain_data_combiner.ipynb
4. combined_swappers_analyzer.ipynb (runs last, combines all swapper data)

Note: master_twap_threshold_analysis.ipynb is a separate analysis and not included here.

Usage:
    python run_all_notebooks.py
"""

import subprocess
import sys
from pathlib import Path
import time

# Notebook execution order - 3 swappers first, then combined analyzer
NOTEBOOKS = [
    "notebooks/butterswap_data_analyzer.ipynb",
    "notebooks/chainflip_volume_analyzer.ipynb",
    "notebooks/thorchain_data_combiner.ipynb",
    "notebooks/combined_swappers_analyzer.ipynb",  # Runs last, combines all swapper data
]

def run_notebook(notebook_path: Path) -> bool:
    """
    Execute a Jupyter notebook using nbconvert.
    
    Parameters:
    -----------
    notebook_path : Path
        Path to the notebook file
    
    Returns:
    --------
    bool
        True if execution succeeded, False otherwise
    """
    print(f"\n{'='*70}")
    print(f"Running: {notebook_path.name}")
    print(f"{'='*70}")
    
    try:
        # Try to find jupyter - check common locations
        jupyter_paths = [
            "/opt/anaconda3/bin/jupyter",
            "jupyter",
            f"{sys.executable} -m jupyter",
        ]
        
        jupyter_cmd = None
        for jp in jupyter_paths:
            try:
                test_result = subprocess.run(
                    [jp.split()[0] if " " in jp else jp, "--version"],
                    capture_output=True,
                    timeout=5
                )
                if test_result.returncode == 0:
                    jupyter_cmd = jp.split() if " " in jp else [jp]
                    break
            except:
                continue
        
        if not jupyter_cmd:
            # Fallback to python -m jupyter
            jupyter_cmd = [sys.executable, "-m", "jupyter"]
        
        # Execute notebook using nbconvert
        cmd = jupyter_cmd + [
            "nbconvert",
            "--to", "notebook",
            "--execute",
            "--inplace",
            "--ExecutePreprocessor.timeout=600",  # 10 minute timeout
            str(notebook_path)
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        print(f"✓ Successfully executed {notebook_path.name}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Error executing {notebook_path.name}")
        print(f"  Return code: {e.returncode}")
        if e.stdout:
            print(f"  stdout: {e.stdout}")
        if e.stderr:
            print(f"  stderr: {e.stderr}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def main():
    """Main execution function"""
    import os
    
    # Get repository root directory
    repo_root = Path(__file__).parent.resolve()
    
    print("="*70)
    print("DeFi Analysis Notebook Runner")
    print("="*70)
    print(f"Repository root: {repo_root}")
    print(f"Python: {sys.executable}")
    print(f"Python version: {sys.version}")
    
    # Change to repository root
    original_dir = Path.cwd()
    os.chdir(repo_root)
    
    try:
        # Check that notebooks exist (combined analyzer is optional)
        missing_notebooks = []
        required_notebooks = NOTEBOOKS[:-1]  # All except the last one (combined analyzer)
        optional_notebooks = [NOTEBOOKS[-1]] if len(NOTEBOOKS) > 0 else []
        
        for notebook in required_notebooks:
            notebook_path = repo_root / notebook
            if not notebook_path.exists():
                missing_notebooks.append(str(notebook_path))
        
        if missing_notebooks:
            print("\n✗ Missing required notebooks:")
            for nb in missing_notebooks:
                print(f"  - {nb}")
            return 1
        
        # Check optional notebooks
        for notebook in optional_notebooks:
            notebook_path = repo_root / notebook
            if not notebook_path.exists():
                print(f"\n⚠ Optional notebook not found: {notebook_path.name}")
                print(f"  Skipping combined analyzer (create it to combine all swapper data)")
                # Remove from list
                NOTEBOOKS.remove(notebook)
        
        # Run notebooks in order
        results = {}
        start_time = time.time()
        
        for i, notebook in enumerate(NOTEBOOKS, 1):
            notebook_path = repo_root / notebook
            print(f"\n[{i}/{len(NOTEBOOKS)}] Processing {notebook_path.name}...")
            
            success = run_notebook(notebook_path)
            results[notebook] = success
            
            if not success:
                print(f"\n⚠ Stopping execution due to error in {notebook_path.name}")
                print("  Fix the error and re-run this script to continue.")
                break
            
            # Small delay between notebooks
            if i < len(NOTEBOOKS):
                time.sleep(2)
        
        # Summary
        elapsed_time = time.time() - start_time
        print(f"\n{'='*70}")
        print("EXECUTION SUMMARY")
        print(f"{'='*70}")
        
        for notebook, success in results.items():
            status = "✓ SUCCESS" if success else "✗ FAILED"
            print(f"{status}: {Path(notebook).name}")
        
        all_success = all(results.values())
        print(f"\nTotal time: {elapsed_time:.1f} seconds")
        
        if all_success:
            print("\n✓ All notebooks executed successfully!")
            return 0
        else:
            print("\n✗ Some notebooks failed. Check output above for details.")
            return 1
            
    except KeyboardInterrupt:
        print("\n\n⚠ Execution interrupted by user")
        return 130
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Restore original directory
        os.chdir(original_dir)

if __name__ == "__main__":
    sys.exit(main())
