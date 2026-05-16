#!/usr/bin/env python3
"""
Customer Churn Prediction Migration Script
Migrates from 'Customer Churn Prediction/' to 'customer-churn-prediction/'
"""

import os
import sys
import shutil
import subprocess

# Colors
G = '\033[92m'  # Green
R = '\033[91m'  # Red
Y = '\033[93m'  # Yellow
B = '\033[94m'  # Blue
E = '\033[0m'   # End

OLD = "Customer Churn Prediction"
NEW = "customer-churn-prediction"


def check_git():
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
        return True
    except:
        return False


def step(n, msg):
    print(f"{B}[STEP {n}/6]{E} {msg}")


def ok(msg):
    print(f"{G}✓{E} {msg}")


def err(msg):
    print(f"{R}✗{E} {msg}")


def warn(msg):
    print(f"{Y}⚠{E}  {msg}")


def main():
    git_ok = check_git()
    files = 0
    
    print(f"\n{B}{'='*60}{E}")
    print("Customer Churn Prediction Migration Script")
    print(f"{B}{'='*60}{E}\n")
    
    # Step 1
    step(1, "Verifying prerequisites...")
    if not os.path.isdir(OLD):
        err(f"Folder '{OLD}' not found!")
        sys.exit(1)
    ok(f"Found '{OLD}'")
    
    # Step 2
    step(2, "Creating directories...")
    for d in ["notebooks", "models", "dataset"]:
        os.makedirs(f"{NEW}/{d}", exist_ok=True)
    ok(f"Created '{NEW}/'")
    
    # Step 3
    step(3, "Copying notebooks...")
    nb = f"{OLD}/notebooks"
    if os.path.isdir(nb):
        en = f"{nb}/Customer Churn Prediction EN.ipynb"
        if os.path.isfile(en):
            shutil.copy2(en, f"{NEW}/notebooks/customer-churn-prediction-en.ipynb")
            ok("customer-churn-prediction-en.ipynb")
            files += 1
        
        pt = f"{nb}/Customer Churn Prediction PT_Portuguese_PT.ipynb"
        if os.path.isfile(pt):
            shutil.copy2(pt, f"{NEW}/notebooks/customer-churn-prediction-pt.ipynb")
            ok("customer-churn-prediction-pt.ipynb")
            files += 1
    
    # Step 4
    step(4, "Copying models...")
    md = f"{OLD}/models"
    if os.path.isdir(md):
        count = 0
        for item in os.listdir(md):
            src = f"{md}/{item}"
            if os.path.isfile(src):
                shutil.copy2(src, f"{NEW}/models/{item}")
                count += 1
                files += 1
        ok(f"Copied {count} files")
    
    # Step 5
    step(5, "Copying additional files...")
    if os.path.isfile(f"{OLD}/README.md"):
        shutil.copy2(f"{OLD}/README.md", f"{NEW}/README.md")
        ok("Copied README.md")
        files += 1
    
    # Step 6
    step(6, "Git operations...")
    if git_ok:
        try:
            subprocess.run(["git", "add", NEW], capture_output=True, check=True)
            ok("Staged files")
            subprocess.run(["git", "commit", "-m", "refactor: migrate with clean URLs"], 
                         capture_output=True, check=True)
            ok("Created commit")
        except:
            warn("Git error")
    else:
        warn("Git not found")
    
    # Summary
    print(f"\n{B}{'='*60}{E}")
    print("Migration Complete")
    print(f"{B}{'='*60}{E}\n")
    ok(f"Created '{NEW}/' with {files} files")
    print(f"\n{B}Next Steps:{E}")
    print("  1. git status")
    print("  2. git push origin main")
    print(f"  3. git rm -r '{OLD}'")
    print("  4. git push origin main")
    print(f"\n{G}✓ Done!{E}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{R}Interrupted{E}\n")
        sys.exit(1)
