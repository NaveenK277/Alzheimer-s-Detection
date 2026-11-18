# ============================================
#   NeuroScan — train.py
#   Run this ONCE to train the model and save
#   the artifacts to backend/saved_model/
#
#   Usage:
#     cd backend
#     python train.py
# ============================================

import os
from model import train_and_save

# ─── Set your dataset path here ───
# Point this to the folder that contains:
#   NonDemented/
#   VeryMildDemented/
#   MildDemented/
#   ModerateDemented/

DATA_DIR = 'C:\Users\karja\Downloads\alzheimers-app-20260409T134143Z-3-001\alzheimers-app\uploads\AugmentedAlzheimerDataset 1000'
# Linux/Mac example:
# DATA_DIR = '/home/user/datasets/AugmentedAlzheimerDataset'

# ─── Validate path ───
if not os.path.isdir(DATA_DIR):
    print(f"❌ Dataset folder not found: {DATA_DIR}")
    print("   Please update DATA_DIR in train.py")
    exit(1)

# ─── Train ───
print("=" * 50)
print("  NeuroScan — Model Training")
print("=" * 50)

best_name, metrics = train_and_save(DATA_DIR)

print("\n" + "=" * 50)
print(f"  ✅ Training complete!")
print(f"  Best model : {best_name}")
print(f"  Test Acc   : {metrics['test_acc']:.4f}")
print(f"  Test F1    : {metrics['test_f1_macro']:.4f}")
print("=" * 50)
print("\nYou can now run: python app.py")