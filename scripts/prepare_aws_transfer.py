#!/usr/bin/env python3
"""
Print an AWS transfer checklist and write aws_setup.sh + requirements_aws.txt
to the project root.  Does NOT transfer any files.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CHECKLIST = """
============================================================
  AWS TRANSFER CHECKLIST
============================================================

1. INSTANCE REQUIREMENTS
   - GPU: g4dn.xlarge or better (T4 16 GB minimum)
   - AMI:  Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)
           (comes with CUDA 12.1, cuDNN, nvidia-smi pre-installed)
   - Disk: >= 200 GB gp3 (MTSD ~80 GB processed, LISA ~5 GB)
   - Python: 3.11 (system or conda)

2. CUDA VERSION REQUIRED
   - torch 2.12.0 was built against CUDA 12.1
   - Verify on the instance:  nvidia-smi  (driver >= 525)
                               nvcc --version  (CUDA 12.1)

3. FILES TO COPY TO THE INSTANCE
   Repo (scripts + configs, no data):
     rsync -av --exclude='data/' --exclude='checkpoints/' --exclude='runs/' \\
       wvsignclassification/  ec2-user@<HOST>:~/wvsignclassification/

   Processed LISA data:
     rsync -av data/processed/lisa/       ec2-user@<HOST>:~/wvsignclassification/data/processed/lisa/
     rsync -av data/processed/lisa_4class/ ec2-user@<HOST>:~/wvsignclassification/data/processed/lisa_4class/
     NOTE: lisa_4class/train/images is a Windows junction; rsync will follow
           it and copy real files.  On AWS, recreate it as a symlink:
             ln -sf ~/wvsignclassification/data/processed/lisa/train/images \\
                    ~/wvsignclassification/data/processed/lisa_4class/train/images

   Processed MTSD data:
     rsync -av data/processed/mtsd/  ec2-user@<HOST>:~/wvsignclassification/data/processed/mtsd/

   Phase-1 checkpoint (starting point for phase 2):
     scp checkpoints/phase1_mtsd_best.pt  ec2-user@<HOST>:~/wvsignclassification/checkpoints/

4. CONFIG FILES NEED PATH UPDATE
   configs/mtsd.yaml and configs/data.yaml have hardcoded Windows paths.
   After copying, fix them on the instance:
     sed -i 's|C:\\\\Users\\\\jrj00048\\\\Projects\\\\wvsignclassification|/home/ec2-user/wvsignclassification|g' \\
       ~/wvsignclassification/configs/mtsd.yaml \\
       ~/wvsignclassification/configs/data.yaml

5. PLATFORM NOTE - train_phase2_lisa.py uses Windows junctions
   The _junction() helper calls  cmd /c mklink /J  which does not exist on Linux.
   On the instance, replace it with a symlink:
     In scripts/train_phase2_lisa.py, change _junction() to:
       def _junction(link, target):
           if link.exists(): return
           link.parent.mkdir(parents=True, exist_ok=True)
           link.symlink_to(target.resolve())

6. RECOMMENDED TRAINING COMMANDS (on the instance)
   Phase 2 - full run (starts from phase1_mtsd_best.pt):
     python scripts/train_phase2_lisa.py --batch 32

   Eval:
     python scripts/eval_models.py

============================================================
"""


AWS_SETUP_SH = """\
#!/usr/bin/env bash
# Run this on a fresh AWS instance after copying the repo.
# Assumes Ubuntu 22.04, Python 3.11, CUDA 12.1 already present.
set -e

REPO="$HOME/wvsignclassification"

echo "=== Creating directory structure ==="
mkdir -p "$REPO/data/processed/lisa/train/images"
mkdir -p "$REPO/data/processed/lisa/train/labels"
mkdir -p "$REPO/data/processed/lisa_4class/train/labels"
mkdir -p "$REPO/data/processed/mtsd/train/images"
mkdir -p "$REPO/data/processed/mtsd/train/labels"
mkdir -p "$REPO/data/processed/mtsd/val/images"
mkdir -p "$REPO/data/processed/mtsd/val/labels"
mkdir -p "$REPO/checkpoints"
mkdir -p "$REPO/runs"

echo "=== Creating lisa_4class images symlink ==="
ln -sf "$REPO/data/processed/lisa/train/images" \\
       "$REPO/data/processed/lisa_4class/train/images"

echo "=== Installing Python packages ==="
pip install --upgrade pip
# PyTorch with CUDA 12.1
pip install torch==2.12.0+cu121 torchvision==0.27.0+cu121 \\
    --index-url https://download.pytorch.org/whl/cu121
# Remaining training deps
pip install -r "$REPO/requirements_aws.txt"

echo "=== Fixing config paths ==="
sed -i "s|C:\\\\\\\\Users\\\\\\\\jrj00048\\\\\\\\Projects\\\\\\\\wvsignclassification|$REPO|g" \\
    "$REPO/configs/mtsd.yaml" \\
    "$REPO/configs/data.yaml"

echo ""
echo "=== Setup complete. Next steps: ==="
echo "  1. Copy processed data into $REPO/data/processed/"
echo "     (lisa/, lisa_4class/train/labels/, mtsd/)"
echo "  2. Copy phase1_mtsd_best.pt into $REPO/checkpoints/"
echo "  3. Fix _junction() in train_phase2_lisa.py (see checklist)"
echo "  4. Run:  python scripts/train_phase2_lisa.py --batch 32"
"""


REQUIREMENTS_AWS = """\
# Minimal training requirements for AWS.
# Install PyTorch separately with the CUDA wheel (see aws_setup.sh).
ultralytics==8.4.60
numpy==2.4.6
opencv-python==4.13.0.92
PyYAML==6.0.3
pandas==3.0.3
Pillow==12.2.0
scipy==1.17.1
matplotlib==3.10.9
"""


def main():
    print(CHECKLIST)

    setup_path = ROOT / "aws_setup.sh"
    setup_path.write_text(AWS_SETUP_SH)
    print(f"Written: {setup_path}")

    req_path = ROOT / "requirements_aws.txt"
    req_path.write_text(REQUIREMENTS_AWS)
    print(f"Written: {req_path}")


if __name__ == "__main__":
    main()
