#!/bin/bash
cd /opt/dlami/nvme/Jacob/wvsignclassification
git add checkpoints/*.pt
git commit -m "Emergency backup — $(date '+%Y-%m-%d %H:%M')"
git push origin main
