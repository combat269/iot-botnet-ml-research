# IoT Botnet ML Research

This repository contains my initial research work on ML-based IoT botnet detection for edge/gateway deployment.

## Current focus
- Dataset: N-BaIoT
- Device: Danmini Doorbell
- Task: Binary classification (benign vs malicious)
- Initial files:
  - `benign_traffic.csv`
  - `mirai_attacks/udp.csv`

## Progress
### Day 1
- Reviewed core ML concepts for the project
- Selected first dataset and attack file
- Loaded and inspected data
- Added binary labels
- Created a balanced sampled subset
- Built `X` and `y`
- Completed train/test split

## Planned next steps
- Train first baseline classifier
- Evaluate with confusion matrix, precision, recall, and F1-score
- Compare lightweight models for edge feasibility