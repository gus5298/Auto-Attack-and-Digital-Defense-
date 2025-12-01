# Auto Attack and Digital Defense for Siemens PLCs

This repository contains **all source code** for the Siemens digital defense project,
including attack and defense implementations for multiple PLC platforms.

## Repository Structure

- `Siemens-410E/`  
  Code and experiments for the Siemens 410E PLC (attack & defense).

- `SIMATIC-S7-1500/`  
  Code and experiments for the SIMATIC S7-1500 PLC.

- `Siemens-Energy-T3000/`  
  Code and experiments for the Siemens Energy T3000 system.

Each subfolder keeps the original project structure from the previous repositories.

## Environment Setup

```bash
# example, adapt to your actual setup
conda create -n siemens-defense python=3.10
conda activate siemens-defense
pip install -r requirements.txt
