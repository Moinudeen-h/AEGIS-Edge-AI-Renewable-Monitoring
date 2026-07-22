import json
from pathlib import Path

root = Path(r"C:\Users\MOINODHEEN\Moinu\AEGIS_Project")

dirs = [
    "results",
    "figures",
    "tables"
]

notebooks = [
    "00_config.ipynb",
    "01_data_audit.ipynb",
    "02_baselines.ipynb",
    "03_local_autoencoder.ipynb",
    "04_threshold_sensitivity.ipynb",
    "05_cross_site_transfer.ipynb",
    "06_federated_learning.ipynb",
    "07_quantization_edge_eval.ipynb",
    "08_esp32_live_validation.ipynb",
    "09_figures_tables.ipynb"
]

nb = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# AEGIS notebook\n",
                "\n",
                "Project: AEGIS rebuild for reproducible wind anomaly, federated learning, and edge deployment evaluation.\n"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

for d in dirs:
    (root / d).mkdir(parents=True, exist_ok=True)

for n in notebooks:
    p = root / n
    if not p.exists():
        p.write_text(json.dumps(nb, indent=2), encoding="utf-8")

print("Created folders:")
for d in dirs:
    print(" -", d)

print("\nCreated notebooks:")
for n in notebooks:
    print(" -", n)