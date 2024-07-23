#!/bin/bash
#SBATCH --job-name=QAOA_maxcut
#SBATCH --output=QAOA_maxcut_%j.log
#SBATCH --error=QAOA_maxcut_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=THIN
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# Carica i moduli necessari (aggiusta in base alla tua configurazione)
module load python/3.9
module load tensorflow
module load keras
module load numpy
module load pennylane

# Verifica che i moduli siano stati caricati correttamente
echo "Moduli caricati:"
module list

# Esegui lo script Python
python ../QAOA.py

