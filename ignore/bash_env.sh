#!/bin/bash
#SBATCH --job-name=Install_Conda
#SBATCH --output=Install_Conda_%j.log
#SBATCH --error=Install_Conda_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=THIN
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# Cancella eventuali file precedenti
rm -f Miniconda3-latest-Linux-x86_64.sh

# Scarica Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Verifica il checksum (opzionale)
EXPECTED_CHECKSUM="2bf16dc434370d374e9eb22b5b2f6555"
DOWNLOADED_CHECKSUM=$(md5sum Miniconda3-latest-Linux-x86_64.sh | awk '{ print $1 }')

if [ "$EXPECTED_CHECKSUM" != "$DOWNLOADED_CHECKSUM" ]; then
    echo "Error: Checksum mismatch. Downloaded file is corrupt."
    exit 1
fi

# Esegui l'installazione
bash Miniconda3-latest-Linux-x86_64.sh -b -p /u/group/user/scratch/miniconda

# Aggiungi Miniconda al PATH
export PATH=/u/group/user/scratch/miniconda/bin:$PATH

# Disattiva l'attivazione automatica dell'ambiente base
conda config --set auto_activate_base false

# Crea un nuovo ambiente Conda
conda create --name deep_learning python=3.9 -y

# Attiva l'ambiente
source activate deep_learning

# Installa i pacchetti necessari
conda install tensorflow keras numpy pennylane -y

# Verifica l'installazione
python -c "import tensorflow as tf; import keras; import numpy as np; import pennylane as qml; print('All packages are correctly installed')"

# Esegui lo script Python
python ../QAOA.py
