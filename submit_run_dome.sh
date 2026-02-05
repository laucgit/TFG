#!/bin/bash
set -euo pipefail

# Ir a la carpeta donde está este script (raíz del proyecto)
cd "$(dirname "$0")"

# Crear carpetas necesarias
mkdir -p logs results

# Si estamos en un sistema con Slurm (CESGA), enviar el job
if command -v sbatch >/dev/null 2>&1; then
  sbatch run_dome.slurm
else
  echo "sbatch no encontrado: esto no es Slurm (no CESGA)."
  echo "Si quieres ejecutar en local, usa:"
  echo '  julia --project=. sweep_experiments_multi.jl "ETT/ETTh2.csv"'
  exit 1
fi
