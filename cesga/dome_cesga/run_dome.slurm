#!/bin/bash
#SBATCH -J dome_ETTh2
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=92:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

# Entorno CESGA correcto para Julia 1.10.7
module --force purge
module load cesga/2022
module load julia/1.10.7

# Evita problemas con stdlib/LOAD_PATH y usa depot en STORE (conservando el depot por defecto con ':')
unset JULIA_LOAD_PATH
export JULIA_DEPOT_PATH="$STORE/julia_depot:"
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results

# (opcional pero útil) info de entorno en logs
echo "HOST=$(hostname)"
echo "PWD=$PWD"
julia -e 'println("Julia ", VERSION)'

# Asegura que existe SymDoME.jl/ (el Manifest lo referencia por ruta)
if [ ! -d "SymDoME.jl" ]; then
  echo "Descomprimiendo SymDoME.zip -> SymDoME.jl/"
  python3 - <<'PY'
import os, zipfile
zip_path = "SymDoME.zip"
out_dir = "."
if not os.path.exists(zip_path):
    raise SystemExit(f"No existe {zip_path}")
with zipfile.ZipFile(zip_path, 'r') as z:
    z.extractall(out_dir)
print("OK: extraído", zip_path)
PY
fi

julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

julia --project=. sweep_experiments_multi.jl "ETT/ETTh2.csv"
