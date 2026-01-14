using Dates
using CSV
using DataFrames

include("funcion.jl")

# ============================================================
# CONFIGURACIÓN DE EXPERIMENTOS
# ============================================================

DATASET_DIR = "data"

WINDOW_SIZES = [5, 10, 20, 40]
HORIZONS     = [1, 3, 5]
TEST_RATIO   = 0.2

# Nombre del fichero de resultados

RESULTS_DIR = "resultados_experimentos"

if !isdir(RESULTS_DIR)
    mkdir(RESULTS_DIR)
end

timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
RESULTS_FILE = joinpath(
    RESULTS_DIR,
    "results_electricdevices_" * timestamp * ".csv"
)

# DataFrame para guardar resultados
results_df = DataFrame(
    window_size = Int[],
    horizon     = Int[],
    test_ratio  = Float64[],
    mse         = Float64[]
)

println("Iniciando experimentos...")
println("Resultados se guardarán en: ", RESULTS_FILE)
println("===================================")

# ============================================================
# BARRIDO DE EXPERIMENTOS
# ============================================================

for w in WINDOW_SIZES
    for h in HORIZONS

        println("Ejecutando experimento: window=$w, horizon=$h")

        mse = run_pipeline(
            data_dir = DATASET_DIR,
            window_size = w,
            horizon = h,
            test_ratio = TEST_RATIO
        )

        push!(results_df, (
            window_size = w,
            horizon     = h,
            test_ratio  = TEST_RATIO,
            mse         = mse
        ))

        println("Resultado -> MSE = ", mse)
        println("-----------------------------------")
    end
end

println("Experimentos finalizados.")
println("===================================")

# ============================================================
# GUARDAR RESULTADOS
# ============================================================

CSV.write(RESULTS_FILE, results_df)

println("Resultados guardados en ", RESULTS_FILE)
