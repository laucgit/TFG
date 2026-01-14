using Dates

include("funcion.jl")

# ============================================================
# CONFIGURACIÓN DE EXPERIMENTOS
# ============================================================

DATASET_DIR = "data"

WINDOW_SIZES = [5, 10, 20, 40]
HORIZONS     = [1, 3, 5]
TEST_RATIO   = 0.2

# Para guardar resultados
results = Vector{NamedTuple}()

println("Iniciando experimentos...")
println("Fecha: ", now())
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

        push!(results, (
            window_size = w,
            horizon = h,
            mse = mse
        ))

        println("Resultado -> MSE = ", mse)
        println("-----------------------------------")
    end
end

println("Experimentos finalizados.")
println("===================================")

# ============================================================
# MOSTRAR RESULTADOS
# ============================================================

println("\nResumen de resultados:")
for r in results
    println(
        "window=", r.window_size,
        " | horizon=", r.horizon,
        " | MSE=", r.mse
    )
end
