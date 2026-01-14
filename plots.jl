using CSV
using DataFrames
using Plots

gr()

# ============================================================
# CARGAR RESULTADOS
# ============================================================

RESULTS_DIR = "resultados_experimentos"
PLOTS_DIR   = "plots"

# Crear carpeta de plots si no existe
if !isdir(PLOTS_DIR)
    mkdir(PLOTS_DIR)
end

# Coge el último CSV generado
files = filter(f -> endswith(f, ".csv"), readdir(RESULTS_DIR))
@assert !isempty(files) "No se encontraron CSVs en resultados_experimentos"

latest_file = sort(files)[end]
results_path = joinpath(RESULTS_DIR, latest_file)

println("Cargando resultados desde: ", results_path)

df = CSV.read(results_path, DataFrame)

# ============================================================
# GRÁFICA: MSE vs WINDOW SIZE (horizon = 1)
# ============================================================

df_h1 = df[df.horizon .== 1, :]

@assert nrow(df_h1) > 0 "No hay resultados con horizon = 1"

plot(
    df_h1.window_size,
    df_h1.mse,
    marker = :circle,
    xlabel = "Window size",
    ylabel = "MSE",
    title = "MSE vs Window size (horizon = 1)",
    legend = false
)

savefig(joinpath(PLOTS_DIR, "mse_vs_window_h1.png"))
println("Guardada gráfica plots/mse_vs_window_h1.png")

# ============================================================
# GRÁFICA: MSE vs HORIZON (window = 10)
# ============================================================

WINDOW_FIXED = 10
df_w10 = df[df.window_size .== WINDOW_FIXED, :]

@assert nrow(df_w10) > 0 "No hay resultados con window_size = $WINDOW_FIXED"

plot(
    df_w10.horizon,
    df_w10.mse,
    marker = :square,
    xlabel = "Horizon",
    ylabel = "MSE",
    title = "MSE vs Horizon (window = $WINDOW_FIXED)",
    legend = false
)

savefig(joinpath(PLOTS_DIR, "mse_vs_horizon_w10.png"))
println("Guardada gráfica plots/mse_vs_horizon_w10.png")
