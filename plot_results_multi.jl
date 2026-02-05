# Headless para HPC: ponerlo ANTES de cargar Plots/GR
ENV["GKSwstype"] = "100"

using CSV
using DataFrames
using Plots
using Statistics

gr()

"""
Plot de resultados para DoME (compatible con tu pipeline nuevo).

ENTRADA:
- Un archivo *_predictions.csv (si guardas predicciones en una segunda pasada)

Soporta columnas:
- Formato legacy: y_true, y_pred, error
- Formato nuevo (preferente): y_true_raw, y_pred_raw, error_raw
- También soporta norm: y_true_norm, y_pred_norm, error_norm

Por defecto usa RAW si existe; si no, NORM; si no, legacy.
"""

# =============================================================================
# UTIL: normalizar nombres de columnas a y_true/y_pred/error
# =============================================================================
function standardize_predictions(pred_df::DataFrame; mode::Symbol=:auto)
    cols = Set(names(pred_df))

    # 1) legacy ya ok
    if all(in.(["y_true","y_pred","error"], Ref(cols)))
        return pred_df, "legacy"
    end

    # 2) decidir modo
    chosen = mode
    if mode == :auto
        if all(in.(["y_true_raw","y_pred_raw"], Ref(cols)))
            chosen = :raw
        elseif all(in.(["y_true_norm","y_pred_norm"], Ref(cols)))
            chosen = :norm
        else
            chosen = :none
        end
    end

    if chosen == :raw
        y_true = pred_df.y_true_raw
        y_pred = pred_df.y_pred_raw
        # si existe error_raw lo ignoramos para fijar signo consistente (y - ŷ)
        err = y_true .- y_pred
        df = DataFrame(y_true=y_true, y_pred=y_pred, error=err)
        return df, "raw"

    elseif chosen == :norm
        y_true = pred_df.y_true_norm
        y_pred = pred_df.y_pred_norm
        err = y_true .- y_pred
        df = DataFrame(y_true=y_true, y_pred=y_pred, error=err)
        return df, "norm"
    end

    error("No encuentro columnas compatibles. Esperaba y_true/y_pred/error o y_true_raw/y_pred_raw o y_true_norm/y_pred_norm.")
end

# =============================================================================
# FUNCIONES DE ANÁLISIS
# =============================================================================

function rolling_mean(x::AbstractVector{<:Real}, window::Int)
    x = Float64.(x)
    n = length(x)
    n < window && return x
    result = zeros(n)
    for i in 1:n
        start_idx = max(1, i - window + 1)
        result[i] = mean(x[start_idx:i])
    end
    return result
end

function mse_by_blocks(errors::AbstractVector{<:Real}, n_blocks::Int=10)
    errors = Float64.(errors)
    n = length(errors)
    block_size = div(n, n_blocks)

    if block_size < 1
        return [mean(errors.^2)], [n/2]
    end

    block_mses = Float64[]
    block_centers = Float64[]

    for i in 1:n_blocks
        start_idx = (i-1) * block_size + 1
        end_idx = (i == n_blocks) ? n : i * block_size
        if end_idx >= start_idx
            block_errors = errors[start_idx:end_idx]
            push!(block_mses, mean(block_errors.^2))
            push!(block_centers, (start_idx + end_idx) / 2)
        end
    end

    return block_mses, block_centers
end

function drift_statistics(errors::AbstractVector{<:Real})
    errors = Float64.(errors)
    n = length(errors)
    t = collect(1:n)
    squared_errors = errors.^2

    t_mean = mean(t)
    y_mean = mean(squared_errors)
    b = sum((t .- t_mean) .* (squared_errors .- y_mean)) / sum((t .- t_mean).^2)
    a = y_mean - b * t_mean
    corr_val = cor(t, squared_errors)

    return Dict(
        "slope" => b,
        "intercept" => a,
        "correlation" => corr_val,
        "trend_direction" => b > 0 ? "creciente" : "decreciente"
    )
end

# =============================================================================
# FUNCIONES DE GRAFICADO
# =============================================================================

function plot_temporal_error(pred_df::DataFrame, window::Int=50)
    errors = pred_df.error
    abs_errors = abs.(errors)
    squared_errors = errors.^2
    n = length(errors)
    t = 1:n

    rolling_abs = rolling_mean(abs_errors, window)
    rolling_sq  = rolling_mean(squared_errors, window)

    p1 = plot(t, abs_errors,
        label="|y-ŷ|", alpha=0.3, color=:gray,
        xlabel="Índice temporal (test)", ylabel="Error absoluto",
        title="Error absoluto (drift)", legend=:topright, linewidth=0.5
    )
    plot!(p1, t, rolling_abs, label="Media móvil (w=$window)", color=:red, linewidth=2)

    p2 = plot(t, squared_errors,
        label="(y-ŷ)²", alpha=0.3, color=:gray,
        xlabel="Índice temporal (test)", ylabel="Error cuadrático",
        title="Error cuadrático (MSE puntual)", legend=:topright, linewidth=0.5
    )
    plot!(p2, t, rolling_sq, label="Media móvil (w=$window)", color=:blue, linewidth=2)

    mse_global = mean(squared_errors)
    hline!(p2, [mse_global], label="MSE global", color=:green, linestyle=:dash, linewidth=2)

    return plot(p1, p2, layout=(2,1), size=(1000, 800))
end

function plot_mse_blocks(pred_df::DataFrame, n_blocks::Int=10)
    errors = pred_df.error
    block_mses, block_centers = mse_by_blocks(errors, n_blocks)

    p = bar(block_centers, block_mses,
        xlabel="Centro del bloque", ylabel="MSE",
        title="MSE por bloques (n=$n_blocks)", legend=false,
        color=:steelblue, alpha=0.7
    )

    mse_global = mean(errors.^2)
    hline!(p, [mse_global], label="MSE global", color=:red, linestyle=:dash, linewidth=2, legend=:topright)
    return p
end

function plot_error_histogram(pred_df::DataFrame)
    errors = pred_df.error
    p = histogram(errors,
        bins=50, xlabel="Error (y-ŷ)", ylabel="Frecuencia",
        title="Distribución de errores", legend=false,
        color=:steelblue, alpha=0.7, normalize=:probability
    )
    vline!(p, [0], color=:red, linestyle=:dash, linewidth=2)
    annotate!(p, :topright, text("Media: $(round(mean(errors),digits=4))\nStd: $(round(std(errors),digits=4))", :left, 8))
    return p
end

function plot_prediction_scatter(pred_df::DataFrame)
    y_true = pred_df.y_true
    y_pred = pred_df.y_pred

    min_val = min(minimum(y_true), minimum(y_pred))
    max_val = max(maximum(y_true), maximum(y_pred))

    p = scatter(y_true, y_pred,
        xlabel="y_true", ylabel="y_pred",
        title="Predicción vs Real", legend=:topleft,
        label="Puntos", alpha=0.5, markersize=3, color=:steelblue
    )
    plot!(p, [min_val, max_val], [min_val, max_val], label="Ideal", color=:red, linestyle=:dash, linewidth=2)

    ss_res = sum((y_true .- y_pred).^2)
    ss_tot = sum((y_true .- mean(y_true)).^2)
    r2_text = ss_tot > 0 ? "R² = $(round(1 - ss_res/ss_tot, digits=4))" : "R² = N/A (y constante)"
    annotate!(p, :topleft, text(r2_text, :left, 10))

    return p
end

function plot_time_series(pred_df::DataFrame, max_points::Int=500)
    n = nrow(pred_df)
    indices = n > max_points ? (1:max(1, cld(n, max_points)):n) : (1:n)

    y_true = pred_df.y_true[indices]
    y_pred = pred_df.y_pred[indices]
    t = collect(indices)

    p = plot(t, y_true, label="Real", color=:black, linewidth=2,
        xlabel="Índice temporal (test)", ylabel="Valor",
        title="Serie temporal (submuestreada)", legend=:topright
    )
    plot!(p, t, y_pred, label="Predicción", color=:red, linewidth=1.5, linestyle=:dash)
    return p
end

# =============================================================================
# MAIN
# =============================================================================

function plot_experiment_diagnostics(
    predictions_file::String;
    output_file::Union{String,Nothing}=nothing,
    rolling_window::Int=50,
    n_blocks::Int=10,
    mode::Symbol=:auto   # :auto | :raw | :norm
)
    println("="^70)
    println("GENERANDO GRÁFICAS")
    println("="^70)
    println("Predictions file: $predictions_file")

    isfile(predictions_file) || error("Archivo no encontrado: $predictions_file")

    raw_df = CSV.read(predictions_file, DataFrame)
    pred_df, used = standardize_predictions(raw_df; mode=mode)
    println("  Filas: $(nrow(pred_df)) | Usando modo: $used")

    drift_stats = drift_statistics(pred_df.error)
    println("  Pendiente (MSE): $(round(drift_stats["slope"], digits=8))")
    println("  Corr(t, MSE):    $(round(drift_stats["correlation"], digits=4))")
    println("  Tendencia:       $(drift_stats["trend_direction"])")

    if output_file === nothing
        base = replace(predictions_file, ".csv" => "")
        output_file = "$(base)_diagnostics.pdf"
    end

    p1 = plot_temporal_error(pred_df, rolling_window)
    p2 = plot_mse_blocks(pred_df, n_blocks)
    p3 = plot_error_histogram(pred_df)
    p4 = plot_prediction_scatter(pred_df)
    p5 = plot_time_series(pred_df)

    final_plot = plot(p1, p2, p3, p4, p5, layout=(5,1), size=(1200, 2400))
    savefig(final_plot, output_file)

    println("✓ Guardado: $output_file")
    println("="^70)
    return drift_stats
end

function plot_all_experiments(results_dir::String="results"; rolling_window::Int=50, n_blocks::Int=10, mode::Symbol=:auto)
    isdir(results_dir) || error("Directorio no encontrado: $results_dir")
    pred_files = filter(f -> endswith(f, "_predictions.csv"), readdir(results_dir, join=true))

    if isempty(pred_files)
        println("No hay *_predictions.csv en $results_dir")
        return
    end

    drift_summary = DataFrame(file=String[], slope=Float64[], correlation=Float64[], trend=String[], mode=String[])

    for (i, f) in enumerate(pred_files)
        println("\n[$i/$(length(pred_files))] $(basename(f))")
        try
            stats = plot_experiment_diagnostics(f; rolling_window=rolling_window, n_blocks=n_blocks, mode=mode)
            push!(drift_summary, (basename(f), stats["slope"], stats["correlation"], stats["trend_direction"], string(mode)))
        catch e
            println("  ✗ ERROR: $e")
        end
    end

    summary_file = joinpath(results_dir, "drift_summary.csv")
    CSV.write(summary_file, drift_summary)
    println("\n✓ Resumen drift: $summary_file")
    return drift_summary
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) == 0
        println("Uso:")
        println("  julia plot_results_multi.jl <predictions_file.csv> [raw|norm]")
        println("  julia plot_results_multi.jl --all [raw|norm]")
        exit(1)
    end

    mode = :auto
    if length(ARGS) >= 2
        if ARGS[2] == "raw"
            mode = :raw
        elseif ARGS[2] == "norm"
            mode = :norm
        end
    end

    if ARGS[1] == "--all"
        plot_all_experiments("results"; mode=mode)
    else
        plot_experiment_diagnostics(ARGS[1]; mode=mode)
    end
end
