include("run_experiment_multi.jl")

using Dates
using Random
using JLD2
using CSV
using DataFrames
using Statistics

# -------------------------
# Utilidades JLD2
# -------------------------
# En JLD2, asignar a una clave existente NO la sobreescribe: hay que borrarla primero.
function jld_overwrite!(f, key::AbstractString, value)
    if haskey(f, key)
        delete!(f, key)
    end
    f[key] = value
    return nothing
end

"""
SWEEP DoME (estilo Dani): 1 dataset -> 1 fichero JLD2.

- Pensado para CESGA: lanza 1 job por dataset (evitas escrituras concurrentes).
- Guarda por configuración y seed:
    * MSE train/test, stop_reason, iterations
    * tiempos: train_time_sec, total_time_sec
    * expression (String) + history (si SAVE_DETAILED=true)
- (Opcional) guarda predicciones en CSV (NO recomendado para sweep completo).

Uso:
  julia sweep_experiments_multi.jl ETT/ETTh2.csv
  julia sweep_experiments_multi.jl --all
"""

# -------------------------
# Configuración
# -------------------------
datasets_config = Dict(
    "ElectricDevices/LD2011_2014.txt" => Dict("target_col" => "MT_196", "windows" => [12, 24, 48]),
    "ETT/ETTh2.csv" => Dict("target_col" => "OT", "windows" => [12, 24, 48]),
    "ETT/ETTm1.csv" => Dict("target_col" => "OT", "windows" => [12, 24, 48]),
    "LCDS/LCD_USW00094789_2024.csv" => Dict("target_col" => "HourlyDryBulbTemperature", "windows" => [12, 24, 48])
)

normalization = "MaxMin"
model = :DoME

# DoME es determinista
seeds = 1:1

# Guardar expression+history en el JLD2
SAVE_DETAILED = true

# Guardar predicciones por experimento
SAVE_PREDICTIONS = false

# -------------------------
# Selección de datasets por CLI
# -------------------------
selected_datasets = String[]
if length(ARGS) >= 1 && ARGS[1] != "--all"
    push!(selected_datasets, ARGS[1])
else
    selected_datasets = collect(keys(datasets_config))
end

for d in selected_datasets
    haskey(datasets_config, d) || error("Dataset no está en datasets_config: $d")
end

# -------------------------
# Salida
# -------------------------
results_dir = "results"
isdir(results_dir) || mkdir(results_dir)

jobtag = get(ENV, "SLURM_JOB_ID", string(getpid()))
timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")

# -------------------------
# Configs (ojo: NO guardamos la Function en el JLD2)
# -------------------------
model_configs = modelConfigurations(model)
configs_serializable = [
    Dict(
        "config_id" => i,
        "max_nodes" => cfg["maxNumNodes"],
        "min_improvement" => cfg["minimumReductionMSE"],
        "use_division" => cfg["useDivisionOperator"],
        "strategy" => cfg["strategyName"]
    )
    for (i, cfg) in enumerate(model_configs)
]

# -------------------------
# Sweep por dataset
# -------------------------
for dataset in selected_datasets
    dataset_conf = datasets_config[dataset]
    windows = dataset_conf["windows"]
    target_col = dataset_conf["target_col"]

    dataset_name = replace(split(dataset, "/")[end], ".csv" => "", ".txt" => "")

    outpath = joinpath(results_dir, "$(dataset_name)__$(timestamp)__$(jobtag).jld2")

    predictions_dir = nothing
    if SAVE_PREDICTIONS
        predictions_dir = joinpath(results_dir, "predictions_$(dataset_name)__$(timestamp)__$(jobtag)")
        isdir(predictions_dir) || mkdir(predictions_dir)
    end

    metadata = Dict(
        "dataset" => dataset,
        "dataset_name" => dataset_name,
        "target_col" => target_col,
        "windows" => windows,
        "seeds" => collect(seeds),
        "normalization" => normalization,
        "model" => string(model),
        "configs" => configs_serializable,
        "created_at" => string(Dates.now()),
        "jobtag" => jobtag
    )

    total_experiments = length(windows) * length(model_configs) * length(seeds)

    println("\n" * "="^70)
    println("SWEEP DoME (JLD2)")
    println("="^70)
    println("Dataset: $dataset")
    println("Output:  $outpath")
    println("Total experiments: $total_experiments | configs=$(length(model_configs)) | seeds=$(length(seeds))")
    println("Detailed: $(SAVE_DETAILED) | Predictions: $(SAVE_PREDICTIONS ? string(predictions_dir) : "(desactivado)")")
    println("="^70)

    start_time = time()
    counter = 0

    # Abrimos el JLD2 una vez y vamos guardando por clave -> sin reescribir todo el fichero cada vez
    jldopen(outpath, "w") do f
        jld_overwrite!(f, "metadata", metadata)
        jld_overwrite!(f, "progress/started_at", string(Dates.now()))
    end

    jldopen(outpath, "a") do f
        for window in windows
            for config_id in 1:length(model_configs)

                key = "results/w$(window)/c$(config_id)"
                if haskey(f, key)
                    # Resume: si ya existe, saltamos
                    continue
                end

                per_seed = Vector{Any}(undef, length(seeds))
                mse_tests = Float64[]
                n_success = 0

                # Para guardar hiperparámetros sin tocar la Function del cfg
                max_nodes_val = missing
                min_impr_val  = missing
                use_div_val   = missing
                strat_val     = missing

                for (si, seed) in enumerate(seeds)
                    counter += 1
                    try
                        res_df, det, preds = run_experiment(
                            dataset=dataset,
                            window=window,
                            normalization=normalization,
                            model=model,
                            config_id=config_id,
                            seed=seed,
                            target_col=target_col,
                            save_predictions=SAVE_PREDICTIONS,
                            save_detailed=SAVE_DETAILED,
                            predictions_dir=predictions_dir,
                            verbose=false
                        )

                        # Captura hiperparámetros (solo una vez)
                        if n_success == 0
                            max_nodes_val = Int(res_df.max_nodes[1])
                            min_impr_val  = Float64(res_df.min_improvement[1])
                            use_div_val   = Bool(res_df.use_division[1])
                            strat_val     = String(res_df.strategy[1])
                        end

                        mse_test_raw = Float64(res_df.mse_test_raw[1])
                        push!(mse_tests, mse_test_raw)
                        n_success += 1

                        expr = ""
                        hist = Float64[]
                        if SAVE_DETAILED && det !== nothing
                            expr = String(det["expression"])
                            hist = det["history"]
                        end

                        per_seed[si] = Dict(
                            "seed" => seed,
                            "mse_initial" => Float64(res_df.mse_initial[1]),
                            "mse_final_train" => Float64(res_df.mse_final_train[1]),
                            "mse_test_norm" => Float64(res_df.mse_test_norm[1]),
                            "mse_test_raw" => mse_test_raw,
                            "improvement_pct" => Float64(res_df.improvement_pct[1]),
                            "iterations" => Int(res_df.iterations[1]),
                            "stop_reason" => String(res_df.stop_reason[1]),
                            "train_time_sec" => Float64(res_df.train_time_sec[1]),
                            "total_time_sec" => Float64(res_df.total_time_sec[1]),
                            "expression" => expr,
                            "history" => hist
                        )

                        # Predicciones (1 CSV por experimento) - opcional
                        if SAVE_PREDICTIONS && preds !== nothing
                            pred_path = joinpath(
                                predictions_dir,
                                "$(dataset_name)_w$(window)_c$(config_id)_s$(seed)_predictions.csv"
                            )
                            CSV.write(pred_path, preds)
                        end

                    catch e
                        per_seed[si] = Dict(
                            "seed" => seed,
                            "error" => string(e),
                            "timestamp" => string(Dates.now())
                        )
                    end

                    if counter % 50 == 0
                        elapsed = time() - start_time
                        println("Progreso: $counter/$total_experiments | $(round(elapsed/60, digits=2)) min")
                        jld_overwrite!(f, "progress/last_updated", string(Dates.now()))
                        jld_overwrite!(f, "progress/counter", counter)
                    end
                end

                entry = Dict(
                    "dataset" => dataset,
                    "window" => window,
                    "config_id" => config_id,

                    "max_nodes" => max_nodes_val,
                    "min_improvement" => min_impr_val,
                    "use_division" => use_div_val,
                    "strategy" => strat_val,

                    "n_success" => n_success,
                    "n_seeds" => length(seeds),
                    "mse_test_raw_mean" => (n_success > 0 ? mean(mse_tests) : Inf),
                    "mse_test_raw_min"  => (n_success > 0 ? minimum(mse_tests) : Inf),

                    "per_seed" => per_seed,
                    "saved_at" => string(Dates.now())
                )

                f[key] = entry
            end
        end

        jld_overwrite!(f, "progress/finished_at", string(Dates.now()))
        jld_overwrite!(f, "progress/counter", counter)
    end

    elapsed = time() - start_time
    println("="^70)
    println("SWEEP COMPLETADO | $(round(elapsed/60, digits=2)) min")
    println("Output: $outpath")
    println("="^70)
end
