include(joinpath(@__DIR__, "run_stream_experiment_multi.jl"))

using Dates
using JLD2
using DataFrames
using Statistics

# -------------------------
# Utilidades JLD2
# -------------------------
function jld_overwrite!(f, key::AbstractString, value)
    if haskey(f, key)
        delete!(f, key)
    end
    f[key] = value
    return nothing
end

# -------------------------
# Configuración (ajústalo)
# -------------------------
datasets_config = Dict(
    # --- "otro paper" (ETT + WTH/LCD) ---
    "ETT/ETTm1.csv" => Dict("target_col" => "OT", "windows" => [12, 24, 48], "train_ratio" => 0.75),
    "ETT/ETTh2.csv" => Dict("target_col" => "OT", "windows" => [12, 24, 48], "train_ratio" => 0.75),
    "LCDS/LCD_USW00094789_2024.csv" => Dict("target_col" => "HourlyDryBulbTemperature", "windows" => [12, 24, 48], "train_ratio" => 0.75),

    # --- UCI (Energy + AirQuality + PRSA/PM2.5) ---
    # Energy (10 min): 12h/24h/48h -> 72/144/288
    "UCI/Energy/energydata_complete.csv" => Dict("target_col" => "Appliances", "windows" => [72, 144, 288], "train_ratio" => 0.80),
    # AirQuality (horario)
    "UCI/AirQuality/AirQualityUCI.csv" => Dict("target_col" => "CO(GT)", "windows" => [12, 24, 48], "train_ratio" => 0.80),
    # PRSA (horario) — si quieres replicar el split del paper, usa 0.30
    "UCI/PRSA/PRSA_data_2010.1.1-2014.12.31.csv" => Dict("target_col" => "pm2.5", "windows" => [12, 24, 48], "train_ratio" => 0.30)

    # Si quieres reactivar ElectricDevices/LD (regresión), añade aquí la ruta correcta:
    # "ElectricDevices/LD2011_2014_mini.txt" => Dict("target_col" => "MT_196", "windows" => [12, 24, 48], "train_ratio" => 0.75)
)

data_dir = "data"
normalization = "MaxMin"
model = :DoME
seed = 1  # DoME determinista

# Stream hyperparams
horizon = 1
default_train_ratio = 0.75
memory_sizes = [200]        # prueba [50, 100, 200] si quieres
batch_sizes  = [1]          # 1 = online, 10/50 = micro-batch
steps_list   = [1, 5, 10]   # lo que te dijo Dani: 1, 5, ...

# Configs DoME a barrer (para empezar, NO lo abras demasiado)
# - Recomendación práctica: fija useDivisionOperator=true y 1-2 estrategias.
configs = modelConfigurations(model)
config_ids = collect(1:length(configs))  # o por ejemplo [1,2,3]

SAVE_DETAILED = true

# -------------------------
# Selección por CLI
# -------------------------
selected = String[]
if length(ARGS) >= 1 && ARGS[1] != "--all"
    push!(selected, ARGS[1])
else
    selected = collect(keys(datasets_config))
end

for d in selected
    haskey(datasets_config, d) || error("Dataset no está en datasets_config: $d")
end

# -------------------------
# Salida
# -------------------------
results_dir = "results_stream"
isdir(results_dir) || mkdir(results_dir)

jobtag = get(ENV, "SLURM_JOB_ID", string(getpid()))
timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")

# -------------------------
# Sweep por dataset
# -------------------------
for dataset in selected
    conf = datasets_config[dataset]
    target_col = conf["target_col"]
    windows = conf["windows"]
    train_ratio = get(conf, "train_ratio", default_train_ratio)

    dataset_name = replace(split(dataset, "/")[end], ".csv" => "", ".txt" => "")
    outpath = joinpath(results_dir, "$(dataset_name)__$(timestamp)__$(jobtag).jld2")

    metadata = Dict(
        "dataset" => dataset,
        "dataset_name" => dataset_name,
        "target_col" => target_col,
        "windows" => windows,
        "seed" => seed,
        "normalization" => normalization,
        "model" => string(model),
        "horizon" => horizon,
        "train_ratio" => train_ratio,
        "default_train_ratio" => default_train_ratio,
        "memory_sizes" => memory_sizes,
        "batch_sizes" => batch_sizes,
        "steps_list" => steps_list,
        "config_ids" => config_ids,
        "created_at" => string(Dates.now()),
        "jobtag" => jobtag
    )

    total_exps = length(windows) * length(memory_sizes) * length(batch_sizes) * length(steps_list) * length(config_ids)

    println("\n" * "="^70)
    println("SWEEP STREAM DoME")
    println("Dataset: $dataset")
    println("Output:  $outpath")
    println("Total experiments: $total_exps")
    println("="^70)

    start_time = time()
    counter = 0

    jldopen(outpath, "w") do f
        jld_overwrite!(f, "metadata", metadata)
        jld_overwrite!(f, "progress/started_at", string(Dates.now()))
    end

    jldopen(outpath, "a") do f
        for window in windows
            for memory_size in memory_sizes
                for batch_size in batch_sizes
                    for steps in steps_list
                        for config_id in config_ids
                            counter += 1

                            key = "results/w$(window)/m$(memory_size)/b$(batch_size)/k$(steps)/c$(config_id)"
                            haskey(f, key) && continue

                            try
                                res_df, det = run_stream_experiment(
                                    dataset=dataset,
                                    window=window,
                                    horizon=horizon,
                                    target_col=target_col,
                                    config_id=config_id,
                                    seed=seed,
                                    data_dir=data_dir,
                                    normalization=normalization,
                                    train_ratio=train_ratio,
                                    memory_size=memory_size,
                                    batch_size=batch_size,
                                    steps_per_update=steps,
                                    verbose=false
                                )

                                entry = Dict(
                                    "summary" => Dict(pairs(eachcol(res_df)) .|> x -> (String(x[1]) => x[2][1]) )
                                )

                                if SAVE_DETAILED
                                    entry["detailed"] = det
                                end

                                f[key] = entry

                            catch e
                                f[key] = Dict(
                                    "error" => string(e),
                                    "saved_at" => string(Dates.now())
                                )
                            end

                            if counter % 10 == 0
                                elapsed = time() - start_time
                                println("Progreso: $counter/$total_exps | $(round(elapsed/60, digits=2)) min")
                                jld_overwrite!(f, "progress/last_updated", string(Dates.now()))
                                jld_overwrite!(f, "progress/counter", counter)
                            end
                        end
                    end
                end
            end
        end

        jld_overwrite!(f, "progress/finished_at", string(Dates.now()))
        jld_overwrite!(f, "progress/counter", counter)
    end

    elapsed = time() - start_time
    println("="^70)
    println("SWEEP STREAM COMPLETADO | $(round(elapsed/60, digits=2)) min")
    println("Output: $outpath")
    println("="^70)
end