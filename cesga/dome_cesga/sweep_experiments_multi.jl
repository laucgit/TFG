#!/usr/bin/env julia
# sweep_experiments_multi.jl
#
# Uso:
#   julia --project=. sweep_experiments_multi.jl "ruta/al/dataset.csv"
#
# Barrido por:
#   - window
#   - config_id (definido en modelConfigurations(:DoME))
#
# Guarda un .jld2 por (window, config_id, seed) y reanuda:
# si existe el fichero, hace SKIP.
#
# Mantiene un CSV de progreso estable por dataset (append).

using Dates
using Printf
using Random
using JLD2

include("run_experiment_multi.jl")

# -----------------------------
# Helpers
# -----------------------------
function ensure_dir(path::AbstractString)
    isdir(path) || mkpath(path)
    return path
end

nowstamp() = Dates.format(now(), "yyyymmdd_HHMMSS")

# CSV robusto (quotea strings)
function csv_quote(x)
    s = x === nothing ? "" : string(x)
    s = replace(s, '\"' => "\"\"")
    return "\"" * s * "\""
end

function write_progress_csv!(csvpath::AbstractString, header::Vector{String}, row::Vector{String})
    if !isfile(csvpath)
        open(csvpath, "w") do io
            println(io, join(header, ","))
        end
    end
    open(csvpath, "a") do io
        println(io, join(row, ","))
    end
end

# -----------------------------
# Entradas
# -----------------------------
if length(ARGS) < 1
    error("Uso: julia --project=. sweep_experiments_multi.jl \"ruta/al/dataset.csv\"")
end
dataset_path = ARGS[1]

jobid = get(ENV, "SLURM_JOB_ID", "local")

dataset_name = replace(split(dataset_path, "/")[end], ".csv" => "", ".txt" => "")
base_dir = ensure_dir(joinpath("results", dataset_name))
progress_csv = joinpath(base_dir, "progress.csv")

println("DATASET=", dataset_path)
println("DATASET_NAME=", dataset_name)
println("JOBID=", jobid)
println("RESULTS_DIR=", base_dir)
println("PROGRESS_CSV=", progress_csv)

# -----------------------------
# Sweep params (según conversación)
# -----------------------------
windows = [12, 24, 48]
seed = 1  # determinista, pero explícito

# Configuraciones: fuente de verdad
model = :DoME
configs = modelConfigurations(model)
nconfigs = length(configs)

println("TOTAL_CONFIGS(modelConfigurations)=", nconfigs)

# Cabecera CSV
header = [
    "timestamp","jobid","dataset","dataset_name","window","config_id",
    "minimumReductionMSE","maxNumNodes","useDivisionOperator","strategyName",
    "status","mse_test_raw","train_time_sec","total_time_sec","error","outfile"
]

# -----------------------------
# Ejecución
# -----------------------------
total_done = 0
total_skipped = 0
total_errors = 0

for window in windows
    wdir = ensure_dir(joinpath(base_dir, "w$(window)"))

    for config_id in 1:nconfigs
        cfg = configs[config_id]

        minred   = cfg["minimumReductionMSE"]
        maxnodes = cfg["maxNumNodes"]
        use_div  = cfg["useDivisionOperator"]
        strat    = cfg["strategyName"]

        outfile = joinpath(wdir, "c$(config_id)_s$(seed).jld2")

        # Reanudar
        if isfile(outfile)
            total_skipped += 1
            write_progress_csv!(progress_csv, header, [
                csv_quote(nowstamp()),
                csv_quote(jobid),
                csv_quote(dataset_path),
                csv_quote(dataset_name),
                string(window),
                string(config_id),
                string(minred),
                string(maxnodes),
                string(use_div),
                csv_quote(strat),
                csv_quote("skip"),
                "Inf",
                "0.0",
                "0.0",
                csv_quote(""),
                csv_quote(outfile)
            ])
            continue
        end

        t0 = time()
        status = "ok"
        err = ""

        mse_test_raw   = Inf
        train_time_sec = Inf
        total_time_sec = Inf

        try
            Random.seed!(seed)

            results_df, detailed_dict, _pred_df = run_experiment(
                dataset=dataset_path,
                window=window,
                normalization="MaxMin",
                model=model,
                config_id=config_id,
                seed=seed,
                save_predictions=false,
                save_detailed=true,
                verbose=true
            )

            # Una fila
            mse_test_raw   = float(results_df.mse_test_raw[1])
            train_time_sec = float(results_df.train_time_sec[1])
            total_time_sec = float(results_df.total_time_sec[1])

            # Guardado completo
            @save outfile dataset_path dataset_name window config_id seed cfg results_df detailed_dict

        catch e
            status = "error"
            err = sprint(showerror, e, catch_backtrace())
            total_errors += 1
            total_time_sec = time() - t0

            # Guardado mínimo para debug
            @save outfile dataset_path dataset_name window config_id seed cfg status err
        end

        total_done += 1

        write_progress_csv!(progress_csv, header, [
            csv_quote(nowstamp()),
            csv_quote(jobid),
            csv_quote(dataset_path),
            csv_quote(dataset_name),
            string(window),
            string(config_id),
            string(minred),
            string(maxnodes),
            string(use_div),
            csv_quote(strat),
            csv_quote(status),
            string(mse_test_raw),
            string(train_time_sec),
            string(total_time_sec),
            csv_quote(replace(err, '\n' => ' ')),
            csv_quote(outfile)
        ])

        @printf("[%s] w=%d cfg=%d/%d maxn=%d minred=%.0e div=%s strat=%s -> MSE_test_raw=%.6g  status=%s  (%.2fs)\n",
            nowstamp(), window, config_id, nconfigs, maxnodes, minred, string(use_div), strat, mse_test_raw, status, time()-t0)

        flush(stdout)
    end
end

println("FIN. Done=", total_done, "  Skipped=", total_skipped, "  Errors=", total_errors, "  Results dir: ", base_dir)
