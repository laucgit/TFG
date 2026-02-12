#!/usr/bin/env julia
# sweep_experiments_multi.jl
#
# Uso:
#   julia --project=. sweep_experiments_multi.jl "ETT/ETTh2.csv"
#
# Qué hace:
# - Recorre windows = [12,24,48]
# - Recorre todas las configuraciones de modelConfigurations(:DoME)
# - Reanuda automáticamente: si existe results/.../c<id>_s<seed>.jld2 -> skip
# - Guarda progreso en results/<dataset>/progress.csv (append)
#
# Robustez HPC:
# - NO guarda objetos complejos (cfg completo, detailed_dict, funciones, tipos, etc.) para evitar errores de JLD2.
# - Guarda solo valores simples + métricas.
# - Escribe primero en $TMPDIR y luego mv al destino para minimizar problemas de Lustre/JLD2.
# - Si falla el guardado, deja un “marcador” .jld2 para no reintentar esa config infinitamente.

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

function safe_save_jld2(outfile::AbstractString; kwargs...)
    tmpdir  = get(ENV, "TMPDIR", "/tmp")
    tmpfile = joinpath(tmpdir, basename(outfile) * ".tmp")

    # Limpieza preventiva
    if isfile(tmpfile)
        rm(tmpfile; force=true)
    end

    # Guardado correcto usando jldsave (acepta kwargs)
    JLD2.jldsave(tmpfile; kwargs...)

    # Mover a destino final
    mv(tmpfile, outfile; force=true)

    return nothing
end


function safe_marker_jld2(outfile::AbstractString; kwargs...)
    try
        safe_save_jld2(outfile; kwargs...)
    catch
        # Último recurso: intentar escribir directamente
        try
            JLD2.jldsave(outfile; kwargs...)
        catch
            # Si incluso esto falla, no hacemos nada más
        end
    end
end


# -----------------------------
# MAIN
# -----------------------------
function main(args::Vector{String})
    # Dataset por argumento o por ENV["DATASET"]
    dataset_path = length(args) >= 1 ? args[1] : get(ENV, "DATASET", "")
    dataset_path == "" && error("Uso: julia --project=. sweep_experiments_multi.jl \"ruta/al/dataset.csv\" (o define ENV[\"DATASET\"])")

    jobid = get(ENV, "SLURM_JOB_ID", "local")

    dataset_name = replace(split(dataset_path, "/")[end], ".csv" => "", ".txt" => "")
    base_dir = ensure_dir(joinpath("results", dataset_name))
    progress_csv = joinpath(base_dir, "progress.csv")

    println("DATASET=", dataset_path)
    println("DATASET_NAME=", dataset_name)
    println("JOBID=", jobid)
    println("RESULTS_DIR=", base_dir)
    println("PROGRESS_CSV=", progress_csv)

    # Sweep params
    windows = [12, 24, 48]
    seed = 1

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

    # Contadores
    total_done = 0
    total_skipped = 0
    total_errors = 0

    # Directorio de datos (en tu caso ya tienes symlink data -> ../data)
    data_dir = "data"

    for window in windows
        wdir = ensure_dir(joinpath(base_dir, "w$(window)"))

        for config_id in 1:nconfigs
            cfg = configs[config_id]

            # Solo valores simples (seguros para JLD2)
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

            # Defaults métricas
            status = "ok"
            err = ""
            mse_test_raw   = Inf
            train_time_sec = Inf
            total_time_sec = Inf

            try
                Random.seed!(seed)

                results_df, _detailed_dict, _pred_df = run_experiment(
                    dataset=dataset_path,
                    window=window,
                    normalization="MaxMin",
                    model=model,
                    config_id=config_id,
                    seed=seed,
                    data_dir=data_dir,
                    save_predictions=false,
                    save_detailed=true,   # puede ser true internamente; NO lo guardamos aquí
                    verbose=true
                )

                mse_test_raw   = float(results_df.mse_test_raw[1])
                train_time_sec = float(results_df.train_time_sec[1])
                total_time_sec = float(results_df.total_time_sec[1])

            catch e
                status = "error"
                err = sprint(showerror, e, catch_backtrace())
                total_errors += 1
                total_time_sec = time() - t0
            end

            # Guardado robusto (nunca guardamos cfg completo ni dicts complejos)
            save_status = status
            save_err = err

            try
                safe_save_jld2(outfile;
                    dataset_path=dataset_path,
                    dataset_name=dataset_name,
                    window=window,
                    config_id=config_id,
                    seed=seed,
                    minimumReductionMSE=minred,
                    maxNumNodes=maxnodes,
                    useDivisionOperator=use_div,
                    strategyName=strat,
                    status=save_status,
                    mse_test_raw=mse_test_raw,
                    train_time_sec=train_time_sec,
                    total_time_sec=total_time_sec,
                    error=save_err
                )
            catch e_save
                # Si el guardado peta, deja marcador mínimo para que no reintente eternamente
                save_status = "save_error"
                save_err = sprint(showerror, e_save, catch_backtrace())
                safe_marker_jld2(outfile;
                    dataset_path=dataset_path,
                    dataset_name=dataset_name,
                    window=window,
                    config_id=config_id,
                    seed=seed,
                    status=save_status,
                    error=save_err
                )
            end

            total_done += 1

            # Progreso CSV: reflejamos el estado final de guardado
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
                csv_quote(save_status),
                string(mse_test_raw),
                string(train_time_sec),
                string(total_time_sec),
                csv_quote(replace(save_err, '\n' => ' ')),
                csv_quote(outfile)
            ])

            @printf("[%s] w=%d cfg=%d/%d maxn=%d minred=%.0e div=%s strat=%s -> MSE_test_raw=%.6g  status=%s  (%.2fs)\n",
                nowstamp(), window, config_id, nconfigs, maxnodes, minred, string(use_div), strat,
                mse_test_raw, save_status, time() - t0
            )
            flush(stdout)
        end
    end

    println("FIN. Done=", total_done, "  Skipped=", total_skipped, "  Errors=", total_errors, "  Results dir: ", base_dir)
    return 0
end

main(ARGS)
