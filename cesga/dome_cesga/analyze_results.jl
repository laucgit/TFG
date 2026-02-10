#!/usr/bin/env julia
# analyze_results.jl
#
# Uso:
#   julia --project=. analyze_results.jl results [topK]
#
# Lee results/ recursivo, recoge SOLO ejecuciones OK con mse finito,
# y genera un CSV con TopK por mse_test_raw (y desempata por tiempo).

using JLD2
using Printf

results_dir = length(ARGS) >= 1 ? ARGS[1] : "results"
topK = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 20

files = String[]
for (root, _, fs) in walkdir(results_dir)
    for f in fs
        endswith(f, ".jld2") && push!(files, joinpath(root, f))
    end
end

if isempty(files)
    println("No hay .jld2 todavía en ", results_dir)
    exit(0)
end

rows = Vector{NamedTuple}()

for f in files
    try
        d = JLD2.load(f)

        window    = get(d, "window", missing)
        config_id = get(d, "config_id", missing)
        seed      = get(d, "seed", missing)

        # Si el sweep guardó status explícito en caso de error
        status = get(d, "status", "ok")
        if status != "ok"
            continue
        end

        mse_test = Inf
        train_time = Inf
        total_time = Inf

        if haskey(d, "results_df")
            rdf = d["results_df"]
            # En tu pipeline nuevo: mse_test_raw / tiempos *_sec
            if hasproperty(rdf, :mse_test_raw)
                mse_test = float(rdf.mse_test_raw[1])
            elseif hasproperty(rdf, :mse_test)
                mse_test = float(rdf.mse_test[1])
            end

            train_time = hasproperty(rdf, :train_time_sec) ? float(rdf.train_time_sec[1]) : Inf
            total_time = hasproperty(rdf, :total_time_sec) ? float(rdf.total_time_sec[1]) : Inf
        end

        # Hiperparámetros (si están)
        minred = missing
        maxnodes = missing
        use_div = missing
        strat = missing

        if haskey(d, "cfg")
            cfg = d["cfg"]
            minred   = get(cfg, "minimumReductionMSE", missing)
            maxnodes = get(cfg, "maxNumNodes", missing)
            use_div  = get(cfg, "useDivisionOperator", missing)
            strat    = get(cfg, "strategyName", missing)
        end

        if isfinite(mse_test)
            push!(rows, (; file=f, window, config_id, seed, mse_test, train_time, total_time, strat, maxnodes, minred, use_div))
        end

    catch e
        @warn "No pude leer $f" exception=(e, catch_backtrace())
    end
end

if isempty(rows)
    println("No hay resultados OK con mse finito todavía en ", results_dir)
    exit(0)
end

sort!(rows, by = r -> (r.mse_test, r.total_time))

best = rows[1:min(topK, length(rows))]

outcsv = joinpath(results_dir, "best_top$(topK).csv")
open(outcsv, "w") do io
    println(io, "rank,window,config_id,seed,mse_test_raw,train_time_sec,total_time_sec,strategyName,maxNumNodes,minimumReductionMSE,useDivisionOperator,file")
    for (i,r) in enumerate(best)
        @printf(io, "%d,%s,%s,%s,%.10g,%.6g,%.6g,%s,%s,%s,%s,%s\n",
            i,
            string(r.window),
            string(r.config_id),
            string(r.seed),
            r.mse_test,
            r.train_time,
            r.total_time,
            string(r.strat),
            string(r.maxnodes),
            string(r.minred),
            string(r.use_div),
            r.file
        )
    end
end

println("Hecho. Top ", length(best), " guardado en: ", outcsv)
