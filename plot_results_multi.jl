# Headless para HPC: ponerlo ANTES de cargar Plots/GR
ENV["GKSwstype"] = "100"

using DataFrames
using CSV
using Statistics
using Plots
using JLD2

gr()

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
prettify_strategy(s::AbstractString) = begin
    if occursin("SelectiveWithConstantOptimization", s) || occursin("WithConstantOptimization", s)
        "Selective+const"
        elseif occursin("Selective", s)
        "Selective"
        elseif occursin("Strategy4", s)
        "Selective"
        elseif occursin("Strategy3", s)
        "Selective+const"
    else
        s
    end
end

function safe_first(df::DataFrame, col::Symbol, default)
    try
        if nrow(df) >= 1 && (col in names(df))
            v = df[1, col]
            return (v === missing || v === nothing) ? default : v
        end
    catch
    end
    return default
end

function tofloat(x, default=NaN)
    if x === missing || x === nothing
        return default
    end
    try
        return Float64(x)
    catch
        return default
    end
end

function toint(x, default=missing)
    if x === missing || x === nothing
        return default
    end
    try
        return Int(x)
    catch
        return default
    end
end

# -----------------------------------------------------------------------------
# Cargar 1 .jld2 (robusto a faltas / errores)
# -----------------------------------------------------------------------------
function read_one_jld2(file::String)
    # defaults (evito missing en floats para que nunca crashee con isnan)
    status = "ok"
    dataset = ""
    window = missing
    config_id = missing
    seed = missing
    max_nodes = missing
    min_improvement = NaN
    use_division = missing
    strategy = ""
    strategy_label = ""
    mse_test_raw = Inf
    mse_test_norm = NaN
    train_time_sec = NaN
    test_time_sec = NaN
    total_time_sec = NaN
    timestamp = ""

    # Para plots de drift (opcional)
    predictions_df = nothing

    try
        d = JLD2.load(file)  # Dict{String,Any}

        # status (si existe)
        status = haskey(d, "status") ? string(d["status"]) : "ok"

        # dataset/window/seed si existen sueltos
        dataset = haskey(d, "dataset") ? string(d["dataset"]) : ""
        window  = haskey(d, "window") ? toint(d["window"], missing) : missing
        seed    = haskey(d, "seed")   ? toint(d["seed"], missing)   : missing

        # cfg (hyperparams)
        if haskey(d, "cfg") && d["cfg"] !== nothing
            cfg = d["cfg"]
            max_nodes      = haskey(cfg, "maxNumNodes") ? toint(cfg["maxNumNodes"], missing) : max_nodes
            min_improvement = haskey(cfg, "minimumReductionMSE") ? tofloat(cfg["minimumReductionMSE"], NaN) : min_improvement
            use_division   = haskey(cfg, "useDivisionOperator") ? Bool(cfg["useDivisionOperator"]) : use_division
            strategy       = haskey(cfg, "strategyName") ? string(cfg["strategyName"]) : strategy
        end

        # métricas sueltas (si existen)
        if haskey(d, "mse_test_raw");   mse_test_raw   = tofloat(d["mse_test_raw"], Inf) end
        if haskey(d, "total_time_sec"); total_time_sec = tofloat(d["total_time_sec"], NaN) end
        if haskey(d, "fit_time_sec");   train_time_sec = tofloat(d["fit_time_sec"], NaN) end
        if haskey(d, "test_time_sec");  test_time_sec  = tofloat(d["test_time_sec"], NaN) end
        if haskey(d, "config_id");      config_id      = toint(d["config_id"], missing) end

        # results_df (fallback)
        if haskey(d, "results_df") && d["results_df"] !== nothing
            rdf = d["results_df"]::DataFrame

            if dataset == ""; dataset = string(safe_first(rdf, :dataset, "")) end
            if window === missing; window = toint(safe_first(rdf, :window, missing), missing) end
            if config_id === missing; config_id = toint(safe_first(rdf, :config_id, missing), missing) end
            if seed === missing; seed = toint(safe_first(rdf, :seed, missing), missing) end

            if max_nodes === missing
                max_nodes = toint(safe_first(rdf, :max_nodes, missing), missing)
            end

            if isnan(min_improvement)
                min_improvement = tofloat(safe_first(rdf, :min_improvement, NaN), NaN)
            end

            if use_division === missing
                v = safe_first(rdf, :use_division, missing)
                use_division = (v === missing) ? use_division : Bool(v)
            end

            if strategy == ""
                strategy = string(safe_first(rdf, :strategy, ""))
            end

            if !isfinite(mse_test_raw) || mse_test_raw == Inf
                mse_test_raw = tofloat(safe_first(rdf, :mse_test_raw, Inf), Inf)
            end

            mse_test_norm  = tofloat(safe_first(rdf, :mse_test_norm, NaN), NaN)
            train_time_sec = tofloat(safe_first(rdf, :train_time_sec, train_time_sec), train_time_sec)
            total_time_sec = tofloat(safe_first(rdf, :total_time_sec, total_time_sec), total_time_sec)
            timestamp      = string(safe_first(rdf, :timestamp, ""))
        end

        # predictions_df (si existe)
        if haskey(d, "predictions_df") && d["predictions_df"] !== nothing
            predictions_df = d["predictions_df"]::DataFrame
        end

        strategy_label = prettify_strategy(strategy)

        # Si status != ok, lo marcamos como no elegible (MSE=Inf)
        if lowercase(status) != "ok"
            mse_test_raw = Inf
        end

    catch
        status = "read_error"
        mse_test_raw = Inf
        strategy_label = ""
    end

    return (
        file=file,
        status=status,
        dataset=dataset,
        window=window,
        config_id=config_id,
        seed=seed,
        max_nodes=max_nodes,
        min_improvement=min_improvement,
        use_division=use_division,
        strategy=strategy,
        strategy_label=strategy_label,
        mse_test_raw=mse_test_raw,
        mse_test_norm=mse_test_norm,
        train_time_sec=train_time_sec,
        test_time_sec=test_time_sec,
        total_time_sec=total_time_sec,
        timestamp=timestamp,
        predictions_df=predictions_df
        )
end

# -----------------------------------------------------------------------------
# Recoger todos los .jld2 de un directorio (recursivo)
# -----------------------------------------------------------------------------
function collect_results(results_dir::String)
    isdir(results_dir) || error("Directorio no encontrado: $results_dir")

    files = String[]
    for (root, _, fs) in walkdir(results_dir)
        for f in fs
            endswith(f, ".jld2") || continue
            push!(files, joinpath(root, f))
        end
    end
    sort!(files)

    rows = DataFrame(
        file=String[],
        status=String[],
        dataset=String[],
        window=Int[],
        config_id=Int[],
        seed=Int[],
        max_nodes=Int[],
        min_improvement=Float64[],
        use_division=Bool[],
        strategy=String[],
        strategy_label=String[],
        mse_test_raw=Float64[],
        mse_test_norm=Float64[],
        train_time_sec=Float64[],
        test_time_sec=Float64[],
        total_time_sec=Float64[],
        timestamp=String[]
        )

    for f in files
        r = read_one_jld2(f)

        # descartamos si window/max_nodes faltan
        if r.window === missing || r.max_nodes === missing
            continue
        end

        push!(rows, (
            r.file,
            r.status,
            r.dataset,
            Int(r.window),
            r.config_id === missing ? -1 : Int(r.config_id),
            r.seed === missing ? -1 : Int(r.seed),
            Int(r.max_nodes),
            Float64(r.min_improvement),
            r.use_division === missing ? false : Bool(r.use_division),
            r.strategy,
            r.strategy_label,
            Float64(r.mse_test_raw),
            Float64(r.mse_test_norm),
            Float64(r.train_time_sec),
            Float64(r.test_time_sec),
            Float64(r.total_time_sec),
            r.timestamp
            ))
    end

    return rows
end

# -----------------------------------------------------------------------------
# TopK por window
# -----------------------------------------------------------------------------
function topk_by_window(df::DataFrame; k::Int=10)
    out = DataFrame()
    for w in sort(unique(df.window))
        sub = df[df.window .== w, :]
        sub = sort(sub, [:mse_test_raw, :total_time_sec])
        subk = first(sub, min(k, nrow(sub)))
        out = vcat(out, subk)
    end
    return out
end

# -----------------------------------------------------------------------------
# Mejor por (window, strategy, max_nodes) minimizando MSE
# -----------------------------------------------------------------------------
function best_by_nodes(df::DataFrame)
    df_ok = df[isfinite.(df.mse_test_raw), :]
    g = groupby(df_ok, [:window, :strategy_label, :max_nodes])
    best = combine(g) do sub
        idx = argmin(sub.mse_test_raw)
        sub[idx, [:window, :strategy_label, :max_nodes, :min_improvement, :config_id, :seed,
                  :mse_test_raw, :total_time_sec, :train_time_sec, :test_time_sec, :file]]
    end
    sort!(best, [:window, :strategy_label, :max_nodes])
    return best
end

# -----------------------------------------------------------------------------
# Plots agregados
# -----------------------------------------------------------------------------
function plot_mse_vs_nodes(best::DataFrame, out_dir::String)
    for w in sort(unique(best.window))
        subw = best[best.window .== w, :]
        nrow(subw) == 0 && continue

        p = plot(
            xlabel="MaxNumNodes",
            ylabel="Mejor MSE (test, raw)",
            title="Mejor MSE vs nodos (window=$w)",
            legend=:topright
            )

        for strat in sort(unique(subw.strategy_label))
            subs = subw[subw.strategy_label .== strat, :]
            sort!(subs, :max_nodes)
            plot!(p, subs.max_nodes, subs.mse_test_raw, label=strat)
        end

        savefig(p, joinpath(out_dir, "mse_vs_nodes_w$(w).pdf"))
    end
end

function plot_time_vs_nodes(best::DataFrame, out_dir::String)
    for w in sort(unique(best.window))
        subw = best[best.window .== w, :]
        nrow(subw) == 0 && continue

        p = plot(
            xlabel="MaxNumNodes",
            ylabel="Tiempo total (s)",
            title="Tiempo vs nodos (window=$w) [mejor config por nodo]",
            legend=:topleft
            )

        for strat in sort(unique(subw.strategy_label))
            subs = subw[subw.strategy_label .== strat, :]
            sort!(subs, :max_nodes)
            plot!(p, subs.max_nodes, subs.total_time_sec, label=strat)
        end

        savefig(p, joinpath(out_dir, "time_vs_nodes_w$(w).pdf"))
    end
end

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
function run_dir(results_dir::String; topk::Int=10)
    df = collect_results(results_dir)
    out_dir = joinpath(results_dir, "plots")
    mkpath(out_dir)

    CSV.write(joinpath(out_dir, "summary_all_runs.csv"), df)

    tk = topk_by_window(df; k=topk)
    CSV.write(joinpath(out_dir, "topK_by_window.csv"), tk)

    best = best_by_nodes(df)
    CSV.write(joinpath(out_dir, "best_by_nodes.csv"), best)

    plot_mse_vs_nodes(best, out_dir)
    plot_time_vs_nodes(best, out_dir)

    println("✓ Listo. Salida en: $out_dir")
end

if abspath(PROGRAM_FILE) == @__FILE__
    isempty(ARGS) && error("Uso: julia --project=. plot_results_multi.jl <results_dir> [topk]")

    path = ARGS[1]
    isdir(path) || error("Ruta no válida (no es directorio): $path")

    topk = (length(ARGS) >= 2) ? parse(Int, ARGS[2]) : 10
    run_dir(path; topk=topk)
end
