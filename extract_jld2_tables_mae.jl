using JLD2
using DataFrames
using CSV
using Statistics

const ONLY_COMPLETE_FILES = true

indir = length(ARGS) >= 1 ? ARGS[1] : "parte1/resultados_Bike"
outdir = length(ARGS) >= 2 ? ARGS[2] : "plots/Bike_tables_mae"
mkpath(outdir)

function cidnum(k)
    m = match(r"c(\d+)$", String(k))
    return m === nothing ? typemax(Int) : parse(Int, m.captures[1])
end

function short_strategy(s::AbstractString)
    if s == "StrategySelective"
        return "Selective"
    elseif s == "StrategySelectiveWithConstantOptimization"
        return "Selective+ConstOpt"
    else
        return String(s)
    end
end

hascol(df, s::Symbol) = s in names(df)

function getfirstcol(row, df, syms::Vector{Symbol}; default=missing)
    for s in syms
        if hascol(df, s)
            return row[s]
        end
    end
    return default
end

function to_float_or_missing(x)
    ismissing(x) && return missing
    return Float64(x)
end

function to_int_or_missing(x)
    ismissing(x) && return missing
    return Int(x)
end

rows = NamedTuple[]

for file in sort(filter(f -> endswith(f, ".jld2"), readdir(indir; join=true)))
    jldopen(file, "r") do f
        # Caso 1: formato con /results/.../cN
        if haskey(f, "results")
            expected_total = haskey(f, "progress/total_experiments") ? Int(f["progress/total_experiments"]) : missing
            tmprows = NamedTuple[]
            done_total = 0

            windows = sort(String.(collect(keys(f["results"]))))
            for w in windows
                group = f["results/" * w]
                cfgs = sort(String.(collect(keys(group))), by=cidnum)
                done_total += length(cfgs)

                for c in cfgs
                    x = f["results/$w/$c"]
                    s = x["per_seed"][1]

                    base = (
                        file = basename(file),
                        full_path = file,
                        window = w,
                        config_id = Int(x["config_id"]),
                        max_nodes = Int(x["max_nodes"]),
                        min_improvement = Float64(x["min_improvement"]),
                        strategy = String(x["strategy"]),
                        strategy_short = short_strategy(String(x["strategy"])),
                        n_success = Int(x["n_success"]),
                    )

                    if haskey(s, "error")
                        push!(tmprows, merge(base, (
                            mae = missing,
                            rmse = missing,
                            mse = missing,
                            r2 = missing,
                            nrmse = missing,
                            batches = missing,
                            train_time_sec = missing,
                            total_time_sec = missing,
                            error = String(s["error"]),
                        )))
                    else
                        push!(tmprows, merge(base, (
                            mae = Float64(s["mae_test_raw"]),
                            rmse = Float64(s["rmse_test_raw"]),
                            mse = Float64(s["mse_test_raw"]),
                            r2 = Float64(s["r2_test_raw"]),
                            nrmse = Float64(s["nrmse_test_raw"]),
                            batches = Int(s["batches"]),
                            train_time_sec = Float64(s["train_time_sec"]),
                            total_time_sec = Float64(s["total_time_sec"]),
                            error = "",
                        )))
                    end
                end
            end

            file_complete = ismissing(expected_total) ? missing : (done_total >= expected_total)

            for r in tmprows
                push!(rows, merge(r, (
                    expected_total = expected_total,
                    done_total = done_total,
                    file_complete = file_complete,
                )))
            end

        # Caso 2: formato con results_df en raíz
        elseif haskey(f, "results_df")
            dfsrc = f["results_df"]

            for row in eachrow(dfsrc)
                mae_val   = getfirstcol(row, dfsrc, [:mae_test_raw, :mae])
                rmse_val  = getfirstcol(row, dfsrc, [:rmse_test_raw, :rmse])
                mse_val   = getfirstcol(row, dfsrc, [:mse_test_raw, :mse])
                r2_val    = getfirstcol(row, dfsrc, [:r2_test_raw, :r2])
                nrmse_val = getfirstcol(row, dfsrc, [:nrmse_test_raw, :nrmse])

                strategy_val = getfirstcol(row, dfsrc, [:strategy]; default="unknown")
                strategy_val = ismissing(strategy_val) ? "unknown" : String(strategy_val)

                err_val = getfirstcol(row, dfsrc, [:error]; default="")
                err_val = ismissing(err_val) ? "" : String(err_val)

                push!(rows, (
                    file = basename(file),
                    full_path = file,
                    window = string(getfirstcol(row, dfsrc, [:window]; default="wNA")),
                    config_id = begin
                        v = getfirstcol(row, dfsrc, [:config_id]; default=-1)
                        ismissing(v) ? -1 : Int(v)
                    end,
                    max_nodes = begin
                        v = getfirstcol(row, dfsrc, [:max_nodes]; default=-1)
                        ismissing(v) ? -1 : Int(v)
                    end,
                    min_improvement = begin
                        v = getfirstcol(row, dfsrc, [:min_improvement]; default=NaN)
                        ismissing(v) ? NaN : Float64(v)
                    end,
                    strategy = strategy_val,
                    strategy_short = short_strategy(strategy_val),
                    n_success = begin
                        v = getfirstcol(row, dfsrc, [:n_success]; default=1)
                        ismissing(v) ? 1 : Int(v)
                    end,
                    mae = to_float_or_missing(mae_val),
                    rmse = to_float_or_missing(rmse_val),
                    mse = to_float_or_missing(mse_val),
                    r2 = to_float_or_missing(r2_val),
                    nrmse = to_float_or_missing(nrmse_val),
                    batches = to_int_or_missing(getfirstcol(row, dfsrc, [:batches])),
                    train_time_sec = to_float_or_missing(getfirstcol(row, dfsrc, [:train_time_sec])),
                    total_time_sec = to_float_or_missing(getfirstcol(row, dfsrc, [:total_time_sec])),
                    error = err_val,
                    expected_total = nrow(dfsrc),
                    done_total = nrow(dfsrc),
                    file_complete = true,
                ))
            end

        else
            @warn "Formato no reconocido, se ignora" file collect(keys(f))
        end
    end
end

df = DataFrame(rows)
CSV.write(joinpath(outdir, "results_all.csv"), df)

if nrow(df) == 0
    error("No se ha encontrado ningún resultado en '$indir'.")
end

dfok = subset(df,
    :error => ByRow(x -> x == ""),
    :mae => ByRow(x -> !ismissing(x))
)

if nrow(dfok) == 0
    error("No hay resultados válidos con MAE.")
end

dfplot = if ONLY_COMPLETE_FILES && (:file_complete in names(dfok))
    tmp = subset(dfok, :file_complete => ByRow(x -> x === true))
    nrow(tmp) > 0 ? tmp : dfok
else
    dfok
end

CSV.write(joinpath(outdir, "results_used_for_plots.csv"), dfplot)

best_by_nodes = combine(groupby(sort(dfplot, [:max_nodes, :mae]), :max_nodes)) do sdf
    row = sdf[1, :]
    (
        max_nodes = Int(row.max_nodes),
        best_mae = Float64(row.mae),
        best_mse = Float64(row.mse),
        best_r2 = Float64(row.r2),
        strategy = String(row.strategy),
        strategy_short = String(row.strategy_short),
        min_improvement = Float64(row.min_improvement),
        file = String(row.file),
    )
end

sort!(best_by_nodes, :max_nodes)
CSV.write(joinpath(outdir, "best_by_nodes.csv"), best_by_nodes)

baseline_nodes = minimum(best_by_nodes.max_nodes)
baseline_mae = best_by_nodes[best_by_nodes.max_nodes .== baseline_nodes, :best_mae][1]
best_by_nodes.mae_reduction_pct = 100 .* (baseline_mae .- best_by_nodes.best_mae) ./ baseline_mae
CSV.write(joinpath(outdir, "best_by_nodes_with_reduction.csv"), best_by_nodes)

best_by_strategy = combine(groupby(sort(dfplot, [:strategy_short, :mae]), :strategy_short)) do sdf
    row = sdf[1, :]
    (
        strategy = String(row.strategy),
        strategy_short = String(row.strategy_short),
        best_mae = Float64(row.mae),
        best_mse = Float64(row.mse),
        best_r2 = Float64(row.r2),
        max_nodes = Int(row.max_nodes),
        min_improvement = Float64(row.min_improvement),
        file = String(row.file),
    )
end

sort!(best_by_strategy, :best_mae)
CSV.write(joinpath(outdir, "best_by_strategy.csv"), best_by_strategy)

println("Tablas guardadas en: $outdir")
println("Filas totales extraídas: ", nrow(df))
println("Filas válidas con MAE: ", nrow(dfok))
println("Filas usadas para gráficas: ", nrow(dfplot))
println("Baseline MAE: max_nodes=$baseline_nodes, mae=$baseline_mae")