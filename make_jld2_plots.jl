using JLD2
using DataFrames
using CSV
using Statistics

indir  = length(ARGS) >= 1 ? ARGS[1] : "parte1/resultados_Bike"
outdir = length(ARGS) >= 2 ? ARGS[2] : "plots/Bike_tables"

mkpath(outdir)

function cidnum(k)
    m = match(r"c(\d+)$", String(k))
    return m === nothing ? typemax(Int) : parse(Int, m.captures[1])
end

rows = NamedTuple[]

for file in sort(filter(f -> endswith(f, ".jld2"), readdir(indir; join=true)))
    jldopen(file, "r") do f
        haskey(f, "results") || return

        for w in sort(String.(collect(keys(f["results"]))))
            group = f["results/" * w]
            cfgs = sort(String.(collect(keys(group))), by=cidnum)

            for c in cfgs
                x = f["results/$w/$c"]
                s = x["per_seed"][1]

                if haskey(s, "error")
                    push!(rows, (
                        file = basename(file),
                        window = w,
                        config_id = Int(x["config_id"]),
                        max_nodes = Int(x["max_nodes"]),
                        min_improvement = Float64(x["min_improvement"]),
                        strategy = String(x["strategy"]),
                        mae = missing,
                        rmse = missing,
                        mse = missing,
                        r2 = missing,
                        nrmse = missing,
                        batches = missing,
                        total_time_sec = missing,
                        error = String(s["error"])
                    ))
                else
                    push!(rows, (
                        file = basename(file),
                        window = w,
                        config_id = Int(x["config_id"]),
                        max_nodes = Int(x["max_nodes"]),
                        min_improvement = Float64(x["min_improvement"]),
                        strategy = String(x["strategy"]),
                        mae = Float64(s["mae_test_raw"]),
                        rmse = Float64(s["rmse_test_raw"]),
                        mse = Float64(s["mse_test_raw"]),
                        r2 = Float64(s["r2_test_raw"]),
                        nrmse = Float64(s["nrmse_test_raw"]),
                        batches = Int(s["batches"]),
                        total_time_sec = Float64(s["total_time_sec"]),
                        error = ""
                    ))
                end
            end
        end
    end
end

df = DataFrame(rows)
CSV.write(joinpath(outdir, "results_all.csv"), df)

dfok = filter(row -> !ismissing(row.mse), df)

best_by_nodes = combine(groupby(dfok, :max_nodes),
    :mse => minimum => :best_mse,
    :mae => minimum => :best_mae,
    :r2 => maximum => :best_r2
)
sort!(best_by_nodes, :max_nodes)
CSV.write(joinpath(outdir, "best_by_nodes.csv"), best_by_nodes)

baseline_nodes = minimum(best_by_nodes.max_nodes)
baseline_mse = best_by_nodes[best_by_nodes.max_nodes .== baseline_nodes, :best_mse][1]
best_by_nodes.mse_reduction_pct =
    100 .* (baseline_mse .- best_by_nodes.best_mse) ./ baseline_mse
CSV.write(joinpath(outdir, "best_by_nodes_with_reduction.csv"), best_by_nodes)

best_by_strategy = combine(groupby(dfok, :strategy),
    :mse => minimum => :best_mse,
    :mae => minimum => :best_mae,
    :r2 => maximum => :best_r2
)
sort!(best_by_strategy, :best_mse)
CSV.write(joinpath(outdir, "best_by_strategy.csv"), best_by_strategy)

println("Tablas guardadas en: ", outdir)