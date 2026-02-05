using SymDoME
using Statistics
using DataFrames
using Dates
using Random
using JSON
using CSV

include("load_data_multi.jl")

struct DoMEParams
    max_nodes::Int
    min_improvement::Float64
    use_division::Bool
    strategy::Function
end

function modelConfigurations(model::Symbol)
    model != :DoME && error("Modelo no soportado: $model")

    MinimumReductionsMSE = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
    MaxNumNodes = [5, 10, 20, 30, 50, 75, 100, 150, 200]
    Strategies = [SymDoME.Strategy4, SymDoME.Strategy3]

    configurations = [
        Dict{String,Any}(
            "minimumReductionMSE" => minimumReductionMSE,
            "maxNumNodes" => maxNumNodes,
            "useDivisionOperator" => useDivisionOperator,
            "strategy" => strategy,
            "strategyName" => string(strategy)
        )
        for maxNumNodes in MaxNumNodes,
            minimumReductionMSE in MinimumReductionsMSE,
            useDivisionOperator in [false, true],
            strategy in Strategies
    ][:]

    return configurations
end

# -----------------------------
# Imputación SIN leakage
# (estadísticos solo de train)
# -----------------------------
function colmeans_finite(X::Matrix{Float64})
    n, d = size(X)
    μ = Vector{Float64}(undef, d)
    for j in 1:d
        s = 0.0; c = 0
        @inbounds for i in 1:n
            v = X[i, j]
            if isfinite(v)
                s += v; c += 1
            end
        end
        μ[j] = (c > 0) ? (s / c) : 0.0
    end
    return μ
end

function mean_finite(y::Vector{Float64})
    s = 0.0; c = 0
    @inbounds for v in y
        if isfinite(v)
            s += v; c += 1
        end
    end
    return (c > 0) ? (s / c) : 0.0
end

function impute_nonfinite_with_means!(X::Matrix{Float64}, μ::Vector{Float64})
    n, d = size(X)
    @inbounds for j in 1:d
        mj = μ[j]
        for i in 1:n
            if !isfinite(X[i, j])
                X[i, j] = mj
            end
        end
    end
    return X
end

function impute_nonfinite_with_mean!(y::Vector{Float64}, μ::Float64)
    @inbounds for i in eachindex(y)
        if !isfinite(y[i])
            y[i] = μ
        end
    end
    return y
end

# -----------------------------
# Train / Eval
# -----------------------------
function train_dome(
    Xtr::Matrix{Float64},
    ytr::Vector{Float64},
    params::DoMEParams;
    max_iterations::Int=1000,
    verbose::Bool=true
)
    verbose && println("[TRAIN] max_nodes=$(params.max_nodes) min_improvement=$(params.min_improvement) div=$(params.use_division)")

    dome_obj = SymDoME.DoME(
        Xtr, ytr;
        maximumNodes=params.max_nodes,
        minimumReductionMSE=params.min_improvement,
        useDivisionOperator=params.use_division,
        strategy=params.strategy
    )

    history = Float64[dome_obj.mse]
    stop_reason = "no_improvement"

    for it in 1:max_iterations
        improved = SymDoME.Step!(dome_obj)
        push!(history, dome_obj.mse)

        if !improved
            stop_reason = "no_improvement"
            break
        end

        if it == max_iterations
            stop_reason = "max_iterations"
        end
    end

    return dome_obj.tree, history, stop_reason
end

"""
EVALUACIÓN BLINDADA:
- siempre evalúa con tree (nunca con String)
- expression solo si need_expression=true
"""
function evaluate_dome(tree, Xte::Matrix{Float64}, yte::Vector{Float64}; need_expression::Bool=false)
    ŷ = collect(SymDoME.evaluateTree(tree, Xte))  # ✅ tree
    mse = mean((ŷ .- yte).^2)
    expr = need_expression ? SymDoME.vectorString(tree) : nothing
    return ŷ, mse, expr
end

# -----------------------------
# Run (sin I/O)
# -----------------------------
function run_experiment(;
    dataset::String,
    window::Int,
    normalization::String="MaxMin",
    model::Symbol=:DoME,
    config_id::Int,
    seed::Int=1,
    data_dir::String="data",
    horizon::Int=1,
    target_col::Union{String,Int,Nothing}=nothing,
    train_ratio::Float64=0.75,
    save_detailed::Bool=true,
    save_predictions::Bool=false,              # <- compatible con tu sweep
    predictions_dir::Union{Nothing,String}=nothing,  # <- se ignora aquí (I/O lo hace el sweep)
    max_iterations::Int=1000,
    verbose::Bool=true
)
    Random.seed!(seed)

    configs = modelConfigurations(model)
    (1 <= config_id <= length(configs)) || error("config_id fuera de rango: $config_id (total: $(length(configs)))")

    cfg = configs[config_id]
    params = DoMEParams(cfg["maxNumNodes"], cfg["minimumReductionMSE"], cfg["useDivisionOperator"], cfg["strategy"])
    strategy_name = cfg["strategyName"]

    verbose && begin
        println("\n" * "="^70)
        println("RUN EXPERIMENT")
        println("="^70)
        println("dataset=$dataset window=$window norm=$normalization model=$model config_id=$config_id seed=$seed")
        println("max_nodes=$(params.max_nodes) min_impr=$(params.min_improvement) div=$(params.use_division) strat=$strategy_name")
    end

    Xtr, ytr, Xte, yte = load_dataset(
        dataset;
        data_dir=data_dir,
        window=window,
        horizon=horizon,
        train_ratio=train_ratio,
        target_col=target_col,
        verbose=verbose
    )

    # Imputación SIN leakage
    μX = colmeans_finite(Xtr)
    μy = mean_finite(ytr)
    impute_nonfinite_with_means!(Xtr, μX)
    impute_nonfinite_with_means!(Xte, μX)
    impute_nonfinite_with_mean!(ytr, μy)
    impute_nonfinite_with_mean!(yte, μy)

    normalization == "MaxMin" || error("Normalización no soportada: $normalization (solo MaxMin)")

    # Normalización (stats de train)
    X_min = minimum(Xtr, dims=1)
    X_max = maximum(Xtr, dims=1)
    X_rng = X_max .- X_min
    X_rng[X_rng .== 0.0] .= 1.0

    Xtr_norm = (Xtr .- X_min) ./ X_rng
    Xte_norm = (Xte .- X_min) ./ X_rng

    y_min = minimum(ytr)
    y_max = maximum(ytr)
    y_rng = y_max - y_min
    y_rng = (y_rng == 0.0) ? 1.0 : y_rng

    ytr_norm = (ytr .- y_min) ./ y_rng
    yte_norm = (yte .- y_min) ./ y_rng

    # Train
    tree, history, stop_reason = train_dome(Xtr_norm, ytr_norm, params; max_iterations=max_iterations, verbose=verbose)

    iterations_real = length(history) - 1
    hit_max_iterations = (stop_reason == "max_iterations")
    converged = iterations_real > 0

    # Eval norm + expresión opcional
    ŷ_test_norm, mse_test_norm, expr = evaluate_dome(tree, Xte_norm, yte_norm; need_expression=save_detailed)

    # Eval raw
    ŷ_raw = ŷ_test_norm .* y_rng .+ y_min
    mse_test_raw = mean((ŷ_raw .- yte).^2)

    mse_initial = history[1]
    mse_final_train = history[end]
    improvement_pct = (mse_initial == 0.0) ? 0.0 : (1 - mse_final_train / mse_initial) * 100

    dataset_name = replace(split(dataset, "/")[end], ".csv" => "", ".txt" => "")
    ts = Dates.now()

    results_df = DataFrame(
        dataset=[dataset],
        dataset_name=[dataset_name],
        window=[window],
        normalization=[normalization],
        model=[string(model)],
        config_id=[config_id],
        seed=[seed],

        max_nodes=[params.max_nodes],
        min_improvement=[params.min_improvement],
        use_division=[params.use_division],
        strategy=[strategy_name],

        mse_initial=[mse_initial],
        mse_final_train=[mse_final_train],
        mse_test_norm=[mse_test_norm],
        mse_test_raw=[mse_test_raw],
        improvement_pct=[improvement_pct],

        iterations=[iterations_real],
        converged=[converged],
        hit_max_iterations=[hit_max_iterations],
        stop_reason=[stop_reason],

        train_samples=[size(Xtr, 1)],
        test_samples=[size(Xte, 1)],
        features=[size(Xtr, 2)],

        timestamp=[ts],
        error=[""]
    )

    detailed_dict = nothing
    if save_detailed
        detailed_dict = Dict(
            "dataset" => dataset,
            "dataset_name" => dataset_name,
            "window" => window,
            "normalization" => normalization,
            "model" => string(model),
            "config_id" => config_id,
            "seed" => seed,

            "max_nodes" => params.max_nodes,
            "min_improvement" => params.min_improvement,
            "use_division" => params.use_division,
            "strategy" => strategy_name,

            "expression" => (expr === nothing ? "" : expr),
            "history" => history,

            "mse_initial" => mse_initial,
            "mse_final_train" => mse_final_train,
            "mse_test_norm" => mse_test_norm,
            "mse_test_raw" => mse_test_raw,
            "improvement_pct" => improvement_pct,

            "iterations" => iterations_real,
            "converged" => converged,
            "hit_max_iterations" => hit_max_iterations,
            "stop_reason" => stop_reason,

            "timestamp" => string(ts)
        )
    end

    predictions_df = nothing
    if save_predictions
        predictions_df = DataFrame(
            y_true_norm = yte_norm,
            y_pred_norm = ŷ_test_norm,
            y_true_raw  = yte,
            y_pred_raw  = ŷ_raw,
            error_norm  = ŷ_test_norm .- yte_norm,
            error_raw   = ŷ_raw .- yte
        )
    end

    return results_df, detailed_dict, predictions_df
end

# CLI (debug local)
if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 6
        println("Uso: julia run_experiment_multi.jl <dataset> <window> <norm> <model> <config_id> <seed> [save_predictions] [save_detailed]")
        exit(1)
    end

    dataset = ARGS[1]
    window = parse(Int, ARGS[2])
    normalization = ARGS[3]
    model = Symbol(ARGS[4])
    config_id = parse(Int, ARGS[5])
    seed = parse(Int, ARGS[6])
    save_predictions = length(ARGS) >= 7 ? parse(Bool, ARGS[7]) : false
    save_detailed = length(ARGS) >= 8 ? parse(Bool, ARGS[8]) : true

    run_experiment(
        dataset=dataset,
        window=window,
        normalization=normalization,
        model=model,
        config_id=config_id,
        seed=seed,
        save_predictions=save_predictions,
        save_detailed=save_detailed,
        verbose=true
    )
end
