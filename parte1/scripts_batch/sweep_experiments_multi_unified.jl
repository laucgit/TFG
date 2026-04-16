include(joinpath(@__DIR__, "run_experiment_multi_unified.jl"))

using Dates
using JLD2
using CSV
using DataFrames
using Statistics

nowstr() = string(Dates.now())

function jld_overwrite!(f, key::AbstractString, value)
    if haskey(f, key)
        delete!(f, key)
    end
    f[key] = value
    return nothing
end

const JLD_WRITE_IOTYPE = JLD2.MmapIO

function jld_with(path::AbstractString, mode::AbstractString, fn::Function)
    jldopen(path, mode; iotype=JLD_WRITE_IOTYPE) do f
        fn(f)
    end
end
jld_with(fn::Function, path::AbstractString, mode::AbstractString) = jld_with(path, mode, fn)

function dataset_basename(ds::AbstractString)
    base = split(String(ds), "/")[end]
    base = replace(base, r"\.csv$" => "")
    base = replace(base, r"\.txt$" => "")
    return base
end

function resolve_dataset_key(arg::AbstractString, datasets_config)::Union{String,Nothing}
    a = String(arg)
    haskey(datasets_config, a) && return a

    base = basename(a)
    haskey(datasets_config, base) && return base

    noext = replace(base, r"\.csv$" => "")
    noext = replace(noext, r"\.txt$" => "")
    for c in (base * ".csv", base * ".txt", noext, noext * ".csv", noext * ".txt")
        haskey(datasets_config, c) && return c
    end
    return nothing
end

function parse_windows_env(val::AbstractString)
    s = strip(String(val))
    isempty(s) && return Int[]
    if occursin(",", s)
        return [parse(Int, strip(x)) for x in split(s, ",") if !isempty(strip(x))]
    end
    if occursin(":", s)
        parts = split(s, ":")
        if length(parts) == 3
            a = parse(Int, parts[1]); step = parse(Int, parts[2]); b = parse(Int, parts[3])
            return collect(a:step:b)
        elseif length(parts) == 2
            a = parse(Int, parts[1]); b = parse(Int, parts[2])
            return collect(a:b)
        end
    end
    return [parse(Int, s)]
end

function parse_bool_env(name::String, default::Bool=false)
    get(ENV, name, default ? "1" : "0") in ("1", "true", "TRUE", "yes", "YES")
end

function parse_optional_int_env(name::String)
    s = strip(get(ENV, name, ""))
    isempty(s) && return nothing
    v = parse(Int, s)
    return v > 0 ? v : nothing
end

function parse_optional_float_env(name::String)
    s = strip(get(ENV, name, ""))
    isempty(s) && return nothing
    v = parse(Float64, s)
    return v > 0 ? v : nothing
end

function acquire_lock!(outpath::AbstractString)
    lock_path = String(outpath) * ".lock"
    if isfile(lock_path)
        error("LOCK activo: $lock_path\nHay otro job escribiendo este output o quedó un lock antiguo.")
    end
    open(lock_path, "w") do io
        println(io, "pid=$(getpid()) jobid=" * get(ENV, "SLURM_JOB_ID", "") * " time=$(nowstr())")
    end
    atexit(() -> (isfile(lock_path) && rm(lock_path; force=true)))
    return lock_path
end

windows_default = collect(10:5:45)
windows_friedman = [168]

datasets_config = Dict{String,Dict{String,Any}}(
    "hour.csv" => Dict{String,Any}(
        "dataset_path" => "hour.csv", "target_col" => "cnt", "windows" => windows_default, "train_ratio" => 0.60,
        "normalization" => "standard", "reference_metric" => "mae", "task_type" => :regression,
        "categorical_encoding" => "error", "input_booleanization" => "none",
    ),
    "friedman_drift_gra_seed42_n1000000.csv" => Dict{String,Any}(
        "dataset_path" => "friedman_drift_gra_seed42_n1000000.csv", "target_col" => "target", "windows" => windows_friedman, "train_ratio" => 0.001,
        "normalization" => "standard", "reference_metric" => "mae", "task_type" => :regression,
        "categorical_encoding" => "error", "input_booleanization" => "none",
    ),
    "friedman_drift_gra_seed42_n500000_thin.csv" => Dict{String,Any}(
        "dataset_path" => "friedman_drift_gra_seed42_n500000_thin.csv", "target_col" => "target", "windows" => windows_friedman, "train_ratio" => 0.80,
        "normalization" => "standard", "reference_metric" => "mae", "task_type" => :regression,
        "categorical_encoding" => "error", "input_booleanization" => "none",
    ),
    "friedman_drift_gra_seed42_n100000_thin.csv" => Dict{String,Any}(
        "dataset_path" => "friedman_drift_gra_seed42_n100000_thin.csv", "target_col" => "target", "windows" => windows_friedman, "train_ratio" => 0.01,
        "normalization" => "standard", "reference_metric" => "mae", "task_type" => :regression,
        "categorical_encoding" => "error", "input_booleanization" => "none",
    ),
    "UNSW_NB15_raw_binary" => Dict{String,Any}(
        "dataset_path" => "UNSW", "target_col" => "label", "windows" => windows_default, "train_ratio" => 0.75,
        "normalization" => "none", "reference_metric" => "accuracy", "task_type" => :binary_classification,
        "categorical_encoding" => "onehot", "input_booleanization" => "median_threshold",
    ),
    "UNSW_NB15_raw_multiclass" => Dict{String,Any}(
        "dataset_path" => "UNSW", "target_col" => "attack_cat", "windows" => windows_default, "train_ratio" => 0.75,
        "normalization" => "none", "reference_metric" => "accuracy", "task_type" => :multiclass_classification,
        "categorical_encoding" => "onehot", "input_booleanization" => "median_threshold",
    ),
)

default_conf() = Dict{String,Any}(
    "dataset_path" => nothing, "target_col" => nothing, "windows" => windows_default, "train_ratio" => 0.75,
    "normalization" => "none", "reference_metric" => "mae", "task_type" => :regression,
    "categorical_encoding" => "error", "input_booleanization" => "none",
)

model = :DoME
seeds = 1:1
SAVE_DETAILED = parse_bool_env("SAVE_DETAILED", true)
SAVE_PREDICTIONS = parse_bool_env("SAVE_PREDICTIONS", false)
data_dir = get(ENV, "DATA_DIR", default_data_dir())
train_max_steps = parse_optional_int_env("TRAIN_MAX_STEPS")
train_max_time_sec = parse_optional_float_env("TRAIN_MAX_TIME_SEC")
train_log_every = something(parse_optional_int_env("TRAIN_LOG_EVERY"), 0)
train_verbose = parse_bool_env("TRAIN_VERBOSE", false)

raw_args = copy(ARGS)
length(raw_args) == 0 && error("Uso: julia sweep_experiments_multi_unified.jl <DATASET> | --all")
selected_datasets = raw_args[1] == "--all" ? collect(keys(datasets_config)) : String.(raw_args)

resolved = Dict{String,Tuple{String,Dict{String,Any}}}()
for a in selected_datasets
    k = resolve_dataset_key(a, datasets_config)
    resolved[a] = (k === nothing) ? (a, default_conf()) : (k, datasets_config[k])
end

results_dir = joinpath(@__DIR__, "results")
isdir(results_dir) || mkdir(results_dir)
jobtag = get(ENV, "SLURM_JOB_ID", string(getpid()))
timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
stable_out = get(ENV, "STABLE_OUT", "") in ("1", "true", "TRUE", "yes", "YES")
fixed_max_nodes = let s = strip(get(ENV, "MAX_NODES_FIXED", "")); isempty(s) ? nothing : parse(Int, s) end
model_configs = modelConfigurations(model)
configs_serializable = [Dict("config_id" => i, "max_nodes" => cfg["maxNumNodes"], "min_improvement" => cfg["minimumReductionMSE"], "use_division" => cfg["useDivisionOperator"], "strategy" => cfg["strategyName"]) for (i, cfg) in enumerate(model_configs)]

for ds_arg in selected_datasets
    dataset_key, dataset_conf = resolved[ds_arg]
    dataset = isnothing(dataset_conf["dataset_path"]) ? dataset_key : String(dataset_conf["dataset_path"])
    windows = dataset_conf["windows"]
    target_col = dataset_conf["target_col"]
    train_ratio = dataset_conf["train_ratio"]
    normalization = dataset_conf["normalization"]
    reference_metric = dataset_conf["reference_metric"]
    task_type = dataset_conf["task_type"]
    categorical_encoding = dataset_conf["categorical_encoding"]
    input_booleanization = dataset_conf["input_booleanization"]

    if haskey(ENV, "WINDOWS")
        wv = parse_windows_env(ENV["WINDOWS"])
        if !isempty(wv)
            windows = wv
        end
    end
    if haskey(ENV, "TRAIN_RATIO"); train_ratio = parse(Float64, ENV["TRAIN_RATIO"]); end
    if haskey(ENV, "NORMALIZATION"); normalization = ENV["NORMALIZATION"]; end
    if haskey(ENV, "REFERENCE_METRIC"); reference_metric = ENV["REFERENCE_METRIC"]; end
    horizon = parse(Int, get(ENV, "HORIZON", "1"))

    c_start = parse(Int, get(ENV, "C_START", "1"))
    c_end = parse(Int, get(ENV, "C_END", string(length(model_configs))))
    c_start = max(1, min(c_start, length(model_configs)))
    c_end = max(1, min(c_end, length(model_configs)))
    if c_start > c_end
        c_start, c_end = c_end, c_start
    end
    config_range = c_start:c_end
    config_ids = [cid for cid in config_range if isnothing(fixed_max_nodes) || model_configs[cid]["maxNumNodes"] == fixed_max_nodes]
    isempty(config_ids) && error("No hay configuraciones para MAX_NODES_FIXED=$(fixed_max_nodes) en el rango $(first(config_range)):$(last(config_range))")
    config_ids = [cid for cid in config_range if isnothing(fixed_max_nodes) || model_configs[cid]["maxNumNodes"] == fixed_max_nodes]
    isempty(config_ids) && error("No hay configuraciones para MAX_NODES_FIXED=$(fixed_max_nodes) en el rango $(first(config_range)):$(last(config_range))")
    config_ids = [cid for cid in config_range if isnothing(fixed_max_nodes) || model_configs[cid]["maxNumNodes"] == fixed_max_nodes]
    isempty(config_ids) && error("No hay configuraciones para MAX_NODES_FIXED=$(fixed_max_nodes) en el rango $(first(config_range)):$(last(config_range))")

    dname = dataset_basename(dataset_key)
    wtag = (length(windows) == 1) ? "w$(only(windows))" : "wmulti"
    outpath = joinpath(results_dir, "$(dname)_$(wtag)__$(timestamp)__$(jobtag).jld2")
    if stable_out && !haskey(ENV, "OUT_JLD2")
        outpath = joinpath(results_dir, "$(dname)_$(wtag).jld2")
    end
    if haskey(ENV, "OUT_JLD2") && length(selected_datasets) == 1
        outpath = ENV["OUT_JLD2"]
    end
    acquire_lock!(outpath)

    metadata = Dict(
        "dataset_arg" => ds_arg,
        "dataset" => dataset,
        "dataset_name" => dname,
        "target_col" => target_col,
        "windows" => windows,
        "horizon" => horizon,
        "train_ratio" => train_ratio,
        "normalization" => normalization,
        "reference_metric" => reference_metric,
        "task_type" => String(task_type),
        "categorical_encoding" => categorical_encoding,
        "input_booleanization" => input_booleanization,
        "seeds" => collect(seeds),
        "data_dir" => data_dir,
        "model" => string(model),
        "configs" => configs_serializable,
        "train_verbose" => train_verbose,
        "train_max_steps" => train_max_steps,
        "train_max_time_sec" => train_max_time_sec,
        "train_log_every" => train_log_every,
        "created_at" => nowstr(),
        "jobtag" => jobtag,
        "max_nodes_fixed" => fixed_max_nodes,
        "selected_config_ids" => collect(config_ids),
    )

    total_experiments = length(windows) * length(config_ids) * length(seeds)
    start_time = time()
    file_exists = isfile(outpath)
    counter = 0
    if file_exists
        jld_with(outpath, "r") do f
            if haskey(f, "progress/counter")
                counter = Int(f["progress/counter"])
            end
        end
    end

    mode = file_exists ? "a" : "w"
    jld_with(outpath, mode) do f
        if !haskey(f, "metadata"); jld_overwrite!(f, "metadata", metadata); end
        if !haskey(f, "progress/started_at"); jld_overwrite!(f, "progress/started_at", nowstr()); end
        if !haskey(f, "progress/counter"); jld_overwrite!(f, "progress/counter", counter); end
        jld_overwrite!(f, "progress/total_experiments", total_experiments)
        jld_overwrite!(f, "progress/last_updated", nowstr())
    end

    for window in windows
        for config_id in config_ids
            key = "results/w$(window)/c$(config_id)"
            already_done = false
            jld_with(outpath, "a") do f
                already_done = haskey(f, key)
            end
            already_done && continue

            per_seed = Vector{Any}(undef, length(seeds))
            reference_scores = Float64[]
            n_success = 0
            cfg = model_configs[config_id]

            for (si, seed) in enumerate(seeds)
                counter += 1
                try
                    res_df, det, preds = run_experiment(
                        dataset=dataset,
                        window=window,
                        horizon=horizon,
                        normalization=normalization,
                        reference_metric=reference_metric,
                        model=model,
                        config_id=config_id,
                        seed=seed,
                        data_dir=data_dir,
                        target_col=target_col,
                        train_ratio=train_ratio,
                        task_type=task_type,
                        categorical_encoding=categorical_encoding,
                        input_booleanization=input_booleanization,
                        save_predictions=SAVE_PREDICTIONS,
                        save_detailed=SAVE_DETAILED,
                        verbose=false,
                        train_verbose=train_verbose,
                        train_max_steps=train_max_steps,
                        train_max_time_sec=train_max_time_sec,
                        train_log_every=train_log_every,
                    )
                    ref_score = Float64(res_df.reference_score[1])
                    push!(reference_scores, ref_score)
                    n_success += 1
                    per_seed[si] = Dict(
                        "seed" => seed,
                        "reference_metric" => String(res_df.reference_metric[1]),
                        "reference_score" => ref_score,
                        "accuracy" => res_df.accuracy[1],
                        "sensitivity" => res_df.sensitivity[1],
                        "precision" => res_df.precision[1],
                        "f1" => res_df.f1[1],
                        "fpr" => res_df.fpr[1],
                        "roc_auc" => res_df.roc_auc[1],
                        "kappa" => res_df.kappa[1],
                        "mse_test_raw" => res_df.mse_test_raw[1],
                        "rmse_test_raw" => res_df.rmse_test_raw[1],
                        "mae_test_raw" => res_df.mae_test_raw[1],
                        "r2_test_raw" => res_df.r2_test_raw[1],
                        "nrmse_test_raw" => res_df.nrmse_test_raw[1],
                        "iterations" => res_df.iterations[1],
                        "stop_reason" => String(res_df.stop_reason[1]),
                        "train_time_sec" => Float64(res_df.train_time_sec[1]),
                        "total_time_sec" => Float64(res_df.total_time_sec[1]),
                        "eval_mode" => ("eval_mode" in names(res_df) ? String(res_df.eval_mode[1]) : "holdout"),
                        "batches" => ("batches" in names(res_df) ? Int(res_df.batches[1]) : missing),
                        "include_remainder" => ("include_remainder" in names(res_df) ? Bool(res_df.include_remainder[1]) : missing),
                        "remainder_rows" => ("remainder_rows" in names(res_df) ? Int(res_df.remainder_rows[1]) : missing),
                        "memory_mb_peak" => ("memory_mb_peak" in names(res_df) ? res_df.memory_mb_peak[1] : missing),
                        "details" => (SAVE_DETAILED ? det : nothing),
                        "predictions" => (SAVE_PREDICTIONS ? preds : nothing),
                    )
                catch e
                    per_seed[si] = Dict("seed" => seed, "error" => string(e), "timestamp" => nowstr())
                end
            end

            entry = Dict(
                "dataset" => dataset,
                "window" => window,
                "config_id" => config_id,
                "max_nodes" => cfg["maxNumNodes"],
                "min_improvement" => cfg["minimumReductionMSE"],
                "use_division" => cfg["useDivisionOperator"],
                "strategy" => cfg["strategyName"],
                "reference_metric" => reference_metric,
                "n_success" => n_success,
                "n_seeds" => length(seeds),
                "reference_score_mean" => (n_success > 0 ? mean(reference_scores) : Inf),
                "reference_score_min" => (n_success > 0 ? minimum(reference_scores) : Inf),
                "per_seed" => per_seed,
                "saved_at" => nowstr(),
            )
            jld_with(outpath, "a") do f
                f[key] = entry
            end
        end
    end
end
