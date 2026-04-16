using SymDoME
using Statistics
using DataFrames
using Dates
using Random
using CSV

include(joinpath(@__DIR__, "load_data_multi_unified.jl"))

struct DoMEParams
    max_nodes::Int
    min_improvement::Float64
    use_division::Bool
    strategy::Function
end

function default_data_dir()
    candidates = [joinpath(@__DIR__, "data"), @__DIR__]
    for c in candidates
        isdir(c) && return c
    end
    return @__DIR__
end

function modelConfigurations(model::Symbol)
    model != :DoME && error("Modelo no soportado: $model")

    MinimumReductionsMSE = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    MaxNumNodes = collect(5:5:200)
    Strategies = [SymDoME.Strategy4, SymDoME.Strategy3]

    configurations = [
        Dict{String,Any}(
            "minimumReductionMSE" => minimumReductionMSE,
            "maxNumNodes" => maxNumNodes,
            "useDivisionOperator" => useDivisionOperator,
            "strategy" => strategy,
            "strategyName" => string(strategy),
        )
        for maxNumNodes in MaxNumNodes,
            minimumReductionMSE in MinimumReductionsMSE,
            useDivisionOperator in [true],
            strategy in Strategies
    ][:]

    return configurations
end

function _safe_std(v)
    s = std(v)
    return (isfinite(s) && s > 0.0) ? s : 1.0
end

function apply_normalization(Xtr, Xte, ytr, yte, normalization::String, task_type::Symbol)
    norm = lowercase(strip(normalization))
    if norm in ("none", "no", "false")
        return Xtr, Xte, ytr, yte, Dict("method" => "none", "y_scaled" => false)
    elseif norm in ("standard", "zscore", "standardscaler")
        μx = vec(mean(Xtr, dims=1))
        σx = [_safe_std(view(Xtr, :, j)) for j in 1:size(Xtr, 2)]
        @inbounds for j in 1:size(Xtr, 2)
            Xtr[:, j] .= (Xtr[:, j] .- μx[j]) ./ σx[j]
            Xte[:, j] .= (Xte[:, j] .- μx[j]) ./ σx[j]
        end
        if task_type == :regression
            μy = mean(ytr)
            σy = _safe_std(ytr)
            return Xtr, Xte, (ytr .- μy) ./ σy, (yte .- μy) ./ σy, Dict("method" => "standard", "μy" => μy, "σy" => σy, "y_scaled" => true)
        else
            return Xtr, Xte, ytr, yte, Dict("method" => "standard", "y_scaled" => false)
        end
    elseif norm in ("minmax", "maxmin")
        X_min = minimum(Xtr, dims=1)
        X_max = maximum(Xtr, dims=1)
        X_rng = X_max .- X_min
        X_rng[X_rng .== 0.0] .= 1.0
        Xtr .= (Xtr .- X_min) ./ X_rng
        Xte .= (Xte .- X_min) ./ X_rng
        if task_type == :regression
            y_min = minimum(ytr)
            y_max = maximum(ytr)
            y_rng = y_max - y_min
            y_rng = (y_rng == 0.0) ? 1.0 : y_rng
            return Xtr, Xte, (ytr .- y_min) ./ y_rng, (yte .- y_min) ./ y_rng, Dict("method" => "minmax", "y_min" => y_min, "y_rng" => y_rng, "y_scaled" => true)
        else
            return Xtr, Xte, ytr, yte, Dict("method" => "minmax", "y_scaled" => false)
        end
    else
        error("Normalización no soportada: $normalization")
    end
end

function denormalize_predictions(ŷ_norm, stats::Dict{String,Any})
    y = vec(Float64.(collect(ŷ_norm)))
    method = String(get(stats, "method", "none"))
    if method == "none" || !get(stats, "y_scaled", false)
        return copy(y)
    elseif method == "standard"
        return y .* Float64(stats["σy"]) .+ Float64(stats["μy"])
    elseif method == "minmax"
        return y .* Float64(stats["y_rng"]) .+ Float64(stats["y_min"])
    else
        error("Método de desnormalización no soportado: $method")
    end
end

function regression_metrics(y_true::Vector{Float64}, y_pred::Vector{Float64})
    err = y_pred .- y_true
    mse = mean(err .^ 2)
    rmse = sqrt(mse)
    mae = mean(abs.(err))
    y_mean = mean(y_true)
    ss_res = sum(err .^ 2)
    ss_tot = sum((y_true .- y_mean) .^ 2)
    r2 = (ss_tot == 0.0) ? 1.0 : (1.0 - ss_res / ss_tot)
    y_range = maximum(y_true) - minimum(y_true)
    nrmse = (y_range == 0.0) ? 0.0 : rmse / y_range
    return Dict("mse" => mse, "rmse" => rmse, "mae" => mae, "r2" => r2, "nrmse" => nrmse)
end

function _binary_auc(y_true::Vector{Float64}, scores::Vector{Float64})
    pos_idx = findall(==(1.0), y_true)
    neg_idx = findall(==(0.0), y_true)
    isempty(pos_idx) && return missing
    isempty(neg_idx) && return missing

    pairs = collect(zip(scores, y_true))
    sort!(pairs, by=x -> x[1])
    ranks = zeros(Float64, length(pairs))
    i = 1
    while i <= length(pairs)
        j = i
        while j < length(pairs) && pairs[j + 1][1] == pairs[i][1]
            j += 1
        end
        avg_rank = (i + j) / 2
        for k in i:j
            ranks[k] = avg_rank
        end
        i = j + 1
    end

    sum_ranks_pos = 0.0
    npos = 0
    for (r, p) in zip(ranks, pairs)
        if p[2] == 1.0
            sum_ranks_pos += r
            npos += 1
        end
    end
    nneg = length(pairs) - npos
    (npos == 0 || nneg == 0) && return missing
    return (sum_ranks_pos - npos * (npos + 1) / 2) / (npos * nneg)
end

function _kappa_from_confusion(conf::Matrix{Float64})
    n = sum(conf)
    n == 0 && return missing
    po = sum(diag(conf)) / n
    rowm = sum(conf, dims=2)
    colm = sum(conf, dims=1)
    pe = sum(rowm .* colm) / (n^2)
    denom = 1 - pe
    denom == 0 && return missing
    return (po - pe) / denom
end

function binary_classification_metrics(y_true::Vector{Float64}, scores::Vector{Float64}; threshold::Float64=0.5)
    y_pred = map(s -> s >= threshold ? 1.0 : 0.0, scores)
    tp = sum((y_true .== 1.0) .& (y_pred .== 1.0))
    tn = sum((y_true .== 0.0) .& (y_pred .== 0.0))
    fp = sum((y_true .== 0.0) .& (y_pred .== 1.0))
    fn = sum((y_true .== 1.0) .& (y_pred .== 0.0))
    n = length(y_true)
    accuracy = n == 0 ? missing : (tp + tn) / n
    recall = (tp + fn) == 0 ? missing : tp / (tp + fn)
    precision = (tp + fp) == 0 ? missing : tp / (tp + fp)
    f1 = (ismissing(precision) || ismissing(recall) || (precision + recall == 0)) ? missing : 2 * precision * recall / (precision + recall)
    fpr = (fp + tn) == 0 ? missing : fp / (fp + tn)
    auc = _binary_auc(y_true, scores)
    kappa = _kappa_from_confusion([tn fp; fn tp])
    return Dict(
        "accuracy" => accuracy,
        "sensitivity" => recall,
        "precision" => precision,
        "f1" => f1,
        "fpr" => fpr,
        "roc_auc" => auc,
        "kappa" => kappa,
    ), y_pred
end

function multiclass_classification_metrics(y_true::Vector{String}, y_pred::Vector{String}, scores::Matrix{Float64}, classes::Vector{String})
    idx = Dict(cls => i for (i, cls) in enumerate(classes))
    conf = zeros(Float64, length(classes), length(classes))
    for (yt, yp) in zip(y_true, y_pred)
        conf[idx[yt], idx[yp]] += 1
    end
    n = sum(conf)
    accuracy = n == 0 ? missing : sum(diag(conf)) / n
    precision_v = Float64[]
    recall_v = Float64[]
    f1_v = Float64[]
    fpr_v = Float64[]
    auc_v = Float64[]
    for (j, cls) in enumerate(classes)
        tp = conf[j, j]
        fn = sum(conf[j, :]) - tp
        fp = sum(conf[:, j]) - tp
        tn = n - tp - fn - fp
        prec = (tp + fp) == 0 ? NaN : tp / (tp + fp)
        rec = (tp + fn) == 0 ? NaN : tp / (tp + fn)
        f1 = (isnan(prec) || isnan(rec) || (prec + rec == 0)) ? NaN : 2 * prec * rec / (prec + rec)
        fpr = (fp + tn) == 0 ? NaN : fp / (fp + tn)
        push!(precision_v, prec)
        push!(recall_v, rec)
        push!(f1_v, f1)
        push!(fpr_v, fpr)
        auc = _binary_auc([yt == cls ? 1.0 : 0.0 for yt in y_true], scores[:, j])
        if !ismissing(auc)
            push!(auc_v, Float64(auc))
        end
    end
    mean_nonan(v) = begin
        vv = filter(x -> !isnan(x), v)
        isempty(vv) ? missing : mean(vv)
    end
    return Dict(
        "accuracy" => accuracy,
        "sensitivity" => mean_nonan(recall_v),
        "precision" => mean_nonan(precision_v),
        "f1" => mean_nonan(f1_v),
        "fpr" => mean_nonan(fpr_v),
        "roc_auc" => (isempty(auc_v) ? missing : mean(auc_v)),
        "kappa" => _kappa_from_confusion(conf),
    )
end

function train_dome(Xtr::Matrix{Float64}, ytr::Vector{Float64}, params::DoMEParams;
    verbose::Bool=true,
    max_steps::Union{Nothing,Int}=nothing,
    max_time_sec::Union{Nothing,Float64}=nothing,
    log_every::Int=0,
)
    verbose && println("[TRAIN] max_nodes=$(params.max_nodes) min_improvement=$(params.min_improvement) div=$(params.use_division)")

    dome_obj = SymDoME.DoME(
        Xtr,
        ytr;
        maximumNodes=params.max_nodes,
        minimumReductionMSE=params.min_improvement,
        useDivisionOperator=params.use_division,
        strategy=params.strategy,
    )

    history = Float64[dome_obj.mse]
    t0 = time()
    steps = 0
    stop_reason = "no_improvement"

    while true
        if max_steps !== nothing && steps >= max_steps
            stop_reason = "max_steps"
            break
        end
        if max_time_sec !== nothing && (time() - t0) >= max_time_sec
            stop_reason = "max_time_sec"
            break
        end

        improved = SymDoME.Step!(dome_obj)
        steps += 1
        push!(history, dome_obj.mse)

        if verbose && log_every > 0 && (steps % log_every == 0)
            println("[TRAIN] step=$steps mse=$(history[end])")
        end

        if !improved
            stop_reason = "no_improvement"
            break
        end
    end

    return dome_obj.tree, history, stop_reason
end

function evaluate_dome(tree, Xte::Matrix{Float64}; need_expression::Bool=false)
    raw_pred = SymDoME.evaluateTree(tree, Xte)
    ŷ = if raw_pred isa Number
        fill(Float64(raw_pred), size(Xte, 1))
    else
        vec(Float64.(collect(raw_pred)))
    end
    expr = need_expression ? SymDoME.vectorString(tree) : nothing
    return ŷ, expr
end

function _fit_multiclass_ovr(Xtr::Matrix{Float64}, ytr::Vector{String}, classes::Vector{String}, params::DoMEParams;
    verbose::Bool=true,
    max_steps::Union{Nothing,Int}=nothing,
    max_time_sec::Union{Nothing,Float64}=nothing,
    log_every::Int=0,
)
    trees = Any[]
    histories = Vector{Vector{Float64}}()
    expressions = String[]
    stop_reasons = String[]
    for cls in classes
        ybin = [y == cls ? 1.0 : 0.0 for y in ytr]
        tree, hist, stop_reason = train_dome(Xtr, ybin, params;
            verbose=verbose,
            max_steps=max_steps,
            max_time_sec=max_time_sec,
            log_every=log_every,
        )
        push!(trees, tree)
        push!(histories, hist)
        push!(expressions, SymDoME.vectorString(tree))
        push!(stop_reasons, stop_reason)
    end
    return trees, histories, expressions, stop_reasons
end

function _predict_multiclass_ovr(trees, Xte::Matrix{Float64}, classes::Vector{String})
    scores = Matrix{Float64}(undef, size(Xte, 1), length(classes))
    for (j, tree) in enumerate(trees)
        scores[:, j] .= collect(SymDoME.evaluateTree(tree, Xte))
    end
    preds = Vector{String}(undef, size(Xte, 1))
    for i in 1:size(Xte, 1)
        _, j = findmax(view(scores, i, :))
        preds[i] = classes[j]
    end
    return preds, scores
end

function _metric_lookup(metrics::AbstractDict{String,<:Any}, key::String)
    haskey(metrics, key) || error("Métrica de referencia no soportada/ausente: $key")
    v = metrics[key]
    (v isa Missing) && error("La métrica de referencia '$key' no se pudo calcular")
    return Float64(v)
end

function _history_summary_for_multiclass(histories::Vector{Vector{Float64}})
    isempty(histories) && return missing, missing
    initials = Float64[]
    finals = Float64[]
    for h in histories
        isempty(h) && continue
        push!(initials, h[1])
        push!(finals, h[end])
    end
    return isempty(initials) ? missing : mean(initials), isempty(finals) ? missing : mean(finals)
end

function run_experiment(; dataset::String, window::Int, normalization::String="standard", reference_metric::String="mae", model::Symbol=:DoME,
    config_id::Int, seed::Int=1, data_dir::String=default_data_dir(), horizon::Int=1,
    target_col::Union{String,Int,Nothing}=nothing, train_ratio::Float64=0.75, task_type::Symbol=:regression,
    categorical_encoding::String="error", input_booleanization::String="none", save_detailed::Bool=true,
    save_predictions::Bool=false, predictions_dir::Union{Nothing,String}=nothing, verbose::Bool=true,
    train_verbose::Bool=false, train_max_steps::Union{Nothing,Int}=nothing,
    train_max_time_sec::Union{Nothing,Float64}=nothing, train_log_every::Int=0)

    Random.seed!(seed)
    train_max_steps = isnothing(train_max_steps) ? nothing : (Int(train_max_steps) > 0 ? Int(train_max_steps) : nothing)
    train_max_time_sec = isnothing(train_max_time_sec) ? nothing : (Float64(train_max_time_sec) > 0 ? Float64(train_max_time_sec) : nothing)
    train_log_every = max(Int(train_log_every), 0)
    configs = modelConfigurations(model)
    (1 <= config_id <= length(configs)) || error("config_id fuera de rango: $config_id (total: $(length(configs)))")
    cfg = configs[config_id]
    params = DoMEParams(cfg["maxNumNodes"], cfg["minimumReductionMSE"], cfg["useDivisionOperator"], cfg["strategy"])
    strategy_name = cfg["strategyName"]

    eval_mode = lowercase(strip(get(ENV, "EVAL_MODE", "")))
    if eval_mode == "bml_horizon_tabular"
        return run_experiment_bml_horizon_tabular(
            dataset=dataset,
            window=window,
            normalization=normalization,
            reference_metric=reference_metric,
            model=model,
            config_id=config_id,
            seed=seed,
            data_dir=data_dir,
            horizon=horizon,
            target_col=target_col,
            train_ratio=train_ratio,
            task_type=task_type,
            categorical_encoding=categorical_encoding,
            input_booleanization=input_booleanization,
            save_detailed=save_detailed,
            save_predictions=save_predictions,
            predictions_dir=predictions_dir,
            verbose=verbose,
            train_verbose=train_verbose,
            train_max_steps=train_max_steps,
            train_max_time_sec=train_max_time_sec,
            train_log_every=train_log_every,
        )
    end

    total_t0 = time()
    Xtr, ytr, Xte, yte, dsmeta = load_dataset(dataset; data_dir=data_dir, window=window, horizon=horizon, train_ratio=train_ratio,
        target_col=target_col, task_type=task_type, categorical_encoding=categorical_encoding, input_booleanization=input_booleanization, verbose=verbose)

    Xtr_n, Xte_n, ytr_n, yte_n, norm_stats = apply_normalization(Xtr, Xte, ytr, yte, normalization, task_type)
    train_t0 = time()
    metrics = Dict{String,Any}()
    expr = nothing
    history = Float64[]
    stop_reason = "no_improvement"
    predictions_df = nothing
    mse_initial = missing
    mse_final_train = missing
    iterations_value = missing

    if task_type == :regression || task_type == :binary_classification
        ytr_float = task_type == :regression ? ytr_n : Float64.(ytr_n)
        tree, history, stop_reason = train_dome(
            Xtr_n,
            ytr_float,
            params;
            verbose=train_verbose,
            max_steps=train_max_steps,
            max_time_sec=train_max_time_sec,
            log_every=train_log_every,
        )
        train_time_sec = time() - train_t0
        ŷ_test_norm, expr = evaluate_dome(tree, Xte_n; need_expression=save_detailed)
        ŷ_raw = denormalize_predictions(ŷ_test_norm, norm_stats)
        if task_type == :regression
            metrics = regression_metrics(Float64.(yte), ŷ_raw)
            reference_score = _metric_lookup(metrics, lowercase(reference_metric))
            if save_predictions
                predictions_df = DataFrame(y_true_raw=Float64.(yte), y_pred_raw=ŷ_raw, y_true_norm=Float64.(yte_n), y_pred_norm=ŷ_test_norm)
            end
        else
            metrics, yhat = binary_classification_metrics(Float64.(yte), ŷ_raw; threshold=0.5)
            reference_score = _metric_lookup(metrics, lowercase(reference_metric))
            if save_predictions
                predictions_df = DataFrame(y_true=Float64.(yte), score=ŷ_raw, y_pred=yhat)
            end
        end
        mse_initial = isempty(history) ? missing : history[1]
        mse_final_train = isempty(history) ? missing : history[end]
        iterations_value = isempty(history) ? missing : max(length(history) - 1, 0)
    elseif task_type == :multiclass_classification
        classes = Vector{String}(dsmeta["class_names"])
        trees, histories, expressions, stop_reasons = _fit_multiclass_ovr(
            Xtr_n,
            Vector{String}(ytr_n),
            classes,
            params;
            verbose=train_verbose,
            max_steps=train_max_steps,
            max_time_sec=train_max_time_sec,
            log_every=train_log_every,
        )
        train_time_sec = time() - train_t0
        expr = join(expressions, "\n---\n")
        yhat, scores = _predict_multiclass_ovr(trees, Xte_n, classes)
        metrics = multiclass_classification_metrics(Vector{String}(yte), yhat, scores, classes)
        reference_score = _metric_lookup(metrics, lowercase(reference_metric))
        if save_predictions
            predictions_df = DataFrame(y_true=Vector{String}(yte), y_pred=yhat)
        end
        m0, mf = _history_summary_for_multiclass(histories)
        mse_initial = m0
        mse_final_train = mf
        stop_reason = isempty(stop_reasons) ? "no_improvement" : join(unique(stop_reasons), ",")
        history = Float64[]
        iter_vals = Int[]
        for h in histories
            if !isempty(h)
                push!(history, h[end])
                push!(iter_vals, max(length(h) - 1, 0))
            end
        end
        iterations_value = isempty(iter_vals) ? missing : round(Int, mean(iter_vals))
    else
        error("task_type no soportado: $task_type")
    end

    improvement_pct = (mse_initial isa Missing || mse_initial == 0.0 || mse_final_train isa Missing) ? missing : (1 - mse_final_train / mse_initial) * 100
    total_time_sec = time() - total_t0
    dataset_name = replace(split(dataset, "/")[end], ".csv" => "", ".txt" => "")
    ts = Dates.now()

    results_df = DataFrame(
        dataset=[dataset], dataset_name=[dataset_name], task_type=[String(task_type)], target_name=[String(dsmeta["target_name"])],
        window=[window], normalization=[normalization], categorical_encoding=[categorical_encoding], input_booleanization=[input_booleanization],
        reference_metric=[lowercase(reference_metric)], reference_score=[reference_score], model=[string(model)], config_id=[config_id], seed=[seed],
        max_nodes=[params.max_nodes], min_improvement=[params.min_improvement], use_division=[params.use_division], strategy=[strategy_name],
        mse_initial=[mse_initial], mse_final_train=[mse_final_train], improvement_pct=[improvement_pct],
        accuracy=[get(metrics, "accuracy", missing)], sensitivity=[get(metrics, "sensitivity", missing)], precision=[get(metrics, "precision", missing)],
        f1=[get(metrics, "f1", missing)], fpr=[get(metrics, "fpr", missing)], roc_auc=[get(metrics, "roc_auc", missing)], kappa=[get(metrics, "kappa", missing)],
        mse_test_raw=[get(metrics, "mse", missing)], rmse_test_raw=[get(metrics, "rmse", missing)], mae_test_raw=[get(metrics, "mae", missing)],
        r2_test_raw=[get(metrics, "r2", missing)], nrmse_test_raw=[get(metrics, "nrmse", missing)],
        iterations=[iterations_value], stop_reason=[stop_reason], train_time_sec=[train_time_sec], total_time_sec=[total_time_sec],
        train_samples=[size(Xtr, 1)], test_samples=[size(Xte, 1)], features=[size(Xtr, 2)], timestamp=[ts], error=[""],
    )

    detailed_dict = nothing
    if save_detailed
        detailed_dict = Dict{String,Any}(
            "dataset" => dataset,
            "dataset_name" => dataset_name,
            "task_type" => String(task_type),
            "target_name" => String(dsmeta["target_name"]),
            "class_names" => get(dsmeta, "class_names", String[]),
            "window" => window,
            "horizon" => horizon,
            "normalization" => normalization,
            "categorical_encoding" => categorical_encoding,
            "input_booleanization" => input_booleanization,
            "reference_metric" => lowercase(reference_metric),
            "reference_score" => reference_score,
            "model" => string(model),
            "config_id" => config_id,
            "seed" => seed,
            "max_nodes" => params.max_nodes,
            "min_improvement" => params.min_improvement,
            "use_division" => params.use_division,
            "strategy" => strategy_name,
            "train_verbose" => train_verbose,
            "train_max_steps" => train_max_steps,
            "train_max_time_sec" => train_max_time_sec,
            "train_log_every" => train_log_every,
            "expression" => (expr === nothing ? "" : expr),
            "history" => history,
            "mse_initial" => mse_initial,
            "mse_final_train" => mse_final_train,
            "improvement_pct" => improvement_pct,
            "metrics" => metrics,
            "iterations" => iterations_value,
            "stop_reason" => stop_reason,
            "train_time_sec" => train_time_sec,
            "total_time_sec" => total_time_sec,
            "dataset_meta" => dsmeta,
            "timestamp" => string(ts),
        )
    end

    return results_df, detailed_dict, predictions_df
end



# FRIEDMAN_BATCH_HELPERS_MARKER
function _batch_ranges(n::Int, horizon::Int)
    horizon > 0 || error("El horizon debe ser > 0")
    ranges = Vector{UnitRange{Int}}()
    i = 1
    while i <= n
        j = min(i + horizon - 1, n)
        push!(ranges, i:j)
        i = j + 1
    end
    return ranges
end

function _model_size_mb(obj)
    try
        return Float64(Base.summarysize(obj)) / (1024.0^2)
    catch
        return 0.0
    end
end

function _tabular_choose_best(cands::Vector{String}, base_lower::String)
    isempty(cands) && return nothing
    pref = [c for c in cands if occursin(base_lower, lowercase(basename(c)))]
    !isempty(pref) && return pref[1]
    length(cands) == 1 && return cands[1]
    sort!(cands, by = x -> length(basename(x)))
    return cands[1]
end

function _resolve_tabular_source(dataset_name::String, data_dir::String)
    base_path = (isfile(dataset_name) || isdir(dataset_name)) ? dataset_name : joinpath(data_dir, dataset_name)

    train_file = nothing
    test_file = nothing
    single_file = nothing

    base_dir = isdir(base_path) ? base_path : dirname(base_path)
    base_name = basename(base_path)
    base_lower = lowercase(base_name)
    if !isdir(base_dir)
        base_dir = data_dir
    end

    if isdir(base_dir)
        all_files = readdir(base_dir)
        train_candidates = String[]
        test_candidates = String[]
        single_candidates = String[]
        for file in all_files
            full = joinpath(base_dir, file)
            isfile(full) || continue
            fl = lowercase(file)
            occursin("label", fl) && continue
            if occursin("train", fl)
                push!(train_candidates, full)
            elseif occursin("test", fl)
                push!(test_candidates, full)
            elseif endswith(fl, ".csv") || endswith(fl, ".txt")
                push!(single_candidates, full)
            end
        end
        train_file = _tabular_choose_best(train_candidates, base_lower)
        test_file = _tabular_choose_best(test_candidates, base_lower)
        single_file = _tabular_choose_best(single_candidates, base_lower)
    elseif isfile(base_path)
        single_file = base_path
    end

    return train_file, test_file, single_file
end

function _count_csv_rows(path::String)
    n = 0
    for _ in CSV.Rows(path; delim=',', reusebuffer=true)
        n += 1
    end
    return n
end

function _row_to_namedtuple(row)
    pnames = Tuple(propertynames(row))
    vals = Tuple(getproperty(row, p) for p in pnames)
    return NamedTuple{pnames}(vals)
end

function _df_to_numeric_matrix(df::DataFrame, cols::Vector{Int})
    n = nrow(df)
    d = length(cols)
    M = Matrix{Float64}(undef, n, d)
    for (j, c) in enumerate(cols)
        col = df[!, c]
        @inbounds for i in 1:n
            M[i, j] = Float64(col[i])
        end
    end
    return M
end

function _fit_normalization_stats(Xtr::Matrix{Float64}, ytr::Vector{Float64}, normalization::String, task_type::Symbol)
    norm = lowercase(strip(normalization))
    if norm in ("none", "no", "false")
        return Dict{String,Any}("method" => "none", "y_scaled" => false)
    elseif norm in ("standard", "zscore", "standardscaler")
        μx = vec(mean(Xtr, dims=1))
        σx = [_safe_std(view(Xtr, :, j)) for j in 1:size(Xtr, 2)]
        if task_type == :regression
            μy = mean(ytr)
            σy = _safe_std(ytr)
            return Dict{String,Any}(
                "method" => "standard",
                "μx" => μx,
                "σx" => σx,
                "μy" => μy,
                "σy" => σy,
                "y_scaled" => true,
            )
        else
            return Dict{String,Any}(
                "method" => "standard",
                "μx" => μx,
                "σx" => σx,
                "y_scaled" => false,
            )
        end
    elseif norm in ("minmax", "maxmin")
        X_min = vec(minimum(Xtr, dims=1))
        X_max = vec(maximum(Xtr, dims=1))
        X_rng = X_max .- X_min
        X_rng[X_rng .== 0.0] .= 1.0
        if task_type == :regression
            y_min = minimum(ytr)
            y_max = maximum(ytr)
            y_rng = y_max - y_min
            y_rng = (y_rng == 0.0) ? 1.0 : y_rng
            return Dict{String,Any}(
                "method" => "minmax",
                "X_min" => X_min,
                "X_rng" => X_rng,
                "y_min" => y_min,
                "y_rng" => y_rng,
                "y_scaled" => true,
            )
        else
            return Dict{String,Any}(
                "method" => "minmax",
                "X_min" => X_min,
                "X_rng" => X_rng,
                "y_scaled" => false,
            )
        end
    else
        error("Normalización no soportada: $normalization")
    end
end

function _apply_normalization_inplace!(X::Matrix{Float64}, y::Vector{Float64}, stats::Dict{String,Any}, task_type::Symbol)
    method = String(get(stats, "method", "none"))
    if method == "none"
        return X, y
    elseif method == "standard"
        μx = Vector{Float64}(stats["μx"])
        σx = Vector{Float64}(stats["σx"])
        @inbounds for j in 1:size(X, 2)
            X[:, j] .= (X[:, j] .- μx[j]) ./ σx[j]
        end
        if task_type == :regression && get(stats, "y_scaled", false)
            μy = Float64(stats["μy"])
            σy = Float64(stats["σy"])
            y .= (y .- μy) ./ σy
        end
        return X, y
    elseif method == "minmax"
        X_min = Vector{Float64}(stats["X_min"])
        X_rng = Vector{Float64}(stats["X_rng"])
        @inbounds for j in 1:size(X, 2)
            X[:, j] .= (X[:, j] .- X_min[j]) ./ X_rng[j]
        end
        if task_type == :regression && get(stats, "y_scaled", false)
            y_min = Float64(stats["y_min"])
            y_rng = Float64(stats["y_rng"])
            y .= (y .- y_min) ./ y_rng
        end
        return X, y
    else
        error("Método de normalización no soportado: $method")
    end
end

function _normalized_batch_copy(X::Matrix{Float64}, y::Vector{Float64}, stats::Dict{String,Any}, task_type::Symbol)
    Xc = copy(X)
    yc = copy(y)
    _apply_normalization_inplace!(Xc, yc, stats, task_type)
    return Xc, yc
end

function _numeric_train_means(df::DataFrame)
    means = Dict{String,Float64}()
    for c in names(df)
        T = _nonmissingtype(eltype(df[!, c]))
        if T <: Real
            vals = Float64[]
            for v in df[!, c]
                if !ismissing(v)
                    push!(vals, Float64(v))
                end
            end
            means[String(c)] = isempty(vals) ? 0.0 : mean(vals)
        end
    end
    return means
end

function _prepare_train_tabular_stream(train_df::DataFrame, dataset_name::String, target_col;
    categorical_encoding::String="error", input_booleanization::String="none", verbose::Bool=true)

    dummy_df = deepcopy(train_df)

    target_col_idx = get_target_column(train_df, target_col)
    preprocess_dataset!(train_df, dummy_df, dataset_name; verbose=verbose)
    target_col_idx = get_target_column(train_df, target_col)

    summary = _count_feature_types(train_df, target_col_idx)
    verbose && println("  Resumen de atributos -> numéricos=$(summary.numeric), categóricos=$(summary.categorical), temporales=$(summary.temporal)")

    numeric_cols = Int[]
    for i in 1:ncol(train_df)
        T = _nonmissingtype(eltype(train_df[!, i]))
        if T <: Real
            push!(numeric_cols, i)
        end
    end

    target_adj = findfirst(==(target_col_idx), numeric_cols)
    target_adj === nothing && error("La variable objetivo no es numérica tras el preprocesado")

    train_mat = _df_to_numeric_matrix(train_df, numeric_cols)
    feature_idx = [j for j in 1:size(train_mat, 2) if j != target_adj]
    isempty(feature_idx) && error("No quedan features tras quitar la variable objetivo")

    Xtr = train_mat[:, feature_idx]
    ytr = vec(Float64.(train_mat[:, target_adj]))
    train_means = _numeric_train_means(train_df)

    dsmeta = Dict(
        "target_name" => String(names(train_df)[target_col_idx]),
        "class_names" => String[],
        "summary_before_encoding" => summary,
        "categorical_encoding" => "none",
        "input_booleanization" => "none",
        "total_features_after_encoding" => size(Xtr, 2),
        "tabular_mode" => true,
    )

    return Xtr, ytr, dsmeta, names(train_df), numeric_cols, target_adj, feature_idx, train_means
end

function _prepare_test_batch_tabular!(batch_df::DataFrame, final_cols, dataset_name::String, train_means::Dict{String,Float64})
    ds = _normalize_dataset_id(dataset_name)

    if occursin("hour", ds)
        drop_cols = String[]
        for c in ("instant", "dteday", "casual", "registered")
            if c in names(batch_df) && !(c in final_cols)
                push!(drop_cols, c)
            end
        end
        if !isempty(drop_cols)
            select!(batch_df, Not(drop_cols))
        end
    end

    for c in names(batch_df)
        _coerce_numeric_column!(batch_df, String(c))
    end

    remaining_non_numeric = String[]
    for c in names(batch_df)
        T = _nonmissingtype(eltype(batch_df[!, c]))
        if !(T <: Real)
            push!(remaining_non_numeric, String(c))
        end
    end
    isempty(remaining_non_numeric) || error(
        "Quedan columnas no numéricas sin transformar en batch: $(remaining_non_numeric)"
    )

    _replace_nonfinite_with_missing!(batch_df)

    for c in names(batch_df)
        T = _nonmissingtype(eltype(batch_df[!, c]))
        if T <: Real
            μ = get(train_means, String(c), 0.0)
            col = batch_df[!, c]
            for i in eachindex(col)
                if ismissing(col[i])
                    col[i] = μ
                end
            end
        end
    end

    missing_cols = setdiff(final_cols, names(batch_df))
    isempty(missing_cols) || error("Faltan columnas en batch tras preprocesado: $(missing_cols)")

    select!(batch_df, final_cols)
    return batch_df
end

function run_experiment_bml_horizon_tabular(; dataset::String, window::Int, normalization::String="standard", reference_metric::String="mae", model::Symbol=:DoME,
    config_id::Int, seed::Int=1, data_dir::String=default_data_dir(), horizon::Int=1,
    target_col::Union{String,Int,Nothing}=nothing, train_ratio::Float64=0.75, task_type::Symbol=:regression,
    categorical_encoding::String="error", input_booleanization::String="none", save_detailed::Bool=true,
    save_predictions::Bool=false, predictions_dir::Union{Nothing,String}=nothing, verbose::Bool=true,
    train_verbose::Bool=false, train_max_steps::Union{Nothing,Int}=nothing,
    train_max_time_sec::Union{Nothing,Float64}=nothing, train_log_every::Int=0)

    task_type == :regression || error("El modo bml_horizon_tabular solo soporta regresión")

    Random.seed!(seed)
    train_max_steps = isnothing(train_max_steps) ? nothing : (Int(train_max_steps) > 0 ? Int(train_max_steps) : nothing)
    train_max_time_sec = isnothing(train_max_time_sec) ? nothing : (Float64(train_max_time_sec) > 0 ? Float64(train_max_time_sec) : nothing)
    train_log_every = max(Int(train_log_every), 0)

    configs = modelConfigurations(model)
    (1 <= config_id <= length(configs)) || error("config_id fuera de rango: $config_id (total: $(length(configs)))")
    cfg = configs[config_id]
    params = DoMEParams(cfg["maxNumNodes"], cfg["minimumReductionMSE"], cfg["useDivisionOperator"], cfg["strategy"])
    strategy_name = cfg["strategyName"]

    total_t0 = time()

    train_file, test_file, single_file = _resolve_tabular_source(dataset, data_dir)

    train_df = nothing
    n_test_total = 0
    train_rows = 0

    if single_file !== nothing
        verbose && begin
            println("\n" * "="^70)
            println("CARGANDO DATASET TABULAR EN MODO STREAM: $dataset")
            println("="^70)
            println("Formato: archivo único (test en streaming por lotes)")
            println("  Archivo: $single_file")
        end
        n_total = _count_csv_rows(single_file)
        n_total >= 2 || error("No hay suficientes filas en el dataset: $single_file")
        train_rows = max(1, floor(Int, train_ratio * n_total))
        train_rows = min(train_rows, n_total - 1)
        n_test_total = n_total - train_rows
        verbose && println("  Split cronológico: train=$(train_rows)  test=$(n_test_total)")
        train_df = CSV.read(single_file, DataFrame; delim=',', limit=train_rows, silencewarnings=true, ignorerepeated=true)
    elseif train_file !== nothing && test_file !== nothing
        verbose && begin
            println("\n" * "="^70)
            println("CARGANDO DATASET TABULAR EN MODO STREAM: $dataset")
            println("="^70)
            println("Formato: archivos TRAIN/TEST (test en streaming por lotes)")
            println("  Train: $train_file")
            println("  Test:  $test_file")
        end
        train_df = safe_read_file(train_file; verbose=verbose)
        train_rows = nrow(train_df)
        n_test_total = _count_csv_rows(test_file)
    else
        # fallback al comportamiento antiguo si no se puede resolver la fuente stream
        Xtr, ytr, Xte, yte, dsmeta = load_dataset_tabular(
            dataset;
            data_dir=data_dir,
            train_ratio=train_ratio,
            target_col=target_col,
            task_type=task_type,
            categorical_encoding=categorical_encoding,
            input_booleanization=input_booleanization,
            verbose=verbose,
        )

        Xtr_n, Xte_n, ytr_n, yte_n, norm_stats = apply_normalization(Xtr, Xte, ytr, yte, normalization, task_type)

        train_t0 = time()
        tree, history, stop_reason = train_dome(
            Xtr_n,
            ytr_n,
            params;
            verbose=train_verbose,
            max_steps=train_max_steps,
            max_time_sec=train_max_time_sec,
            log_every=train_log_every,
        )
        train_time_sec = time() - train_t0

        expr = save_detailed ? SymDoME.vectorString(tree) : nothing
        mse_initial = isempty(history) ? missing : history[1]
        mse_final_train = isempty(history) ? missing : history[end]
        iterations_value = isempty(history) ? missing : max(length(history) - 1, 0)
        model_memory_mb = _model_size_mb(tree)

        batches = _batch_ranges(length(yte), horizon)

        batch_metrics_df = DataFrame(
            batch_id=Int[],
            batch_start=Int[],
            batch_end=Int[],
            batch_size=Int[],
            MAE=Float64[],
            MSE=Float64[],
            RMSE=Float64[],
            R2=Float64[],
            NRMSE=Float64[],
            Memory_MB=Float64[],
            CompTime_s=Float64[],
        )

        predictions_df = save_predictions ? DataFrame(
            batch_id=Int[],
            row_in_test=Int[],
            y_true_raw=Float64[],
            y_pred_raw=Float64[],
        ) : nothing

        for (bi, rg) in enumerate(batches)
            pt0 = time()
            yhat_norm, _ = evaluate_dome(tree, Xte_n[rg, :]; need_expression=false)
            comp_time_sec = time() - pt0
            yhat_raw = denormalize_predictions(yhat_norm, norm_stats)

            mm = regression_metrics(Float64.(yte[rg]), yhat_raw)

            push!(batch_metrics_df, (
                batch_id=bi,
                batch_start=first(rg),
                batch_end=last(rg),
                batch_size=length(rg),
                MAE=Float64(mm["mae"]),
                MSE=Float64(mm["mse"]),
                RMSE=Float64(mm["rmse"]),
                R2=Float64(mm["r2"]),
                NRMSE=Float64(mm["nrmse"]),
                Memory_MB=model_memory_mb,
                CompTime_s=Float64(comp_time_sec),
            ))

            if save_predictions
                for (off, idx) in enumerate(rg)
                    push!(predictions_df, (
                        batch_id=bi,
                        row_in_test=idx,
                        y_true_raw=Float64(yte[idx]),
                        y_pred_raw=Float64(yhat_raw[off]),
                    ))
                end
            end
        end

        agg_metrics = Dict(
            "mae" => mean(batch_metrics_df.MAE),
            "mse" => mean(batch_metrics_df.MSE),
            "rmse" => mean(batch_metrics_df.RMSE),
            "r2" => mean(batch_metrics_df.R2),
            "nrmse" => mean(batch_metrics_df.NRMSE),
        )

        reference_score = _metric_lookup(agg_metrics, lowercase(reference_metric))
        improvement_pct = (mse_initial isa Missing || mse_initial == 0.0 || mse_final_train isa Missing) ? missing : (1 - mse_final_train / mse_initial) * 100
        total_time_sec = time() - total_t0
        dataset_name = replace(split(dataset, "/")[end], ".csv" => "", ".txt" => "")
        ts = Dates.now()

        results_df = DataFrame(
            dataset=[dataset],
            dataset_name=[dataset_name],
            task_type=[String(task_type)],
            target_name=[String(dsmeta["target_name"])],
            window=[window],
            horizon=[horizon],
            eval_mode=["bml_horizon_tabular"],
            normalization=[normalization],
            categorical_encoding=[categorical_encoding],
            input_booleanization=[input_booleanization],
            reference_metric=[lowercase(reference_metric)],
            reference_score=[reference_score],
            model=[string(model)],
            config_id=[config_id],
            seed=[seed],
            max_nodes=[params.max_nodes],
            min_improvement=[params.min_improvement],
            use_division=[params.use_division],
            strategy=[strategy_name],
            mse_initial=[mse_initial],
            mse_final_train=[mse_final_train],
            improvement_pct=[improvement_pct],
            accuracy=[missing],
            sensitivity=[missing],
            precision=[missing],
            f1=[missing],
            fpr=[missing],
            roc_auc=[missing],
            kappa=[missing],
            mse_test_raw=[agg_metrics["mse"]],
            rmse_test_raw=[agg_metrics["rmse"]],
            mae_test_raw=[agg_metrics["mae"]],
            r2_test_raw=[agg_metrics["r2"]],
            nrmse_test_raw=[agg_metrics["nrmse"]],
            iterations=[iterations_value],
            stop_reason=[stop_reason],
            train_time_sec=[train_time_sec],
            total_time_sec=[total_time_sec],
            train_samples=[size(Xtr, 1)],
            test_samples=[size(Xte, 1)],
            features=[size(Xtr, 2)],
            batches=[length(batches)],
            include_remainder=[(length(yte) % horizon) != 0],
            remainder_rows=[length(yte) % horizon],
            memory_mb_peak=[model_memory_mb],
            timestamp=[ts],
            error=[""],
        )

        detailed_dict = nothing
        if save_detailed
            detailed_dict = Dict{String,Any}(
                "dataset" => dataset,
                "dataset_name" => dataset_name,
                "task_type" => String(task_type),
                "target_name" => String(dsmeta["target_name"]),
                "class_names" => get(dsmeta, "class_names", String[]),
                "window" => window,
                "window_metadata_only" => true,
                "horizon" => horizon,
                "normalization" => normalization,
                "categorical_encoding" => categorical_encoding,
                "input_booleanization" => input_booleanization,
                "reference_metric" => lowercase(reference_metric),
                "reference_score" => reference_score,
                "model" => string(model),
                "config_id" => config_id,
                "seed" => seed,
                "max_nodes" => params.max_nodes,
                "min_improvement" => params.min_improvement,
                "use_division" => params.use_division,
                "strategy" => strategy_name,
                "train_verbose" => train_verbose,
                "train_max_steps" => train_max_steps,
                "train_max_time_sec" => train_max_time_sec,
                "train_log_every" => train_log_every,
                "expression" => (expr === nothing ? "" : expr),
                "history" => history,
                "mse_initial" => mse_initial,
                "mse_final_train" => mse_final_train,
                "improvement_pct" => improvement_pct,
                "metrics" => agg_metrics,
                "iterations" => iterations_value,
                "stop_reason" => stop_reason,
                "train_time_sec" => train_time_sec,
                "total_time_sec" => total_time_sec,
                "dataset_meta" => dsmeta,
                "timestamp" => string(ts),
                "evaluation_mode" => "bml_horizon_tabular",
                "fixed_model_after_training" => true,
                "train_size_initial" => size(Xtr, 1),
                "test_size_total" => size(Xte, 1),
                "include_remainder" => ((length(yte) % horizon) != 0),
                "remainder_rows" => (length(yte) % horizon),
                "evaluated_test_rows" => length(yte),
                "batch_metrics" => batch_metrics_df,
                "memory_mb_peak" => model_memory_mb,
            )
        end

        return results_df, detailed_dict, predictions_df
    end

    Xtr, ytr, dsmeta, final_cols, numeric_cols, target_adj, feature_idx, train_means = _prepare_train_tabular_stream(
        train_df,
        dataset,
        target_col;
        categorical_encoding=categorical_encoding,
        input_booleanization=input_booleanization,
        verbose=verbose,
    )

    norm_stats = _fit_normalization_stats(Xtr, ytr, normalization, task_type)
    Xtr_n = copy(Xtr)
    ytr_n = copy(ytr)
    _apply_normalization_inplace!(Xtr_n, ytr_n, norm_stats, task_type)

    train_t0 = time()
    tree, history, stop_reason = train_dome(
        Xtr_n,
        ytr_n,
        params;
        verbose=train_verbose,
        max_steps=train_max_steps,
        max_time_sec=train_max_time_sec,
        log_every=train_log_every,
    )
    train_time_sec = time() - train_t0

    expr = save_detailed ? SymDoME.vectorString(tree) : nothing
    mse_initial = isempty(history) ? missing : history[1]
    mse_final_train = isempty(history) ? missing : history[end]
    iterations_value = isempty(history) ? missing : max(length(history) - 1, 0)
    model_memory_mb = _model_size_mb(tree)

    batch_metrics_df = save_detailed ? DataFrame(
        batch_id=Int[],
        batch_start=Int[],
        batch_end=Int[],
        batch_size=Int[],
        MAE=Float64[],
        MSE=Float64[],
        RMSE=Float64[],
        R2=Float64[],
        NRMSE=Float64[],
        Memory_MB=Float64[],
        CompTime_s=Float64[],
    ) : nothing

    predictions_df = save_predictions ? DataFrame(
        batch_id=Int[],
        row_in_test=Int[],
        y_true_raw=Float64[],
        y_pred_raw=Float64[],
    ) : nothing

    sum_mae = 0.0
    sum_mse = 0.0
    sum_rmse = 0.0
    sum_r2 = 0.0
    sum_nrmse = 0.0
    nbatches = 0
    row_in_test_cursor = 0

    function _process_batch!(batch_df::DataFrame)
        isempty(batch_df) && return

        _prepare_test_batch_tabular!(batch_df, final_cols, dataset, train_means)

        batch_mat = _df_to_numeric_matrix(batch_df, numeric_cols)
        Xba = batch_mat[:, feature_idx]
        yba = vec(Float64.(batch_mat[:, target_adj]))
        Xba_n, yba_n = _normalized_batch_copy(Xba, yba, norm_stats, task_type)

        pt0 = time()
        yhat_norm, _ = evaluate_dome(tree, Xba_n; need_expression=false)
        comp_time_sec = time() - pt0
        yhat_raw = denormalize_predictions(yhat_norm, norm_stats)

        mm = regression_metrics(Float64.(yba), yhat_raw)

        nbatches += 1
        batch_start = row_in_test_cursor + 1
        batch_end = row_in_test_cursor + length(yba)
        row_in_test_cursor = batch_end

        sum_mae += Float64(mm["mae"])
        sum_mse += Float64(mm["mse"])
        sum_rmse += Float64(mm["rmse"])
        sum_r2 += Float64(mm["r2"])
        sum_nrmse += Float64(mm["nrmse"])

        if save_detailed
            push!(batch_metrics_df, (
                batch_id=nbatches,
                batch_start=batch_start,
                batch_end=batch_end,
                batch_size=length(yba),
                MAE=Float64(mm["mae"]),
                MSE=Float64(mm["mse"]),
                RMSE=Float64(mm["rmse"]),
                R2=Float64(mm["r2"]),
                NRMSE=Float64(mm["nrmse"]),
                Memory_MB=model_memory_mb,
                CompTime_s=Float64(comp_time_sec),
            ))
        end

        if save_predictions
            for i in 1:length(yba)
                push!(predictions_df, (
                    batch_id=nbatches,
                    row_in_test=batch_start + i - 1,
                    y_true_raw=Float64(yba[i]),
                    y_pred_raw=Float64(yhat_raw[i]),
                ))
            end
        end

        return nothing
    end

    batch_rows = NamedTuple[]

    if single_file !== nothing
        row_idx = 0
        for row in CSV.Rows(single_file; delim=',', reusebuffer=true)
            row_idx += 1
            row_idx <= train_rows && continue
            push!(batch_rows, _row_to_namedtuple(row))
            if length(batch_rows) == horizon
                _process_batch!(DataFrame(batch_rows))
                empty!(batch_rows)
            end
        end
    else
        for row in CSV.Rows(test_file; delim=',', reusebuffer=true)
            push!(batch_rows, _row_to_namedtuple(row))
            if length(batch_rows) == horizon
                _process_batch!(DataFrame(batch_rows))
                empty!(batch_rows)
            end
        end
    end

    if !isempty(batch_rows)
        _process_batch!(DataFrame(batch_rows))
        empty!(batch_rows)
    end

    nbatches > 0 || error("No se generó ningún batch de test")

    agg_metrics = Dict(
        "mae" => sum_mae / nbatches,
        "mse" => sum_mse / nbatches,
        "rmse" => sum_rmse / nbatches,
        "r2" => sum_r2 / nbatches,
        "nrmse" => sum_nrmse / nbatches,
    )

    reference_score = _metric_lookup(agg_metrics, lowercase(reference_metric))
    improvement_pct = (mse_initial isa Missing || mse_initial == 0.0 || mse_final_train isa Missing) ? missing : (1 - mse_final_train / mse_initial) * 100
    total_time_sec = time() - total_t0
    dataset_name = replace(split(dataset, "/")[end], ".csv" => "", ".txt" => "")
    ts = Dates.now()

    results_df = DataFrame(
        dataset=[dataset],
        dataset_name=[dataset_name],
        task_type=[String(task_type)],
        target_name=[String(dsmeta["target_name"])],
        window=[window],
        horizon=[horizon],
        eval_mode=["bml_horizon_tabular"],
        normalization=[normalization],
        categorical_encoding=[categorical_encoding],
        input_booleanization=[input_booleanization],
        reference_metric=[lowercase(reference_metric)],
        reference_score=[reference_score],
        model=[string(model)],
        config_id=[config_id],
        seed=[seed],
        max_nodes=[params.max_nodes],
        min_improvement=[params.min_improvement],
        use_division=[params.use_division],
        strategy=[strategy_name],
        mse_initial=[mse_initial],
        mse_final_train=[mse_final_train],
        improvement_pct=[improvement_pct],
        accuracy=[missing],
        sensitivity=[missing],
        precision=[missing],
        f1=[missing],
        fpr=[missing],
        roc_auc=[missing],
        kappa=[missing],
        mse_test_raw=[agg_metrics["mse"]],
        rmse_test_raw=[agg_metrics["rmse"]],
        mae_test_raw=[agg_metrics["mae"]],
        r2_test_raw=[agg_metrics["r2"]],
        nrmse_test_raw=[agg_metrics["nrmse"]],
        iterations=[iterations_value],
        stop_reason=[stop_reason],
        train_time_sec=[train_time_sec],
        total_time_sec=[total_time_sec],
        train_samples=[size(Xtr, 1)],
        test_samples=[n_test_total],
        features=[size(Xtr, 2)],
        batches=[nbatches],
        include_remainder=[(n_test_total % horizon) != 0],
        remainder_rows=[n_test_total % horizon],
        memory_mb_peak=[model_memory_mb],
        timestamp=[ts],
        error=[""],
    )

    detailed_dict = nothing
    if save_detailed
        detailed_dict = Dict{String,Any}(
            "dataset" => dataset,
            "dataset_name" => dataset_name,
            "task_type" => String(task_type),
            "target_name" => String(dsmeta["target_name"]),
            "class_names" => get(dsmeta, "class_names", String[]),
            "window" => window,
            "window_metadata_only" => true,
            "horizon" => horizon,
            "normalization" => normalization,
            "categorical_encoding" => categorical_encoding,
            "input_booleanization" => input_booleanization,
            "reference_metric" => lowercase(reference_metric),
            "reference_score" => reference_score,
            "model" => string(model),
            "config_id" => config_id,
            "seed" => seed,
            "max_nodes" => params.max_nodes,
            "min_improvement" => params.min_improvement,
            "use_division" => params.use_division,
            "strategy" => strategy_name,
            "train_verbose" => train_verbose,
            "train_max_steps" => train_max_steps,
            "train_max_time_sec" => train_max_time_sec,
            "train_log_every" => train_log_every,
            "expression" => (expr === nothing ? "" : expr),
            "history" => history,
            "mse_initial" => mse_initial,
            "mse_final_train" => mse_final_train,
            "improvement_pct" => improvement_pct,
            "metrics" => agg_metrics,
            "iterations" => iterations_value,
            "stop_reason" => stop_reason,
            "train_time_sec" => train_time_sec,
            "total_time_sec" => total_time_sec,
            "dataset_meta" => dsmeta,
            "timestamp" => string(ts),
            "evaluation_mode" => "bml_horizon_tabular",
            "fixed_model_after_training" => true,
            "train_size_initial" => size(Xtr, 1),
            "test_size_total" => n_test_total,
            "include_remainder" => ((n_test_total % horizon) != 0),
            "remainder_rows" => (n_test_total % horizon),
            "evaluated_test_rows" => n_test_total,
            "batch_metrics" => batch_metrics_df,
            "memory_mb_peak" => model_memory_mb,
        )
    end

    return results_df, detailed_dict, predictions_df
end


if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 7
        println("Uso: julia run_experiment_multi_unified.jl <dataset> <window> <normalization> <reference_metric> <model> <config_id> <seed>")
        exit(1)
    end

    dataset = ARGS[1]
    window = parse(Int, ARGS[2])
    normalization = ARGS[3]
    reference_metric = ARGS[4]
    model = Symbol(ARGS[5])
    config_id = parse(Int, ARGS[6])
    seed = parse(Int, ARGS[7])

    run_experiment(
        dataset=dataset,
        window=window,
        normalization=normalization,
        reference_metric=reference_metric,
        model=model,
        config_id=config_id,
        seed=seed,
        verbose=true,
        train_verbose=get(ENV, "TRAIN_VERBOSE", "0") in ("1", "true", "TRUE", "yes", "YES"),
        train_max_steps=let s = strip(get(ENV, "TRAIN_MAX_STEPS", "")); isempty(s) ? nothing : parse(Int, s) end,
        train_max_time_sec=let s = strip(get(ENV, "TRAIN_MAX_TIME_SEC", "")); isempty(s) ? nothing : parse(Float64, s) end,
        train_log_every=let s = strip(get(ENV, "TRAIN_LOG_EVERY", "")); isempty(s) ? 0 : parse(Int, s) end,
    )
end
