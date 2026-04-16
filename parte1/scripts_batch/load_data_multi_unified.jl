using CSV
using DataFrames
using Statistics
using DelimitedFiles
using Missings

_nonmissingtype(T) = try
    Base.nonmissingtype(T)
catch
    Missings.nonmissingtype(T)
end

const UNSW_RAW_COLUMNS = [
    "srcip", "sport", "dstip", "dsport", "proto", "state", "dur", "sbytes", "dbytes",
    "sttl", "dttl", "sloss", "dloss", "service", "Sload", "Dload", "Spkts", "Dpkts",
    "swin", "dwin", "stcpb", "dtcpb", "smeansz", "dmeansz", "trans_depth", "res_bdy_len",
    "Sjit", "Djit", "Stime", "Ltime", "Sintpkt", "Dintpkt", "tcprtt", "synack", "ackdat",
    "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd",
    "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm",
    "ct_dst_sport_ltm", "ct_dst_src_ltm", "attack_cat", "label"
]

function _looks_like_wrong_delim(df::DataFrame)::Bool
    return (ncol(df) == 1) && (length(names(df)) == 1) && (
        occursin(",", String(names(df)[1])) ||
        occursin(";", String(names(df)[1])) ||
        occursin('\t', String(names(df)[1]))
    )
end

function _normalize_dataset_id(x::AbstractString)
    s = lowercase(basename(String(x)))
    s = replace(s, r"\.csv$" => "")
    s = replace(s, r"\.txt$" => "")
    return s
end

function _strip_python_bytes_literal(s::AbstractString)
    str = strip(String(s))
    m = match(r"^b[\"'](.*)[\"']$", str)
    return m === nothing ? str : String(m.captures[1])
end

function _clean_string_value(v)
    if ismissing(v)
        return missing
    end
    s = strip(string(v))
    isempty(s) && return missing
    s = _strip_python_bytes_literal(s)
    s = strip(s)
    isempty(s) && return missing
    lowercase(s) in ("na", "nan", "null", "none", "missing") && return missing
    return s
end

function _force_categorical_columns(dataset_name::AbstractString)
    ds = _normalize_dataset_id(dataset_name)
    if ds == "electricity"
        return Set(["day"])
    end
    return Set{String}()
end

function parse_custom_format(filepath::String)::DataFrame
    lines = readlines(filepath)
    filter!(l -> !isempty(strip(l)), lines)
    isempty(lines) && error("Archivo vacío/no parseable: $filepath")

    rows = Vector{Vector{Float64}}()
    for line in lines
        tokens = split(line, r"[,;\s]+")
        nums = Float64[]
        for tok in tokens
            t = strip(tok)
            isempty(t) && continue
            try
                push!(nums, parse(Float64, t))
            catch
            end
        end
        !isempty(nums) && push!(rows, nums)
    end

    isempty(rows) && error("No se pudo parsear: $filepath")

    maxlen = maximum(length.(rows))
    mat = fill(NaN, length(rows), maxlen)
    for (i, r) in enumerate(rows)
        mat[i, 1:length(r)] .= r
    end

    col_names = ["X$i" for i in 1:size(mat, 2)]
    return DataFrame(mat, col_names)
end

function safe_read_file(filepath::String; verbose::Bool=true, kwargs...)::DataFrame
    verbose && println("Leyendo archivo: $filepath")
    ext = lowercase(splitext(filepath)[2])

    if ext == ".csv"
        try
            df = CSV.read(filepath, DataFrame; silencewarnings=true, ignorerepeated=true, kwargs...)
            if nrow(df) > 0 && ncol(df) > 0 && !_looks_like_wrong_delim(df)
                return df
            end
        catch
        end
    end

    try
        df = CSV.read(filepath, DataFrame; delim=';', decimal=',', silencewarnings=true, ignorerepeated=true, kwargs...)
        if nrow(df) > 0 && ncol(df) > 0 && !_looks_like_wrong_delim(df)
            return df
        end
    catch
    end

    try
        first_line = ""
        open(filepath, "r") do io
            while !eof(io)
                l = strip(readline(io))
                if !isempty(l)
                    first_line = l
                    break
                end
            end
        end
        isempty(first_line) && error("Archivo vacío")

        delims = [',', ';', '\t', ' ']
        best_delim = ','
        max_count = -1
        for d in delims
            c = count(==(d), first_line)
            if c > max_count
                max_count = c
                best_delim = d
            end
        end

        df = CSV.read(filepath, DataFrame; delim=best_delim, silencewarnings=true, ignorerepeated=true, kwargs...)
        if nrow(df) > 0 && ncol(df) > 0 && !_looks_like_wrong_delim(df)
            return df
        end
    catch
    end

    try
        mat, header = readdlm(filepath, ',', Float64; header=true)
        col_names = [String(strip(String(h))) for h in header]
        return DataFrame(mat, col_names)
    catch
    end

    return parse_custom_format(filepath)
end

function _normalize_target_name(s::AbstractString)
    return lowercase(replace(strip(String(s)), r"\s+" => ""))
end

function get_target_column(df::DataFrame, target_col::Union{String,Int,Nothing})
    if target_col === nothing
        return ncol(df)
    elseif target_col isa Int
        idx = target_col
        (1 <= idx <= ncol(df)) || error("Índice de variable objetivo fuera de rango: $idx (ncol=$(ncol(df)))")
        return idx
    else
        wanted = _normalize_target_name(String(target_col))
        for (i, c) in enumerate(names(df))
            if _normalize_target_name(String(c)) == wanted
                return i
            end
        end
        error("La columna objetivo '$target_col' no existe. Columnas disponibles: $(names(df))")
    end
end

function _coerce_numeric_column!(df::DataFrame, colname::AbstractString)
    (colname in names(df)) || return

    col = df[!, colname]
    T = _nonmissingtype(eltype(col))
    if T <: Real
        return
    end

    newcol = Vector{Union{Missing,Float64}}(undef, length(col))
    for i in eachindex(col)
        vv = _clean_string_value(col[i])
        if vv === missing
            newcol[i] = missing
            continue
        end
        s = String(vv)
        try
            newcol[i] = parse(Float64, s)
            continue
        catch
        end
        try
            newcol[i] = parse(Float64, replace(s, ',' => '.'))
            continue
        catch
            newcol[i] = missing
        end
    end

    df[!, colname] = newcol
end

function _count_feature_types(df::DataFrame, target_col_idx::Int)
    numeric = 0
    categorical = 0
    temporal = 0
    for (i, c) in enumerate(names(df))
        i == target_col_idx && continue
        lc = lowercase(String(c))
        T = _nonmissingtype(eltype(df[!, c]))
        if c == "dteday" || occursin("date", lc) || occursin("time", lc)
            temporal += 1
        elseif T <: Real
            numeric += 1
        else
            categorical += 1
        end
    end
    return (numeric=numeric, categorical=categorical, temporal=temporal)
end

function _replace_nonfinite_with_missing!(df::DataFrame)
    for c in names(df)
        T = _nonmissingtype(eltype(df[!, c]))
        if T <: Real
            col = df[!, c]
            for i in eachindex(col)
                v = col[i]
                if !ismissing(v) && !isfinite(Float64(v))
                    col[i] = missing
                end
            end
        end
    end
    return df
end

function _impute_numeric_with_train_means!(train_df::DataFrame, test_df::DataFrame)
    for c in names(train_df)
        c in names(test_df) || continue
        T = _nonmissingtype(eltype(train_df[!, c]))
        if T <: Real
            vals = Float64[]
            for v in train_df[!, c]
                if !ismissing(v)
                    push!(vals, Float64(v))
                end
            end
            μ = isempty(vals) ? 0.0 : mean(vals)
            for df in (train_df, test_df)
                col = df[!, c]
                for i in eachindex(col)
                    if ismissing(col[i])
                        col[i] = μ
                    end
                end
            end
        end
    end
    return train_df, test_df
end

function _split_chronological(df::DataFrame, train_ratio::Float64)
    n = nrow(df)
    n >= 2 || error("No hay suficientes filas para hacer split cronológico")
    ntrain = max(1, floor(Int, train_ratio * n))
    ntrain = min(ntrain, n - 1)
    return df[1:ntrain, :], df[(ntrain + 1):n, :]
end

function _read_unsw_raw_dir(dirpath::AbstractString; verbose::Bool=true)::DataFrame
    parts = [
        joinpath(dirpath, "UNSW-NB15_1.csv"),
        joinpath(dirpath, "UNSW-NB15_2.csv"),
        joinpath(dirpath, "UNSW-NB15_3.csv"),
        joinpath(dirpath, "UNSW-NB15_4.csv"),
    ]
    all(isfile.(parts)) || error("No están los 4 CSV raw de UNSW-NB15 en $dirpath")

    dfs = DataFrame[]
    for p in parts
        verbose && println("Leyendo parte raw: $p")
        df = CSV.read(p, DataFrame; header=false, normalizenames=false, silencewarnings=true, ignorerepeated=true)
        ncol(df) == length(UNSW_RAW_COLUMNS) || error("Número de columnas inesperado en $p: $(ncol(df))")
        rename!(df, Symbol.(UNSW_RAW_COLUMNS))
        push!(dfs, df)
    end
    df = vcat(dfs...)
    if "attack_cat" in names(df)
        col = df[!, "attack_cat"]
        for i in eachindex(col)
            v = _clean_string_value(col[i])
            col[i] = (v === missing) ? "Normal" : String(v)
        end
    end
    return df
end

function _maybe_load_unsw(dataset_path::AbstractString; verbose::Bool=true)
    path = String(dataset_path)
    if isdir(path) && isfile(joinpath(path, "UNSW-NB15_1.csv"))
        return _read_unsw_raw_dir(path; verbose=verbose)
    end
    return nothing
end

function _prepare_target(train_df::DataFrame, test_df::DataFrame, target_col_idx::Int, task_type::Symbol)
    target_name = String(names(train_df)[target_col_idx])

    if task_type == :regression
        _coerce_numeric_column!(train_df, target_name)
        _coerce_numeric_column!(test_df, target_name)
        ytr = Float64[]
        yte = Float64[]
        for v in train_df[!, target_name]
            ismissing(v) && error("Target con missing en train: $target_name")
            push!(ytr, Float64(v))
        end
        for v in test_df[!, target_name]
            ismissing(v) && error("Target con missing en test: $target_name")
            push!(yte, Float64(v))
        end
        return ytr, yte, Dict("target_name" => target_name)
    elseif task_type == :binary_classification
        ytr = Float64[]
        yte = Float64[]
        positive = Set(["1", "true", "yes", "attack", "anomaly", "positive", "up", "malicious"])
        negative = Set(["0", "false", "no", "normal", "negative", "down", "benign"])
        for (vec, out) in ((train_df[!, target_name], ytr), (test_df[!, target_name], yte))
            for v in vec
                vv = _clean_string_value(v)
                vv === missing && error("Target con missing en tarea binaria: $target_name")
                lv = lowercase(String(vv))
                if lv in positive
                    push!(out, 1.0)
                elseif lv in negative
                    push!(out, 0.0)
                else
                    try
                        x = parse(Float64, String(vv))
                        push!(out, x > 0 ? 1.0 : 0.0)
                    catch
                        error("No se pudo interpretar el target binario '$vv' en la columna $target_name")
                    end
                end
            end
        end
        return ytr, yte, Dict("target_name" => target_name, "class_names" => ["0", "1"])
    elseif task_type == :multiclass_classification
        ytr = String[]
        yte = String[]
        for v in train_df[!, target_name]
            vv = _clean_string_value(v)
            vv === missing && error("Target multiclase con missing en train: $target_name")
            push!(ytr, String(vv))
        end
        for v in test_df[!, target_name]
            vv = _clean_string_value(v)
            vv === missing && error("Target multiclase con missing en test: $target_name")
            push!(yte, String(vv))
        end
        classes = sort!(collect(unique(vcat(ytr, yte))))
        return ytr, yte, Dict("target_name" => target_name, "class_names" => classes)
    else
        error("task_type no soportado: $task_type")
    end
end

function _feature_matrices(train_df::DataFrame, test_df::DataFrame, target_col_idx::Int;
    categorical_encoding::String="error", dataset_name::AbstractString="")

    train_cols = names(train_df)
    test_cols = names(test_df)
    train_cols == test_cols || error("Train y test no tienen las mismas columnas")

    force_categorical = _force_categorical_columns(dataset_name)
    numeric_cols = String[]
    categorical_cols = String[]
    temporal_cols = String[]

    for (i, c) in enumerate(train_cols)
        i == target_col_idx && continue
        cname = String(c)
        lc = lowercase(cname)
        is_forced_categorical = cname in force_categorical

        if !is_forced_categorical
            _coerce_numeric_column!(train_df, cname)
            _coerce_numeric_column!(test_df, cname)
        end

        T = _nonmissingtype(eltype(train_df[!, cname]))
        if cname == "dteday" || occursin("date", lc) || occursin("time", lc)
            push!(temporal_cols, cname)
        end

        if !is_forced_categorical && (T <: Real)
            push!(numeric_cols, cname)
        else
            push!(categorical_cols, cname)
        end
    end

    _replace_nonfinite_with_missing!(train_df)
    _replace_nonfinite_with_missing!(test_df)

    enc = lowercase(strip(categorical_encoding))
    if enc in ("error", "strict", "none")
        isempty(categorical_cols) || error("Quedan atributos categóricos y no se ha pedido codificación explícita: $(categorical_cols)")
    elseif enc != "onehot"
        error("categorical_encoding no soportado: $categorical_encoding")
    end

    _impute_numeric_with_train_means!(train_df, test_df)

    ntr = nrow(train_df)
    nte = nrow(test_df)
    total_dim = length(numeric_cols)
    levels_map = Dict{String,Vector{String}}()
    if enc == "onehot"
        for c in categorical_cols
            lvls = String[]
            seen = Set{String}()
            for v in train_df[!, c]
                vv = _clean_string_value(v)
                vv === missing && continue
                s = String(vv)
                if !(s in seen)
                    push!(lvls, s)
                    push!(seen, s)
                end
            end
            push!(lvls, "__UNK__")
            levels_map[c] = lvls
            total_dim += length(lvls)
        end
    end

    Xtr = Matrix{Float64}(undef, ntr, total_dim)
    Xte = Matrix{Float64}(undef, nte, total_dim)
    colptr = 1
    for c in numeric_cols
        trcol = train_df[!, c]
        tecol = test_df[!, c]
        @inbounds for i in 1:ntr
            Xtr[i, colptr] = Float64(trcol[i])
        end
        @inbounds for i in 1:nte
            Xte[i, colptr] = Float64(tecol[i])
        end
        colptr += 1
    end

    if enc == "onehot"
        for c in categorical_cols
            lvls = levels_map[c]
            lvl_to_pos = Dict{String,Int}(lvl => j for (j, lvl) in enumerate(lvls))
            startcol = colptr
            endcol = colptr + length(lvls) - 1
            Xtr[:, startcol:endcol] .= 0.0
            Xte[:, startcol:endcol] .= 0.0
            unkpos = lvl_to_pos["__UNK__"]
            trcol = train_df[!, c]
            tecol = test_df[!, c]
            for i in 1:ntr
                vv = _clean_string_value(trcol[i])
                pos = (vv === missing) ? unkpos : get(lvl_to_pos, String(vv), unkpos)
                Xtr[i, startcol + pos - 1] = 1.0
            end
            for i in 1:nte
                vv = _clean_string_value(tecol[i])
                pos = (vv === missing) ? unkpos : get(lvl_to_pos, String(vv), unkpos)
                Xte[i, startcol + pos - 1] = 1.0
            end
            colptr = endcol + 1
        end
    end

    return Xtr, Xte, Dict(
        "numeric_columns" => numeric_cols,
        "categorical_columns" => categorical_cols,
        "temporal_columns" => temporal_cols,
        "categorical_encoding" => categorical_encoding,
        "total_features_after_encoding" => total_dim,
    )
end

function _booleanize_inputs(Xtr::Matrix{Float64}, Xte::Matrix{Float64}, method::String)
    m = lowercase(strip(method))
    if m in ("none", "no", "false")
        return Xtr, Xte, Dict("input_booleanization" => "none")
    elseif m in ("median_threshold", "median", "threshold")
        thr = vec(mapslices(median, Xtr; dims=1))
        Xtr_b = Float64.(Xtr .> reshape(thr, 1, :))
        Xte_b = Float64.(Xte .> reshape(thr, 1, :))
        return Xtr_b, Xte_b, Dict("input_booleanization" => "median_threshold", "thresholds" => thr)
    else
        error("input_booleanization no soportado: $method")
    end
end

function build_xy(X::Matrix{Float64}, y::AbstractVector, window::Int, horizon::Int)
    n = size(X, 1)
    nfeat = size(X, 2)
    nsamples = n - window - horizon + 1
    nsamples <= 0 && error("No hay suficientes filas ($n) para window=$window y horizon=$horizon")

    Xw = Matrix{Float64}(undef, nsamples, window * nfeat)
    yw = Vector{eltype(y)}(undef, nsamples)
    @inbounds for i in 1:nsamples
        win = X[i:(i + window - 1), :]
        Xw[i, :] = vec(win')
        yw[i] = y[i + window + horizon - 1]
    end
    return Xw, yw
end

function preprocess_dataset!(train_df::DataFrame, test_df::DataFrame, dataset_name::String; verbose::Bool=true)
    ds = _normalize_dataset_id(dataset_name)

    if occursin("hour", ds)
        drop_cols = String[]
        for c in ("instant", "dteday", "casual", "registered")
            if c in names(train_df) && c in names(test_df)
                push!(drop_cols, c)
            end
        end
        if !isempty(drop_cols)
            select!(train_df, Not(drop_cols))
            select!(test_df, Not(drop_cols))
        end
    end

    for c in names(train_df)
        _coerce_numeric_column!(train_df, String(c))
        c in names(test_df) && _coerce_numeric_column!(test_df, String(c))
    end

    remaining_non_numeric = String[]
    for c in names(train_df)
        T = _nonmissingtype(eltype(train_df[!, c]))
        if !(T <: Real)
            push!(remaining_non_numeric, String(c))
        end
    end
    isempty(remaining_non_numeric) || error(
        "Quedan columnas no numéricas sin transformar: $(remaining_non_numeric). Revísalas explícitamente antes de lanzar los experimentos."
    )

    _replace_nonfinite_with_missing!(train_df)
    _replace_nonfinite_with_missing!(test_df)
    _impute_numeric_with_train_means!(train_df, test_df)

    return train_df, test_df
end


function load_dataset_tabular(dataset_name::String; data_dir::String=joinpath(@__DIR__, "data"),
    train_ratio::Float64=0.75, target_col::Union{String,Int,Nothing}=nothing,
    task_type::Symbol=:regression, categorical_encoding::String="error",
    input_booleanization::String="none", verbose::Bool=true)

    task_type == :regression || error("load_dataset_tabular solo soporta regresión")
    lowercase(strip(categorical_encoding)) in ("error", "none", "strict") || error("categorical_encoding no soportado en modo tabular: $categorical_encoding")
    lowercase(strip(input_booleanization)) in ("none", "no", "false") || error("input_booleanization no soportado en modo tabular: $input_booleanization")

    verbose && begin
        println("\\n" * "="^70)
        println("CARGANDO DATASET TABULAR: $dataset_name")
        println("="^70)
    end

    base_path = (isfile(dataset_name) || isdir(dataset_name)) ? dataset_name : joinpath(data_dir, dataset_name)

    train_df = nothing
    test_df = nothing

    unsw_df = _maybe_load_unsw(base_path; verbose=verbose)
    if unsw_df !== nothing
        train_df, test_df = _split_chronological(unsw_df, train_ratio)
        verbose && println("Formato: UNSW raw único concatenado | split cronológico: train=$(nrow(train_df)) test=$(nrow(test_df))")
    else
        train_file = nothing
        test_file = nothing
        single_file = nothing

        base_dir = isdir(base_path) ? base_path : dirname(base_path)
        base_name = basename(base_path)
        base_lower = lowercase(base_name)
        if !isdir(base_dir)
            base_dir = data_dir
        end

        function choose_best(cands::Vector{String}, base_lower::String)
            isempty(cands) && return nothing
            pref = [c for c in cands if occursin(base_lower, lowercase(basename(c)))]
            !isempty(pref) && return pref[1]
            length(cands) == 1 && return cands[1]
            sort!(cands, by = x -> length(basename(x)))
            return cands[1]
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
            train_file = choose_best(train_candidates, base_lower)
            test_file = choose_best(test_candidates, base_lower)
            single_file = choose_best(single_candidates, base_lower)
        elseif isfile(base_path)
            single_file = base_path
        end

        if train_file !== nothing && test_file !== nothing
            verbose && begin
                println("Formato: archivos TRAIN/TEST")
                println("  Train: $train_file")
                println("  Test:  $test_file")
            end
            train_df = safe_read_file(train_file; verbose=verbose)
            test_df = safe_read_file(test_file; verbose=verbose)
        elseif single_file !== nothing
            verbose && begin
                println("Formato: archivo único")
                println("  Archivo: $single_file")
            end
            df = safe_read_file(single_file; verbose=verbose)
            train_df, test_df = _split_chronological(df, train_ratio)
            verbose && println("  Split cronológico: train=$(nrow(train_df))  test=$(nrow(test_df))")
        else
            error("No se encontró el dataset: $dataset_name en $data_dir")
        end
    end

    target_col_idx = get_target_column(train_df, target_col)
    preprocess_dataset!(train_df, test_df, dataset_name; verbose=verbose)
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

    function df_to_mat(df::DataFrame, cols::Vector{Int})
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

    train_mat = df_to_mat(train_df, numeric_cols)
    test_mat  = df_to_mat(test_df, numeric_cols)

    feature_idx = [j for j in 1:size(train_mat, 2) if j != target_adj]
    isempty(feature_idx) && error("No quedan features tras quitar la variable objetivo")

    Xtr = train_mat[:, feature_idx]
    ytr = vec(Float64.(train_mat[:, target_adj]))
    Xte = test_mat[:, feature_idx]
    yte = vec(Float64.(test_mat[:, target_adj]))

    dsmeta = Dict(
        "target_name" => String(names(train_df)[target_col_idx]),
        "class_names" => String[],
        "summary_before_encoding" => summary,
        "categorical_encoding" => "none",
        "input_booleanization" => "none",
        "total_features_after_encoding" => size(Xtr, 2),
        "tabular_mode" => true,
    )

    return Xtr, ytr, Xte, yte, dsmeta
end


function load_dataset(dataset_name::String; data_dir::String=joinpath(@__DIR__, "data"), window::Int, horizon::Int=1,
    train_ratio::Float64=0.75, target_col::Union{String,Int,Nothing}=nothing, univariate::Bool=false,
    task_type::Symbol=:regression, categorical_encoding::String="error", input_booleanization::String="none", verbose::Bool=true)

    verbose && begin
        println("\n" * "="^70)
        println("CARGANDO DATASET: $dataset_name")
        println("="^70)
    end

    base_path = (isfile(dataset_name) || isdir(dataset_name)) ? dataset_name : joinpath(data_dir, dataset_name)

    train_df = nothing
    test_df = nothing

    unsw_df = _maybe_load_unsw(base_path; verbose=verbose)
    if unsw_df !== nothing
        train_df, test_df = _split_chronological(unsw_df, train_ratio)
        verbose && println("Formato: UNSW raw único concatenado | split cronológico: train=$(nrow(train_df)) test=$(nrow(test_df))")
    else
        train_file = nothing
        test_file = nothing
        single_file = nothing

        base_dir = isdir(base_path) ? base_path : dirname(base_path)
        base_name = basename(base_path)
        base_lower = lowercase(base_name)
        if !isdir(base_dir)
            base_dir = data_dir
        end

        function choose_best(cands::Vector{String}, base_lower::String)
            isempty(cands) && return nothing
            pref = [c for c in cands if occursin(base_lower, lowercase(basename(c)))]
            !isempty(pref) && return pref[1]
            length(cands) == 1 && return cands[1]
            sort!(cands, by = x -> length(basename(x)))
            return cands[1]
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
            train_file = choose_best(train_candidates, base_lower)
            test_file = choose_best(test_candidates, base_lower)
            single_file = choose_best(single_candidates, base_lower)
        elseif isfile(base_path)
            single_file = base_path
        end

        if train_file !== nothing && test_file !== nothing
            verbose && begin
                println("Formato: archivos TRAIN/TEST")
                println("  Train: $train_file")
                println("  Test:  $test_file")
            end
            train_df = safe_read_file(train_file; verbose=verbose)
            test_df = safe_read_file(test_file; verbose=verbose)
        elseif single_file !== nothing
            verbose && begin
                println("Formato: archivo único")
                println("  Archivo: $single_file")
            end
            df = safe_read_file(single_file; verbose=verbose)
            train_df, test_df = _split_chronological(df, train_ratio)
            verbose && println("  Split cronológico: train=$(nrow(train_df))  test=$(nrow(test_df))")
        else
            error("No se encontró el dataset: $dataset_name en $data_dir")
        end
    end

    target_col_idx = get_target_column(train_df, target_col)

    if task_type == :regression
        preprocess_dataset!(train_df, test_df, dataset_name; verbose=verbose)
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

        function df_to_mat(df::DataFrame, cols::Vector{Int})
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

        train_mat = df_to_mat(train_df, numeric_cols)
        test_mat  = df_to_mat(test_df, numeric_cols)
        feature_idx = univariate ? [target_adj] : [j for j in 1:size(train_mat, 2) if j != target_adj]
        isempty(feature_idx) && error("No quedan features tras quitar la variable objetivo")
        Xtr, ytr = build_xy(train_mat[:, feature_idx], train_mat[:, target_adj], window, horizon)
        Xte, yte = build_xy(test_mat[:, feature_idx], test_mat[:, target_adj], window, horizon)
        return Xtr, ytr, Xte, yte, Dict(
            "target_name" => String(names(train_df)[target_col_idx]),
            "class_names" => String[],
            "summary_before_encoding" => summary,
            "categorical_encoding" => "none",
            "input_booleanization" => "none",
            "total_features_after_encoding" => size(Xtr, 2),
        )
    else
        ytr, yte, target_meta = _prepare_target(train_df, test_df, target_col_idx, task_type)
        summary = _count_feature_types(train_df, target_col_idx)
        verbose && println("  Resumen de atributos -> numéricos=$(summary.numeric), categóricos=$(summary.categorical), temporales=$(summary.temporal)")
        Xtr_base, Xte_base, feat_meta = _feature_matrices(train_df, test_df, target_col_idx;
            categorical_encoding=categorical_encoding, dataset_name=dataset_name)
        Xtr, ytrw = build_xy(Xtr_base, ytr, window, horizon)
        Xte, ytew = build_xy(Xte_base, yte, window, horizon)
        Xtr, Xte, bool_meta = _booleanize_inputs(Xtr, Xte, input_booleanization)
        dsmeta = Dict(
            "target_name" => target_meta["target_name"],
            "class_names" => get(target_meta, "class_names", String[]),
            "summary_before_encoding" => summary,
            "numeric_columns" => feat_meta["numeric_columns"],
            "categorical_columns" => feat_meta["categorical_columns"],
            "temporal_columns" => feat_meta["temporal_columns"],
            "categorical_encoding" => feat_meta["categorical_encoding"],
            "input_booleanization" => bool_meta["input_booleanization"],
            "total_features_after_encoding" => size(Xtr, 2),
        )
        return Xtr, ytrw, Xte, ytew, dsmeta
    end
end
