using CSV
using DataFrames
using Statistics
using DelimitedFiles
using Missings

# -----------------------------
# Utils
# -----------------------------
_nonmissingtype(T) = try
    Base.nonmissingtype(T)
catch
    Missings.nonmissingtype(T)
end

# Heurística: si al parsear sale 1 sola columna y su nombre contiene comas,
# normalmente es que se ha usado un delimitador incorrecto (p.ej. ';' en un CSV con ',').
function _looks_like_wrong_delim(df::DataFrame)::Bool
    return (ncol(df) == 1) && (length(names(df)) == 1) && occursin(",", String(names(df)[1]))
end

"""
Lee un archivo CSV/TXT de forma robusta:
- intenta CSV.read estándar
- intenta ';' con decimal ','
- autodetecta delimitador por la primera línea
- fallback readdlm
- fallback parse custom numérico
"""
function safe_read_file(filepath::String; verbose::Bool=true)::DataFrame
    ext = lowercase(splitext(filepath)[2])

    # 1) CSV estándar
    if ext == ".csv"
        try
            df = CSV.read(filepath, DataFrame; silencewarnings=true, ignorerepeated=true)
            if nrow(df) > 0 && ncol(df) > 0 && !_looks_like_wrong_delim(df)
                return df
            end
        catch
        end
    end

    # 2) CSV estilo europeo
    try
        df = CSV.read(filepath, DataFrame; delim=';', decimal=',', silencewarnings=true, ignorerepeated=true)
        if nrow(df) > 0 && ncol(df) > 0 && !_looks_like_wrong_delim(df)
            return df
        end
    catch
    end

    # 3) Autodetección por primera línea (sin cargar todo el fichero en memoria)
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

        df = CSV.read(filepath, DataFrame; delim=best_delim, silencewarnings=true, ignorerepeated=true)
        if nrow(df) > 0 && ncol(df) > 0 && !_looks_like_wrong_delim(df)
            return df
        else
            error("Autodetección de delimitador fallida")
        end
    catch
    end

    # 4) Fallback readdlm
    try
        mat, header = readdlm(filepath, ',', Float64; header=true)
        col_names = [String(strip(String(h))) for h in header]
        return DataFrame(mat, col_names)
    catch
    end

    # 5) Custom: tokens numéricos por línea
    return parse_custom_format(filepath)
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

"""
Determina la columna objetivo.
- nothing  -> última columna
- Int     -> índice absoluto en df
- String  -> nombre exacto
"""
function get_target_column(df::DataFrame, target_col::Union{String,Int,Nothing})
    if target_col === nothing
        return ncol(df)
    elseif target_col isa Int
        idx = target_col
        (1 <= idx <= ncol(df)) || error("target_col índice fuera de rango: $idx (ncol=$(ncol(df)))")
        return idx
    else
        name = String(target_col)
        (name in names(df)) || error("target_col '$name' no existe. Columnas: $(names(df))")
        return findfirst(==(name), names(df))
    end
end

"""
Crea ventanas temporales (X,y) para train y test.

Robusto a missing:
- Detecta columnas numéricas por nonmissingtype(eltype) <: Real
- Convierte missing -> NaN (la imputación sin leakage se hace después en run_experiment)
"""
function create_windows(
    train_df::DataFrame,
    test_df::DataFrame,
    target_col_idx::Int,
    window::Int,
    horizon::Int,
    univariate::Bool=false;
    verbose::Bool=true
)
    numeric_cols = Int[]
    for i in 1:ncol(train_df)
        T = _nonmissingtype(eltype(train_df[!, i]))
        if T <: Real
            push!(numeric_cols, i)
        end
    end
    isempty(numeric_cols) && error("No hay columnas numéricas en el DataFrame")

    verbose && println("  Columnas numéricas detectadas: $(length(numeric_cols)) / $(ncol(train_df))")

    # Construir matrices Float64 con missing->NaN
    function df_to_mat(df::DataFrame, cols::Vector{Int})
        n = nrow(df); d = length(cols)
        M = Matrix{Float64}(undef, n, d)
        for (j, c) in enumerate(cols)
            col = df[!, c]
            @inbounds for i in 1:n
                v = col[i]
                M[i, j] = ismissing(v) ? NaN : Float64(v)
            end
        end
        return M
    end

    train_mat = df_to_mat(train_df, numeric_cols)
    test_mat  = df_to_mat(test_df, numeric_cols)

    target_adj = findfirst(==(target_col_idx), numeric_cols)
    target_adj === nothing && error("La columna objetivo (índice $target_col_idx) no es numérica")

    feature_idx = if univariate
        verbose && println("  Modo: UNIVARIATE (solo lags del objetivo)")
        [target_adj]
    else
        verbose && println("  Modo: MULTIVARIATE (todas numéricas, incluyendo lags del objetivo)")
        collect(1:size(train_mat, 2))
    end

    Xtr, ytr = build_xy(train_mat, feature_idx, target_adj, window, horizon)
    Xte, yte = build_xy(test_mat, feature_idx, target_adj, window, horizon)

    verbose && begin
        println("  Xtr: $(size(Xtr))  ytr: $(size(ytr))")
        println("  Xte: $(size(Xte))  yte: $(size(yte))")
    end

    return Xtr, ytr, Xte, yte
end

function build_xy(
    data::Matrix{Float64},
    feature_idx::Vector{Int},
    target_idx::Int,
    window::Int,
    horizon::Int
)
    n = size(data, 1)
    nfeat = length(feature_idx)

    nsamples = n - window - horizon + 1
    nsamples <= 0 && error("No hay suficientes filas ($n) para window=$window y horizon=$horizon")

    X = Matrix{Float64}(undef, nsamples, window * nfeat)
    y = Vector{Float64}(undef, nsamples)

    @inbounds for i in 1:nsamples
        win = data[i:(i+window-1), feature_idx]
        X[i, :] = vec(win')  # (nfeat, window) -> vector
        y[i] = data[i + window + horizon - 1, target_idx]
    end

    return X, y
end

"""
Carga datasets en diferentes formatos y crea ventanas temporales.

Retorna: Xtr, ytr, Xte, yte
"""
function load_dataset(
    dataset_name::String;
    data_dir::String="data",
    window::Int,
    horizon::Int=1,
    train_ratio::Float64=0.75,
    target_col::Union{String,Int,Nothing}=nothing,
    univariate::Bool=false,
    verbose::Bool=true
)
    verbose && begin
        println("\n" * "="^70)
        println("CARGANDO DATASET: $dataset_name")
        println("="^70)
    end

    base_path = joinpath(data_dir, dataset_name)

    train_file = nothing
    test_file  = nothing
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
        test_candidates  = String[]
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
        test_file  = choose_best(test_candidates, base_lower)
        single_file = choose_best(single_candidates, base_lower)
    elseif isfile(base_path)
        single_file = base_path
    end

    train_df = nothing
    test_df  = nothing

    if train_file !== nothing && test_file !== nothing
        verbose && begin
            println("Formato: Archivos separados TRAIN/TEST")
            println("  Train: $train_file")
            println("  Test:  $test_file")
            println("\n[1/4] Leyendo archivos...")
        end

        train_df = safe_read_file(train_file; verbose=verbose)
        test_df  = safe_read_file(test_file; verbose=verbose)

    elseif single_file !== nothing
        verbose && begin
            println("Formato: Archivo único")
            println("  Archivo: $single_file")
            println("\n[1/4] Leyendo archivo...")
        end

        df = safe_read_file(single_file; verbose=verbose)

        # split train/test por ratio
        n = nrow(df)
        ntrain = max(1, floor(Int, train_ratio * n))
        ntrain = min(ntrain, n-1)  # asegurar al menos 1 en test

        train_df = df[1:ntrain, :]
        test_df  = df[(ntrain+1):n, :]

        verbose && println("  Split: train=$(nrow(train_df))  test=$(nrow(test_df))")

    else
        error("No se encontró el dataset: $dataset_name en $data_dir")
    end

    verbose && println("\n[2/4] Determinando columna objetivo...")
    target_col_idx = get_target_column(train_df, target_col)

    verbose && println("\n[3/4] Creando ventanas (window=$window, horizon=$horizon)...")
    Xtr, ytr, Xte, yte = create_windows(
        train_df, test_df, target_col_idx, window, horizon, univariate;
        verbose=verbose
    )

    return Xtr, ytr, Xte, yte
end
