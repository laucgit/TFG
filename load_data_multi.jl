using CSV
using DataFrames
using Statistics
using DelimitedFiles

"""
    load_dataset(
        dataset_name::String;
        data_dir::String = "data",
        window::Int,
        horizon::Int = 1,
        train_ratio::Float64 = 0.75,
        target_col::Union{String,Int,Nothing} = nothing,
        univariate::Bool = false
    )

Carga datasets en diferentes formatos y crea ventanas temporales.

CAMBIO CRÍTICO:
- Por defecto INCLUYE los lags del objetivo en los features (NO los elimina)
- Esto es correcto para forecasting: X tiene t-window...t-1, y predice t+horizon
- NO hay data leakage porque y está en el futuro respecto a X
- Si univariate=true, usa SOLO lags del objetivo (autoregresivo puro)

Formatos soportados:
1. Archivo único (.txt, .csv) con división train/test por train_ratio
2. Archivos separados (_TRAIN/_TEST o train/test)
3. ETT datasets (ETTh2.csv, ETTm1.csv)
4. LCDS datasets (LCD_USW00094789_2024.csv)
5. LD2011_2014_mini.txt (electricity data)

Parámetros:
- dataset_name: Nombre del dataset
- data_dir: Directorio base donde están los datasets
- window: Tamaño de la ventana temporal
- horizon: Pasos adelante para predecir
- train_ratio: Proporción de datos para entrenamiento (solo si es archivo único)
- target_col: Nombre o índice de la columna a predecir
- univariate: Si true, usa SOLO lags del objetivo (autoregresivo)

Retorna: Xtr, ytr, Xte, yte
"""
function load_dataset(
    dataset_name::String;
    data_dir::String = "data",
    window::Int,
    horizon::Int = 1,
    train_ratio::Float64 = 0.75,
    target_col::Union{String,Int,Nothing} = nothing,
    univariate::Bool = false
)
    println("\n" * "="^70)
    println("CARGANDO DATASET: $dataset_name")
    println("="^70)

    # Construir ruta base
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
        if isempty(cands)
            return nothing
        end
        pref = [c for c in cands if occursin(base_lower, lowercase(basename(c)))]
        if !isempty(pref)
            return pref[1]
        end
        if length(cands) == 1
            return cands[1]
        end
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

            if occursin("label", fl)
                continue
            end

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
        println("Formato: Archivos separados TRAIN/TEST")
        println("  Train: $train_file")
        println("  Test:  $test_file")

        println("\n[1/4] Leyendo archivos...")
        train_df = safe_read_file(train_file)
        test_df  = safe_read_file(test_file)

        println("  Train: $(size(train_df))")
        println("  Test:  $(size(test_df))")

    elseif single_file !== nothing
        println("Formato: Archivo único")
        println("  Archivo: $single_file")

        println("\n[1/4] Leyendo archivo único...")
        full_df = safe_read_file(single_file)
        println("  Dataset: $(size(full_df))")

        n = nrow(full_df)
        split_idx = Int(floor(n * train_ratio))

        println("\n[2/4] Separando train/test (ratio=$train_ratio)...")
        train_df = full_df[1:split_idx, :]
        test_df  = full_df[(split_idx+1):end, :]

        println("  Train: $(size(train_df))")
        println("  Test:  $(size(test_df))")
    else
        error("No se encontraron archivos válidos para '$dataset_name' en '$base_dir'")
    end

    # Paso común: definir target
    step_offset = (train_file !== nothing && test_file !== nothing) ? 2 : 3
    target_col_idx = get_target_column(train_df, target_col)
    target_col_name = names(train_df)[target_col_idx]

    println("\n[$step_offset/4] Columna objetivo: índice $target_col_idx - '$target_col_name'")

    # Crear ventanas
    println("\n[$((step_offset+1))/4] Creando ventanas temporales...")
    Xtr, ytr, Xte, yte = create_windows(
        train_df, test_df, target_col_idx, window, horizon, univariate
    )

    return Xtr, ytr, Xte, yte
end

"""
Lee un archivo CSV/TXT de forma robusta.
"""
function safe_read_file(filepath::String)::DataFrame
    ext = lowercase(splitext(filepath)[2])

    if ext == ".csv"
        try
            df = CSV.read(filepath, DataFrame; silencewarnings=true)
            if nrow(df) > 0 && ncol(df) > 0
                return df
            end
        catch
        end
    end

    try
        df = CSV.read(
            filepath,
            DataFrame;
            delim=';',
            silencewarnings=true,
            decimal=','
        )
        if nrow(df) > 0 && ncol(df) > 0
            return df
        end
    catch
    end

    try
        content = read(filepath, String)
        lines = split(content, '\n')
        non_empty = filter(l -> !isempty(strip(l)), lines)

        if isempty(non_empty)
            error("Archivo vacío")
        end

        first_line = strip(non_empty[1])
        delims = [',', ';', '\t', ' ']
        best_delim = ','
        max_count = 0

        for d in delims
            c = count(==(d), first_line)
            if c > max_count
                max_count = c
                best_delim = d
            end
        end

        df = CSV.read(
            filepath,
            DataFrame;
            delim=best_delim,
            silencewarnings=true
        )
        return df
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

"""
Parser custom para formatos especiales.
"""
function parse_custom_format(filepath::String)::DataFrame
    lines = readlines(filepath)
    filter!(l -> !isempty(strip(l)), lines)

    series = Vector{Vector{Float64}}()

    for line in lines
        tokens = split(line, r"[,;\s]+")
        nums = Float64[]
        for token in tokens
            t = strip(token)
            isempty(t) && continue
            try
                push!(nums, parse(Float64, t))
            catch
            end
        end
        push!(series, nums)
    end

    maxlen = maximum(length.(series))
    mat = fill(NaN, length(series), maxlen)
    for (i, s) in enumerate(series)
        mat[i, 1:length(s)] .= s
    end

    col_names = ["X$i" for i in 1:size(mat, 2)]
    return DataFrame(mat, col_names)
end

"""
Determina la columna objetivo.
"""
function get_target_column(df::DataFrame, target_col::Union{String,Int,Nothing})
    if target_col === nothing
        return ncol(df)
    elseif target_col isa Int
        idx = target_col
        if idx < 1 || idx > ncol(df)
            error("target_col índice fuera de rango: $idx (ncol=$(ncol(df)))")
        end
        return idx
    else
        name = String(target_col)
        if !(name in names(df))
            error("target_col '$name' no existe en columnas: $(names(df))")
        end
        return findfirst(==(name), names(df))
    end
end

"""
Crea ventanas temporales (X,y) para train y test.

CORRECCIÓN CRÍTICA:
- Por defecto INCLUYE los lags del objetivo en X (NO los elimina)
- Si univariate=true, usa SOLO lags del objetivo
- NO hay data leakage: X tiene t-window...t-1, y predice t+horizon
"""
function create_windows(
    train_df::DataFrame,
    test_df::DataFrame,
    target_col_idx::Int,
    window::Int,
    horizon::Int,
    univariate::Bool = false
)
    # CORRECCIÓN CRÍTICA: Excluir columnas no numéricas (fechas, strings, etc.)
    numeric_cols = Int[]
    for i in 1:ncol(train_df)
        col_type = eltype(train_df[!, i])
        if col_type <: Real || col_type <: AbstractFloat || col_type <: Integer
            push!(numeric_cols, i)
        end
    end
    
    if isempty(numeric_cols)
        error("No hay columnas numéricas en el DataFrame")
    end
    
    println("  Columnas numéricas detectadas: $(length(numeric_cols)) de $(ncol(train_df))")
    
    # Convertir solo columnas numéricas a Matrix{Float64}
    train_mat = Matrix{Float64}(train_df[:, numeric_cols])
    test_mat  = Matrix{Float64}(test_df[:, numeric_cols])
    
    # Ajustar target_col_idx al nuevo espacio de índices
    target_col_idx_adjusted = findfirst(==(target_col_idx), numeric_cols)
    if target_col_idx_adjusted === nothing
        error("La columna objetivo (índice $target_col_idx) no es numérica")
    end

    # CORRECCIÓN: INCLUIR lags del objetivo
    if univariate
        # Solo autoregresivo (lags del objetivo)
        feature_idx = [target_col_idx_adjusted]
        println("  Modo: UNIVARIATE (solo lags del objetivo)")
    else
        # Todas las columnas numéricas (incluyendo lags del objetivo)
        feature_idx = collect(1:size(train_mat, 2))
        println("  Modo: MULTIVARIATE (todas las columnas numéricas, incluyendo lags del objetivo)")
    end

    Xtr, ytr = build_xy(train_mat, feature_idx, target_col_idx_adjusted, window, horizon)
    Xte, yte = build_xy(test_mat, feature_idx, target_col_idx_adjusted, window, horizon)

    println("  Xtr: $(size(Xtr))  ytr: $(size(ytr))")
    println("  Xte: $(size(Xte))  yte: $(size(yte))")

    return Xtr, ytr, Xte, yte
end

"""
Construye pares (X,y) para una matriz:
- X: (n_samples, window * n_features) con lags t-window...t-1
- y: (n_samples,) con valores en t+horizon
- NO hay data leakage porque y está en el futuro respecto a X
"""
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
    if nsamples <= 0
        error("No hay suficientes filas ($n) para window=$window y horizon=$horizon")
    end

    X = Matrix{Float64}(undef, nsamples, window * nfeat)
    y = Vector{Float64}(undef, nsamples)

    for i in 1:nsamples
        # X: ventana de t-window a t-1
        win = data[i:(i+window-1), feature_idx]
        X[i, :] = vec(win')
        
        # y: valor objetivo en t+horizon-1 (futuro)
        y[i] = data[i + window + horizon - 1, target_idx]
    end

    return X, y
end