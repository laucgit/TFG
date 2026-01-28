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
        target_col::Union{String,Int,Nothing} = nothing
    )

Carga datasets en diferentes formatos y crea ventanas temporales.

Formatos soportados:
1. Archivo único (.txt, .csv) con división train/test por train_ratio
2. Archivos separados (_TRAIN/_TEST o train/test)
3. CAMBIAR!!

Parámetros:
- dataset_name: Nombre del dataset (ej: "ElectricDevices/LD2011_2014_mini.txt", "CinCECGTorso", "CAMBIAR!!")
- data_dir: Directorio base donde están los datasets
- window: Tamaño de la ventana temporal
- horizon: Pasos adelante para predecir
- train_ratio: Proporción de datos para entrenamiento (solo si es archivo único)
- target_col: Nombre o índice de la columna a predecir

Retorna: Xtr, ytr, Xte, yte
"""
function load_dataset(
    dataset_name::String;
    data_dir::String = "data",
    window::Int,
    horizon::Int = 1,
    train_ratio::Float64 = 0.75,
    target_col::Union{String,Int,Nothing} = nothing
)
    println("\n" * "="^70)
    println("CARGANDO DATASET: $dataset_name")
    println("="^70)

    # Construir ruta base (si ya trae subruta, joinpath lo respeta)
    base_path = joinpath(data_dir, dataset_name)

    # Detección principal:
    # - Si base_path es carpeta: buscar dentro
    # - Si base_path es archivo: usarlo o buscar en su carpeta
    train_file = nothing
    test_file  = nothing
    single_file = nothing

    # Elegir directorio donde buscar
    base_dir = isdir(base_path) ? base_path : dirname(base_path)
    base_name = basename(base_path)
    base_lower = lowercase(base_name)

    # Si el directorio no existe, intentar en data_dir directamente
    if !isdir(base_dir)
        base_dir = data_dir
    end

    # Helper local para elegir mejor candidato
    function choose_best(cands::Vector{String}, base_lower::String)
        if isempty(cands)
            return nothing
        end
        # Preferir los que contengan el nombre base
        pref = [c for c in cands if occursin(base_lower, lowercase(basename(c)))]
        if !isempty(pref)
            return pref[1]
        end
        # Caso especial PEMS_SF: los ficheros suelen llamarse PEMS_train/test
        if base_lower == "pems_sf"
            pref2 = [c for c in cands if occursin("pems", lowercase(basename(c)))]
            if !isempty(pref2)
                return pref2[1]
            end
        end
        # Si solo hay uno, usarlo
        if length(cands) == 1
            return cands[1]
        end
        # Si hay varios, coger el más corto (suele ser el "principal")
        sort!(cands, by = x -> length(basename(x)))
        return cands[1]
    end

    # Buscar archivos reales
    if isdir(base_dir)
        all_files = readdir(base_dir)

        train_candidates = String[]
        test_candidates  = String[]
        single_candidates = String[]

        for file in all_files
            full = joinpath(base_dir, file)
            isfile(full) || continue

            fl = lowercase(file)

            # Ignorar labels siempre (especialmente PEMS)
            if occursin("label", fl)
                continue
            end

            # Train/Test separados
            if occursin("train", fl)
                push!(train_candidates, full)
                continue
            elseif occursin("test", fl)
                push!(test_candidates, full)
                continue
            end

            # Archivos únicos (no train/test)
            push!(single_candidates, full)
        end

        # Elegir mejor train/test (con preferencia por los que contengan base_lower)
        train_file = choose_best(train_candidates, base_lower)
        test_file  = choose_best(test_candidates, base_lower)

        # Si no hay split train/test, elegir archivo único
        if train_file === nothing || test_file === nothing
            train_file = nothing
            test_file = nothing

            # Preferir el que contenga base_lower
            single_pref = [c for c in single_candidates if occursin(base_lower, lowercase(basename(c)))]
            if !isempty(single_pref)
                single_file = single_pref[1]
            else
                # Si dataset_name trae extensión y existe, lo usamos
                if isfile(base_path)
                    single_file = base_path
                else
                    # Si solo hay un archivo "usable" en la carpeta, usarlo
                    usable = [c for c in single_candidates if lowercase(splitext(c)[2]) in [".txt",".csv",".arff",".ts",""]]
                    if length(usable) == 1
                        single_file = usable[1]
                    else
                        single_file = nothing
                    end
                end
            end
        end
    end

    # Si dataset_name ya incluye extensión y existe (por si no entró en el bloque anterior)
    if train_file === nothing && test_file === nothing && single_file === nothing && isfile(base_path)
        single_file = base_path
    end

    # Cargar según el formato encontrado
    if train_file !== nothing && test_file !== nothing
        println("Formato: Archivos train/test separados")
        println("  Train: $train_file")
        println("  Test:  $test_file")
        return load_separate_files(train_file, test_file, window, horizon, target_col)
    elseif single_file !== nothing
        println("Formato: Archivo único")
        println("  Archivo: $single_file")
        return load_single_file(single_file, window, horizon, train_ratio, target_col)
    else
        error("No se encontró el dataset '$dataset_name' en '$data_dir'. Verificar rutas y nombres.")
    end
end

"""
Carga un archivo único y divide en train/test
"""
function load_single_file(
    filepath::String,
    window::Int,
    horizon::Int,
    train_ratio::Float64,
    target_col::Union{String,Int,Nothing}
)
    println("\n[1/4] Leyendo archivo único...")

    # Detectar formato
    ext = lowercase(splitext(filepath)[2])

    if ext == ".txt" || ext == ".csv" || ext == ""
        df = load_txt_csv(filepath)
    elseif ext == ".arff"
        df = load_arff(filepath)
    elseif ext == ".ts"
        df = load_ts(filepath)
    else
        error("Formato no soportado: $ext")
    end

    println("  Dataset: $(size(df))")

    # Determinar columna objetivo
    target_col_idx = get_target_column(df, target_col)
    println("\n[2/4] Columna objetivo: índice $target_col_idx - '$(names(df)[target_col_idx])'")

    # Split train/test
    println("\n[3/4] Separando train/test (ratio=$train_ratio)...")
    n = nrow(df)
    ntrain = max(1, min(n-1, floor(Int, train_ratio * n)))

    train_data = df[1:ntrain, :]
    test_data  = df[(ntrain+1):end, :]

    println("  Train: $(size(train_data))")
    println("  Test:  $(size(test_data))")

    # Crear ventanas
    println("\n[4/4] Creando ventanas temporales...")
    return create_windows(train_data, test_data, target_col_idx, window, horizon)
end

"""
Carga archivos train/test separados
"""
function load_separate_files(
    train_file::String,
    test_file::String,
    window::Int,
    horizon::Int,
    target_col::Union{String,Int,Nothing}
)
    println("\n[1/4] Leyendo archivos separados...")

    # Detectar formato por extensión (si no hay extensión, tratamos como txt)
    ext = lowercase(splitext(train_file)[2])

    if ext == ".txt" || ext == ".csv" || ext == ""
        train_df = load_txt_csv(train_file)
        test_df  = load_txt_csv(test_file)
    elseif ext == ".arff"
        train_df = load_arff(train_file)
        test_df  = load_arff(test_file)
    elseif ext == ".ts"
        train_df = load_ts(train_file)
        test_df  = load_ts(test_file)
    else
        error("Formato no soportado: $ext")
    end

    println("  Train: $(size(train_df))")
    println("  Test:  $(size(test_df))")
    
    # Validar que los datasets no estén vacíos
    if nrow(train_df) == 0 || ncol(train_df) == 0
        error("El archivo de entrenamiento '$train_file' está vacío o no se pudo cargar correctamente")
    end
    if nrow(test_df) == 0 || ncol(test_df) == 0
        error("El archivo de test '$test_file' está vacío o no se pudo cargar correctamente")
    end

    # Determinar columna objetivo
    target_col_idx = get_target_column(train_df, target_col)
    println("\n[2/4] Columna objetivo: índice $target_col_idx - '$(names(train_df)[target_col_idx])'")

    # Crear ventanas
    println("\n[3/4] Creando ventanas temporales...")
    return create_windows(train_df, test_df, target_col_idx, window, horizon)
end

"""
Lee archivos .txt o .csv (si no hay extensión, también entra aquí)
"""
function load_txt_csv(filepath::String)
    # 1) Leer primera línea no vacía para inferir formato
    first_line = ""
    open(filepath, "r") do io
        while !eof(io)
            line = strip(readline(io))
            if !isempty(line)
                first_line = line
                break
            end
        end
    end
    isempty(first_line) && error("Archivo vacío: $filepath")

    # 2) Inferir delimitador
    has_comma = occursin(",", first_line)
    has_semi  = occursin(";", first_line)
    has_tab   = occursin("\t", first_line)
    looks_whitespace = (!has_comma && !has_semi && !has_tab && occursin(" ", first_line))

    # 3) Heurística: si la primera línea parece numérica => probablemente NO hay header
    is_numeric_token(t) = occursin(r"^-?\d+(\.\d+)?([eE]-?\d+)?$", t)
    toks = split(first_line)
    numeric_ratio = isempty(toks) ? 0.0 : sum(is_numeric_token.(toks)) / length(toks)
    no_header = numeric_ratio > 0.9

    # 4) Leer con CSV.jl
    if looks_whitespace
        file = CSV.File(filepath;
                        delim=' ',
                        ignorerepeated=true,
                        header = no_header ? false : 1)
        df = DataFrame(file)
    else
        delim = has_semi ? ';' : has_tab ? '\t' : ','
        file = CSV.File(filepath;
                        delim=delim,
                        header = no_header ? false : 1)
        df = DataFrame(file)
    end

    # 5) Si no había header, poner nombres X1..Xd
    if no_header
        rename!(df, Symbol.("X" .* string.(1:ncol(df))))
    end

    # 6) Si hay timestamp tipo string en primera columna (caso electricity), quitarlo
    if ncol(df) >= 2 && (eltype(df[!, 1]) <: Union{String, Missing})
        df = df[:, 2:end]
    end

    # 7) Convertir a Float64 elemento a elemento (IMPORTANTE: no "rellenar" la columna entera con NaN)
    tofloat(x) = begin
        if x === missing
            return NaN
        end
        if x isa Real
            return Float64(x)
        end
        s = strip(string(x))
        isempty(s) && return NaN
        s = replace(s, "," => ".")
        v = tryparse(Float64, s)
        return v === nothing ? NaN : v
    end

    for col in names(df)
        df[!, col] = tofloat.(df[!, col])
    end

    # Quitar columnas que sean TODO NaN/Inf (suele pasar por una columna vacía extra)
    cols_keep = [any(isfinite, df[!, c]) for c in names(df)]
    df = df[:, cols_keep]


    return df
end


"""
Lee archivos .arff (formato WEKA)
PEMS-SF: Ignora la columna de labels (última columna con valores 1-7)
"""
function load_arff(filepath::String)
    lines = readlines(filepath)

    # Encontrar inicio de datos
    data_start = findfirst(x -> startswith(lowercase(x), "@data"), lines)
    if data_start === nothing
        error("No se encontró @DATA en archivo ARFF")
    end

    # Leer datos
    data_lines = lines[(data_start+1):end]
    data_lines = filter(x -> !isempty(strip(x)), data_lines)

    # Parsear datos
    data_matrix = Vector{Vector{Float64}}()
    for line in data_lines
        values = split(strip(line), ",")
        push!(data_matrix, parse.(Float64, values))
    end

    data_matrix = hcat(data_matrix...)'

    # CASO ESPECIAL PEMS-SF: última columna son labels (1-7), las ignoramos
    if occursin("PEMS", uppercase(filepath))
        println("  Detectado PEMS-SF: ignorando última columna (labels)")
        data_matrix = data_matrix[:, 1:(end-1)]
    end

    n_cols = size(data_matrix, 2)
    col_names = ["X$i" for i in 1:n_cols]
    df = DataFrame(data_matrix, col_names)

    return df
end

"""
Lee archivos .ts (Time Series Classification format)
"""
function load_ts(filepath::String)
    lines = readlines(filepath)

    # Encontrar @data
    data_start = findfirst(x -> strip(lowercase(x)) == "@data", lines)
    if data_start === nothing
        error("No se encontró @data en archivo .ts")
    end

    data_lines = lines[(data_start+1):end]
    data_lines = filter(x -> !isempty(strip(x)) && !startswith(strip(x), "#"), data_lines)

    # Cada línea suele ser: dim1:...,dim2:...:class
    # Aquí hacemos una lectura simple: extraer números de la primera dimensión
    series = Vector{Vector{Float64}}()
    for line in data_lines
        # quitar etiqueta de clase si existe (tras el último ':')
        parts = split(strip(line), ":")
        # nos quedamos con todo menos el último si parece clase, pero esto depende del dataset
        # intento robusto: extraer todos los floats de la línea
        nums = Float64[]
        for token in split(replace(line, ":" => ","), ",")
            t = strip(token)
            isempty(t) && continue
            try
                push!(nums, parse(Float64, t))
            catch
            end
        end
        push!(series, nums)
    end

    # Convertir a matriz (padding con NaN si longitudes distintas)
    maxlen = maximum(length.(series))
    mat = fill(NaN, length(series), maxlen)
    for (i, s) in enumerate(series)
        mat[i, 1:length(s)] .= s
    end

    col_names = ["X$i" for i in 1:size(mat, 2)]
    return DataFrame(mat, col_names)
end

"""
Determina la columna objetivo:
- Si target_col es Int: usa ese índice
- Si target_col es String: busca ese nombre
- Si es nothing: usa la última columna
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

- train_df/test_df: DataFrames completos
- target_col_idx: índice de la columna objetivo (y)
- window: tamaño de ventana
- horizon: pasos adelante a predecir
"""
function create_windows(
    train_df::DataFrame,
    test_df::DataFrame,
    target_col_idx::Int,
    window::Int,
    horizon::Int
)
    # Convertir a matriz
    train_mat = Matrix(train_df)
    test_mat  = Matrix(test_df)

    # Features = todas menos la target
    feature_idx = collect(1:size(train_mat, 2))
    deleteat!(feature_idx, findfirst(==(target_col_idx), feature_idx))

    # Construir (X,y)
    Xtr, ytr = build_xy(train_mat, feature_idx, target_col_idx, window, horizon)
    Xte, yte = build_xy(test_mat, feature_idx, target_col_idx, window, horizon)

    println("  Xtr: $(size(Xtr))  ytr: $(size(ytr))")
    println("  Xte: $(size(Xte))  yte: $(size(yte))")

    return Xtr, ytr, Xte, yte
end

"""
Construye pares (X,y) para una matriz:
- X: (n_samples, window * n_features)
- y: (n_samples,)
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
        win = data[i:(i+window-1), feature_idx]
        X[i, :] = vec(win')  # flatten (features primero)
        y[i] = data[i + window + horizon - 1, target_idx]
    end

    return X, y
end