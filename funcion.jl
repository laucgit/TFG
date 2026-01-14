using Downloads
using CSV
using DataFrames
using Statistics
using LinearAlgebra
using Random

# ============================================================
#  DATASET LOADING (UCR: .ts OR .tsv)
# ============================================================

function load_ucr_electric_devices(data_dir::String) #Habrá que luego cambiar la función para no hardcodear el dataset

    ts_train = joinpath(data_dir, "ElectricDevices/ElectricDevices_TRAIN.ts")
    ts_test  = joinpath(data_dir, "ElectricDevices/ElectricDevices_TEST.ts")

    tsv_train = joinpath(data_dir, "ElectricDevices/ElectricDevices_TRAIN.tsv")
    tsv_test  = joinpath(data_dir, "ElectricDevices/ElectricDevices_TEST.tsv")

    if isfile(ts_train) && isfile(ts_test)
        println("Detectado formato .ts (UCR/UEA)")
        return load_ucr_ts(ts_train, ts_test)

    elseif isfile(tsv_train) && isfile(tsv_test)
        println("Detectado formato .tsv (UCR antiguo)")
        return load_ucr_tsv(tsv_train, tsv_test)

    else
        error(
            "No se encontraron archivos ElectricDevices en formato .ts ni .tsv en $data_dir"
        )
    end
end


function load_ucr_ts(train_path::String, test_path::String)

    function read_ts_file(path)
        series = Vector{Vector{Float64}}()
        labels = Int[]

        open(path, "r") do io
            in_data = false
            for line in eachline(io)
                line = strip(line)

                isempty(line) && continue

                if startswith(lowercase(line), "@data")
                    in_data = true
                    continue
                end

                (!in_data || startswith(line, "@")) && continue

                parts = split(line, ":"; limit=2)

                values = parse.(Float64, split(parts[1], ","))
                label  = parse(Int, parts[2])

                push!(series, values)
                push!(labels, label)
            end
        end

        n_series = length(series)
        series_length = length(series[1])

        X = Array{Float64}(undef, n_series, series_length)
        for i in 1:n_series
            X[i, :] = series[i]
        end

        return X, labels
    end

    Xtr, ytr = read_ts_file(train_path)
    Xte, yte = read_ts_file(test_path)

    return vcat(Xtr, Xte), vcat(ytr, yte)
end


function load_ucr_tsv(train_path::String, test_path::String)
    df_tr = CSV.read(train_path, DataFrame; delim='\t', header=false)
    df_te = CSV.read(test_path, DataFrame; delim='\t', header=false)

    df = vcat(df_tr, df_te)

    y = df[:, 1]
    X = Matrix(df[:, 2:end])

    return X, y
end

# ============================================================
#  PREPROCESSING
# ============================================================

function zscore_normalize(X::Matrix)
    μ = mean(X, dims=1)
    σ = std(X, dims=1)
    σ[σ .== 0.0] .= 1.0
    return (X .- μ) ./ σ
end

# ============================================================
#  WINDOWING
# ============================================================

function create_windows(
    X::Matrix,
    y::Vector;
    window_size::Int,
    horizon::Int = 1
)
    n_samples, n_features = size(X)

    n_windows = n_samples - window_size - horizon + 1
    @assert n_windows > 0 "Window size demasiado grande"

    Xw = Array{Float64}(undef, n_windows, window_size * n_features)
    yw = Array{eltype(y)}(undef, n_windows)

    for i in 1:n_windows
        window = X[i:i+window_size-1, :]
        Xw[i, :] = vec(window)
        yw[i] = y[i + window_size + horizon - 1]
    end

    return Xw, yw
end

# ============================================================
#  TRAIN / TEST SPLIT
# ============================================================

function temporal_train_test_split(X, y; test_ratio=0.2)
    n = size(X, 1)
    split_idx = Int(floor((1 - test_ratio) * n))

    X_train = X[1:split_idx, :]
    y_train = y[1:split_idx]

    X_test  = X[split_idx+1:end, :]
    y_test  = y[split_idx+1:end]

    return X_train, y_train, X_test, y_test
end

# ============================================================
#  BASELINE MODEL
# ============================================================

function train_linear_baseline(X, y)
    X_aug = hcat(ones(size(X,1)), X)
    w = X_aug \ y
    return w
end

function predict_linear(w, X)
    X_aug = hcat(ones(size(X,1)), X)
    return X_aug * w
end

# ============================================================
#  FULL PIPELINE
# ============================================================

function run_pipeline(;
    data_dir = "data",
    window_size = 20,
    horizon = 1,
    test_ratio = 0.25
)
    println("Cargando dataset...")
    X, y = load_ucr_electric_devices(data_dir)

    println("Normalizando...")
    X = zscore_normalize(X)

    println("Creando ventanas...")
    Xw, yw = create_windows(
        X, y;
        window_size=window_size,
        horizon=horizon
    )

    println("Split train/test...")
    Xtr, ytr, Xte, yte = temporal_train_test_split(
        Xw, yw;
        test_ratio=test_ratio
    )

    println("Entrenando baseline...")
    w = train_linear_baseline(Xtr, ytr)

    println("Evaluando...")
    y_pred = predict_linear(w, Xte)
    mse = mean((y_pred .- yte).^2)

    println("MSE test = ", mse)

    return mse
end
