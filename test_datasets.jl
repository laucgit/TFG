# test_datasets.jl
include("run_experiment_multi.jl")
using Statistics
using SymDoME

println("="^70)
println("TEST RÁPIDO - VERIFICACIÓN DE DATASETS")
println("="^70)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

function limit_data_size(Xtr, ytr, Xte, yte, max_samples)
    n_train = min(size(Xtr, 1), max_samples)
    n_test  = min(size(Xte, 1), max_samples ÷ 2)
    return Xtr[1:n_train, :], ytr[1:n_train], Xte[1:n_test, :], yte[1:n_test]
end

count_nonfinite(A::AbstractArray{Float64}) = count(v -> !isfinite(v), A)

# Imputación: sustituir NaN/Inf por la media de la columna (o 0 si no hay finitos)
function impute_nonfinite!(X::Matrix{Float64})
    n, d = size(X)
    for j in 1:d
        s = 0.0; c = 0
        @inbounds for i in 1:n
            v = X[i, j]
            if isfinite(v)
                s += v; c += 1
            end
        end
        μ = (c > 0) ? (s / c) : 0.0
        @inbounds for i in 1:n
            v = X[i, j]
            if !isfinite(v)
                X[i, j] = μ
            end
        end
    end
    return X
end

function impute_nonfinite!(y::Vector{Float64})
    s = 0.0; c = 0
    @inbounds for v in y
        if isfinite(v)
            s += v; c += 1
        end
    end
    μ = (c > 0) ? (s / c) : 0.0
    @inbounds for i in eachindex(y)
        if !isfinite(y[i])
            y[i] = μ
        end
    end
    return y
end

# ------------------------------------------------------------
# Configuración de test
# ------------------------------------------------------------

test_configs = [
    Dict("name" => "ElectricDevices/LD2011_2014_mini.txt",
         "target_col" => "MT_196",
         "window" => 12,
         "max_nodes" => 10,
         "max_samples" => 200),

    Dict("name" => "CinCECGTorso",
         "target_col" => nothing,
         "window" => 12,
         "max_nodes" => 10,
         "max_samples" => 200),

    Dict("name" => "CAMBIAR!!",
         "target_col" => nothing,     # importante: no fuerces índice aquí
         "window" => 12,
         "max_nodes" => 10,
         "max_samples" => 200)
]

# ------------------------------------------------------------
# Ejecutar tests
# ------------------------------------------------------------

for (i, config) in enumerate(test_configs)
    println("\n" * "="^70)
    println("TEST $i/$(length(test_configs)): $(config["name"])")
    println("="^70)

    try
        Xtr, ytr, Xte, yte = load_dataset(
            config["name"],
            data_dir="data",
            window=config["window"],
            horizon=1,
            target_col=config["target_col"],
            train_ratio=0.75
        )

        Xtr, ytr, Xte, yte = limit_data_size(Xtr, ytr, Xte, yte, config["max_samples"])
        println("\n[LIMITADO] Xtr=$(size(Xtr)), ytr=$(size(ytr)), Xte=$(size(Xte)), yte=$(size(yte))")

        nf_before = (
            count_nonfinite(Xtr),
            count_nonfinite(ytr),
            count_nonfinite(Xte),
            count_nonfinite(yte)
        )
        if sum(nf_before) > 0
            println("  [WARN] No finitos antes -> Xtr=$(nf_before[1]) ytr=$(nf_before[2]) Xte=$(nf_before[3]) yte=$(nf_before[4])")
            println("  [INFO] Imputando (en vez de dropear filas)...")
        end

        impute_nonfinite!(Xtr); impute_nonfinite!(ytr)
        impute_nonfinite!(Xte); impute_nonfinite!(yte)

        nf_after = (
            count_nonfinite(Xtr),
            count_nonfinite(ytr),
            count_nonfinite(Xte),
            count_nonfinite(yte)
        )
        println("  [OK] No finitos después -> Xtr=$(nf_after[1]) ytr=$(nf_after[2]) Xte=$(nf_after[3]) yte=$(nf_after[4])")

        if size(Xtr, 1) < 5
            error("Muy pocas muestras (Xtr tiene $(size(Xtr,1)) filas).")
        end

        # Entrenar DoME
        params = DoMEParams(config["max_nodes"], :selective, 1e-6)

        println("\n[ENTRENANDO] DoME con max_nodes=$(config["max_nodes"])...")
        tree, history = dome(
            Xtr, ytr,
            maximumNodes=params.max_nodes,
            minimumReductionMSE=params.min_improvement
        )

        # Evaluar en test
        ŷ_test = [SymDoME.evaluateTree(tree, Xte[j, :]) for j in 1:size(Xte, 1)]
        mse_test = mean((ŷ_test .- yte).^2)

        println("\n[RESULTADO]")
        println("  ✓ MSE train (último): $(round(history[end], digits=6))")
        println("  ✓ MSE test: $(round(mse_test, digits=6))")
        println("  ✓ Iteraciones: $(length(history))")
        println("  ✓ Dataset OK: $(config["name"])")

    catch e
        println("\n[ERROR] Falló el test para $(config["name"])")
        println("  Mensaje: $e")
        showerror(stdout, e, catch_backtrace())
        println()
    end
end

println("\n" * "="^70)
println("TESTS COMPLETADOS")
println("="^70)