println("="^60)
println("DIAGNÓSTICO DETALLADO - PROBLEMA CON DOME")
println("="^60)

# Simular datos correctos para DoME
using Statistics

# Crear datos sintéticos bien formateados
n_train = 100
n_test = 30
n_features = 24  # window size

# Generar datos sintéticos que tengan una relación clara
X_train = randn(n_train, n_features)
y_train = vec(sum(X_train[:, 1:3], dims=2)) .+ 0.1 * randn(n_train)  # y depende de las 3 primeras features

X_test = randn(n_test, n_features)
y_test = vec(sum(X_test[:, 1:3], dims=2)) .+ 0.1 * randn(n_test)

println("\nDatos sintéticos creados:")
println("  X_train: ", size(X_train))
println("  y_train: ", size(y_train))
println("  X_test:  ", size(X_test))
println("  y_test:  ", size(y_test))

println("\nEstadísticas:")
println("  y_train - Media: ", round(mean(y_train), digits=4), ", Std: ", round(std(y_train), digits=4))
println("  y_test  - Media: ", round(mean(y_test), digits=4), ", Std: ", round(std(y_test), digits=4))

# Intentar ejecutar DoME con estos datos
try
    using SymDoME
    
    println("\n" * "="^60)
    println("EJECUTANDO DOME CON DATOS SINTÉTICOS")
    println("="^60)
    
    tree, history = dome(
        X_train, 
        y_train;
        maximumNodes = 10,
        minimumReductionMSE = 1e-6,
        showText = true
    )
    
    println("\nDOME FUNCIONÓ CORRECTAMENTE!")
    println("\nÁrbol generado:")
    println(tree)
    
    println("\nHistorial de MSE:")
    for (i, mse) in enumerate(history)
        println("  Nodo $i: MSE = $mse")
    end
    
    # Evaluar en test (USAR evaluateTree, no evaluate)
    println("\nEvaluando en test...")
    y_pred = zeros(size(X_test, 1))
    for i in eachindex(y_pred)
        y_pred[i] = SymDoME.evaluateTree(tree, X_test[i, :])
    end
    
    mse_test = mean((y_pred .- y_test).^2)
    
    println("\nMSE en test: ", mse_test)
    
    if mse_test > 0.01
        println("El MSE es razonable")
    end
    
catch e
    println("\nError al ejecutar DoME:")
    println(e)
    println("\nStacktrace:")
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
end