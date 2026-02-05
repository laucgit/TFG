#!/usr/bin/env julia
"""
Script para convertir archivos .jld2 a formato texto

Uso:
    julia jld2_to_txt.jl archivo.jld2
    julia jld2_to_txt.jl archivo.jld2 salida.txt

Requiere:
    usando Pkg
    Pkg.add("JLD2")
"""

using JLD2

function format_value(value, indent=0)
    """Formatea un valor para visualización"""
    prefix = "  " ^ indent
    
    if isa(value, Dict)
        result = "{\n"
        for (k, v) in value
            result *= "$(prefix)  $k => $(format_value(v, indent+1))\n"
        end
        result *= "$(prefix)}"
        return result
    elseif isa(value, Array)
        if length(value) > 10
            return "Array $(size(value)) con $(length(value)) elementos\n$(prefix)  Primeros: $(value[1:min(10, end)])"
        else
            return string(value)
        end
    elseif isa(value, String)
        return "\"$value\""
    else
        return string(value)
    end
end

function jld2_to_txt(input_file::String, output_file::String="")
    """Convierte un archivo JLD2 a formato texto"""
    
    if !isfile(input_file)
        println("ERROR: El archivo '$input_file' no existe")
        return false
    end
    
    if output_file == ""
        output_file = replace(input_file, ".jld2" => ".txt")
    end
    
    println("Leyendo archivo: $input_file")
    println("Generando salida: $output_file")
    
    try
        # Abrir archivo JLD2
        jldopen(input_file, "r") do file
            open(output_file, "w") do out
                println(out, "=" ^ 80)
                println(out, "CONTENIDO DEL ARCHIVO: $(basename(input_file))")
                println(out, "=" ^ 80)
                println(out)
                
                # Listar todas las claves
                println(out, "CLAVES DEL ARCHIVO:")
                println(out, "-" ^ 80)
                all_keys = keys(file)
                for key in all_keys
                    println(out, "  • $key")
                end
                println(out)
                
                # Explorar cada clave
                println(out, "=" ^ 80)
                println(out, "CONTENIDO DETALLADO:")
                println(out, "=" ^ 80)
                println(out)
                
                for key in all_keys
                    println(out, "\n[$key]")
                    println(out, "-" ^ 40)
                    
                    try
                        value = file[key]
                        tipo = typeof(value)
                        println(out, "Tipo: $tipo")
                        println(out)
                        
                        if isa(value, Dict)
                            println(out, "Diccionario con $(length(value)) elementos:")
                            for (k, v) in value
                                println(out, "  $k:")
                                println(out, "    $(format_value(v, 2))")
                            end
                        elseif isa(value, Array)
                            println(out, "Array de tamaño: $(size(value))")
                            println(out, "Elementos: $(length(value))")
                            if length(value) <= 100
                                println(out, "Valores:")
                                println(out, value)
                            else
                                println(out, "Primeros 100 valores:")
                                println(out, value[1:100])
                                println(out, "...")
                                println(out, "($(length(value) - 100) elementos más)")
                            end
                        else
                            println(out, "Valor:")
                            println(out, format_value(value))
                        end
                        
                    catch e
                        println(out, "ERROR al leer esta clave: $e")
                    end
                    
                    println(out)
                end
                
                # Resumen final
                println(out, "\n" * "=" ^ 80)
                println(out, "RESUMEN:")
                println(out, "=" ^ 80)
                println(out, "Total de claves: $(length(all_keys))")
                println(out, "Archivo procesado: $input_file")
                println(out, "Fecha: $(now())")
            end
        end
        
        println("✓ Conversión completada exitosamente")
        println("✓ Archivo guardado en: $output_file")
        return true
        
    catch e
        println("ERROR al procesar el archivo: $e")
        println(stacktrace(catch_backtrace()))
        return false
    end
end

# Función principal
function main()
    if length(ARGS) < 1
        println("=" ^ 60)
        println("Script para convertir archivos JLD2 a TXT")
        println("=" ^ 60)
        println("\nUso:")
        println("  julia jld2_to_txt.jl archivo.jld2")
        println("  julia jld2_to_txt.jl archivo.jld2 salida.txt")
        println("\nEjemplo:")
        println("  julia jld2_to_txt.jl datos.jld2")
        println("\nRequiere el paquete JLD2:")
        println("  using Pkg")
        println("  Pkg.add(\"JLD2\")")
        exit(1)
    end
    
    input_file = ARGS[1]
    output_file = length(ARGS) >= 2 ? ARGS[2] : ""
    
    success = jld2_to_txt(input_file, output_file)
    exit(success ? 0 : 1)
end

# Ejecutar si es el script principal
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end