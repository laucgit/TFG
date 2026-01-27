using CSV
using DataFrames

# Cargar el archivo original
original_file = "data/ElectricDevices/LD2011_2014.txt"
data = CSV.File(original_file, delim=";", quotechar='"')

# Crear un mini dataset tomando las primeras 100 filas
mini_data = DataFrame(data[1:100, :])

# Guardar el mini dataset en un nuevo archivo CSV
mini_file = "mini_LD2011_2014.csv"
CSV.write(mini_file, mini_data)
