module Practica1IAA

# Importar dependencias
using DataFrames
using CSV
using Statistics
using Random
using Plots

# Exportar funciones si las tienes
export load_data, evaluate_model

# Función de ejemplo
function load_data(filepath)
    return CSV.read(filepath, DataFrame)
end

function evaluate_model(y_true, y_pred)
    mse = mean((y_true .- y_pred).^2)
    rmse = sqrt(mse)
    return rmse
end

end # module
