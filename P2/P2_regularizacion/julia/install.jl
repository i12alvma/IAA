# Instalación simple de dependencias (Julia)
import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
Pkg.add("DataFrames")
Pkg.add("CSV")