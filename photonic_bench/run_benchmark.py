"""Load one benchmark, run both solvers and compare"""

from photonic_bench.backends.fmmax_backend import run_fmmax
from photonic_bench.backends.s4_backend import run_s4
from photonic_bench.model.config import load_benchmark_config

# Convert the YAML values into a BenchmarkConfig Python object
benchmark = load_benchmark_config("photonic_bench/configs/square_particle.yaml")

# Both backends recieve exactly the same physical benchmark to run
fmmax_result = run_fmmax(benchmark)
s4_result = run_s4(benchmark)

print(fmmax_result)
print(s4_result)

# The results should be physically close even if not identical
print("TE difference:",
    abs(fmmax_result.reflectance_te - s4_result.reflectance_te),
)

print("TM difference:",
    abs(fmmax_result.reflectance_tm - s4_result.reflectance_tm),
)

if __name__ == "__main__":
    main()