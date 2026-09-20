## Running the benchmarks

### FMMax convergence

```bash
cd fmmax
../../fmmax/.venv/bin/python metal_grating_benchmark.py --study convergence
```

### FMMax wavelength sweep
```bash
cd fmmax
../../fmmax/.venv/bin/python metal_grating_benchmark.py --study wavelength
```

### S4 convergence
```bash
cd s4
../../s4/S4/build/S4 \
  -a 'study="convergence"; form="fft"' \
  metal_grating.lua \
  > results/s4_metal_grating_comparison_1.csv
```

### S4 wavelength sweep 
```bash
cd s4
../../s4/S4/build/S4 \
  -a 'study="wavelength"; form="fft"' \
  metal_grating.lua \
  > results/s4_metal_grating_spectrum.csv
```

### Generate comparison plot
```bash
cd Comparison
python3 compare_metal_grating.py
python3 compare_metal_grating_wavelength_sweep.py
```

