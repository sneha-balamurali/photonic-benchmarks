# Metal Grating Documentation

## What this Section Covers:

This script reproduces the FMMax metal grating benchmark using the Stanford Stratified Structure Solver ($S^4$).

The objective is to compare convergence and runtime behaviour between FMMax and $S^4$ using equivalent geometries and parameters. In this example, we will be simulating a 1D grating.  

## Physical Structure

![metal_grating_diagram](/photonic-benchmarks/images/s4/metal_grating.svg)

**Figure 1:** Schematic of the metal grating benchmark geometry from FMMax reproduced in S4. The structure consists of a semi-infinite air superstrate, a 20nm planarization layer and an 80nm thick patterened layer containing 60nm wide metal stripes embedded in the planarization material, and a semi-infinite metal substrate. The grating period is 180nm. *Diagram not to scale*

## Simulation Parameters

|Parameter|Value|
|--------|----|
|Pitch|180nm|
|Grating Width|60nm|
|Grating Thickness|80nm|
|Planarization Thickness|20nm|
|Wavelength|500nm|
|$\varepsilon_{\mathrm{ambient}}$|1.0+0.0i|
|$\varepsilon_{\mathrm{planarization}}$|2.25+0.0i|
|$\varepsilon_{\mathrm{substrate}}$|-7.632+0.731|