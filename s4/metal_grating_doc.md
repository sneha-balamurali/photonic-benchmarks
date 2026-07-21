# Metal Grating Documentation

## What this Section Covers:

This script reproduces the FMMax metal grating benchmark using the Stanford Stratified Structure Solver ($S^4$).

The objective is to compare convergence behaviour, accuracy and computational performance of $S^4$ and FMMax for the same physical 1D grating. 

The implementation follows the physical structure used by FMMax while adapting it to S4 API and S4's one dimensional RCWA formulation. 

## Overall Program Structure

![overall_program_structure](../images/s4/metal_grating/overall_program_structure.svg)

**Figure 1:** The script first has command line configuration options where you can modify the formulation or Fourier basis sizes without changing the code, defines physical parameters and helper functions, constructs an S4 simulation, solves the electromagnetic problem while extracting the reflection coefficients and runtime, performs the convergence study over the specified Fourier basis sizes and formulation and outputs the benchmark results as CSV data for subsequent analysis and plotting.

## Physical Structure

![metal_grating_diagram](../images/s4/metal_grating/metal_grating.svg)

**Figure 2:** Schematic of the metal grating benchmark geometry from FMMax reproduced in S4. The structure consists of a semi-infinite air superstrate, a 20nm planarization layer and an 80nm thick patterned layer containing 60nm wide metal stripes embedded in the planarization material, and a semi-infinite metal substrate. The grating period is 180nm. *Diagram not to scale*

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
|$\varepsilon_{\mathrm{substrate}}$|-7.632+0.731i|

## Program Sections

### 1. Command Line Configurations

This first section allows parameters to be supplied from the terminal. 

```lua
pcall(loadstring(S4.arg))
form = form or 'fft'
```

Instead of modifying the source code everytime a different formulation or Fourier basis size is required, variables can be passed directly on the command line such as:

```lua
S4 -a 'NG=9; form="fft"' metal_grating.lua
```

`S4.arg` is the string containing the text that is supplied through S4's `-a` command-line option.  So in the example above, `S4.arg` is `'form="fft";NG=0'` `loadstring(S4.arg)` then converts that text into executable Lua code. `pcall(...)` means protected call and it runs the function returned by `loadstring()` and doesn't terminate the script if it fails. This way if the argument is invalid or there is no -a argument provided, the entire script doesn't immediately crash. 


If no formulation is supplied, the FFT formulation is selected as the default. 

### 2. Physical Parameters

The next section defines the physical dimensions of the benchmark. 

### 3. Material Definitions

S4 represents complex permittivity using two-element Lua tables. 

```lua
local substrate_permittivity = {-7.632, 0.731}
```

means $\varepsilon_{\mathrm{substrate}}$=-7.632+0.731. Lua doesn't have a built in complex number datatype like Python so this is the representation you will see in S4.

### 4. Basis Sweep

```lua
local basis_sweep = {
	{fmmax_equivalent_terms = 9,   s4_num_g = 3},
	{fmmax_equivalent_terms = 25,  s4_num_g = 5},
	{fmmax_equivalent_terms = 49,  s4_num_g = 7},
	{fmmax_equivalent_terms = 81,  s4_num_g = 9},
	{fmmax_equivalent_terms = 121, s4_num_g = 11},
	{fmmax_equivalent_terms = 169, s4_num_g = 13},
	{fmmax_equivalent_terms = 225, s4_num_g = 15},
	{fmmax_equivalent_terms = 289, s4_num_g = 17},
	{fmmax_equivalent_terms = 361, s4_num_g = 19},
	{fmmax_equivalent_terms = 441, s4_num_g = 21},
	{fmmax_equivalent_terms = 529, s4_num_g = 23},
	{fmmax_equivalent_terms = 625, s4_num_g = 25},
	{fmmax_equivalent_terms = 729, s4_num_g = 27},
	{fmmax_equivalent_terms = 841, s4_num_g = 29}
}
```

This benchmark looks at convergence with increasing Fourier basis size. This table defines every basis size to be tested. Each entry stores the equivalent basis used in FMMax and the corresponding S4 values. The convergence study simply loops over this table if you don't provide a NG value which would just test one basis size of NG. 

This was done to make it easy to add new basis sizes, remove basis sizes and also compare directly with FMMax. 

### 5. Complex-Number Helper Functions

Lua does not provide a built-in complex number type, so two helper functions are used to perform complex division and calculate a complex magnitude squared. 

### 6. Building a Simulation

`create_metal_grating_simulation()` constructs a complete S4 simulation. This function doesn't solve Maxwell's equations, it simply constructs the object being simulated. 

It performs:
- Creating an empty simulation 
- Defining the lattice
- Fourier basis selection
- Creating the materials used
- Creating the layers 
- Applying grating patterns
- Plane-wave excitation: specifies the direction, polarisation and phase of the incoming plane wave that illuminates the grating
- Formulation selection

### 7. Solving One Simulation

```lua
-- Solve one simulation and extract the complex reflection coefficient for the 0th order reflected wave
local function solve_and_extract_reflection(requested_num_g, polarization)
	local simulation = create_metal_grating_simulation(requested_num_g, polarization)

	-- Returns the 1-based Lua index of the (0,0) diffraction order.
	local zero_order = simulation:GetDiffractionOrder(0, 0)

	local start_cpu_time = os.clock()

	-- Get the forward incident amplitude and backward reflected amplitude
	-- in the uniform ambient layer. Requesting these amplitudes causes S4 to
	-- solve the complete layer stack
	local forward_amplitudes, backward_amplitudes = simulation:GetAmplitudes('Ambient', 0)

	local cpu_seconds = os.clock() - start_cpu_time

	-- SetNumG gives an upper bound; GetNumG returns the actual number of harmonics
	-- used in the simulation.
	local actual_num_g = simulation:GetNumG()


	local amplitude_index = zero_order

	if polarization == 'p' then
		amplitude_index = zero_order + actual_num_g
	end

	local incident_amplitude = forward_amplitudes[amplitude_index]
	local reflected_amplitude = backward_amplitudes[amplitude_index]

	-- Compute the complex reflection coefficient:
	-- ratio of reflected to incident amplitude
	local reflection_coefficient = divide_complex(reflected_amplitude, incident_amplitude)

	return reflection_coefficient, actual_num_g, cpu_seconds

end
```

This function performs
$$
r = \frac{E_{\mathrm{reflected}}}{{E_{\mathrm{incident}}}}
$$

for one polarisation and one Fourier basis size.

It builds the structure, tells S4 to solve Maxwell's equations, extracts the desired quantities and converts them into reflection coefficients. 

#### Build the simulation object:
```lua
local simulation = create_metal_grating_simulation(requested_num_g, polarization)
```

Specifies the lattice, materials, geometry, source, wavelength, numerical formulation. 

#### Diffraction Order:
```lua
-- Returns the 1-based Lua index of the (0,0) diffraction order.
local zero_order = simulation:GetDiffractionOrder(0, 0)
```
A periodic grating can scatter an incident wave into different directions. These outgoing waves are labelled by integers called diffraction orders: 

$$
(m,n) = (0,0), (1,0), (-1,0),...
$$

Since we are dealing with a 1D grating, the periodic variation in the grating only occurs along x, therefore there is only one reciprocal lattice vector in the x direction and the relevant orders are effectively: 

$$
m = 0, +/-1, +/-2,...

n = 0

$$

`simulation.GetDiffractionOrder(0,0)` asks S4 at which position in the amplitude array the (0,0) diffraction order is stored. It returns an array index which we can then use to access the relevant amplitudes. 

Lua arrays are 1-based meaning that the index starts at 1 and not 0, so this also does the indexing for us automatically, which prevents mistakes. 

#### Timing:
```lua
local start_cpu_time = os.clock()
```
Starts the timer

#### Forward and Backward Amplitudes

```lua
-- Get the forward incident amplitude and backward reflected amplitude
-- in the uniform ambient layer. Requesting these amplitudes causes S4 to
-- solve the complete layer stack
local forward_amplitudes, backward_amplitudes = simulation:GetAmplitudes('Ambient', 0)
```

This is an important line. This is what causes S4 to solve the complete multilayer structure. `"Ambient"` only specifies where you want S4 to evaluate and return the modal amplitudes.

In our code, S4 solves the electromagnetic scattering problem across the complete layer stack and then returns the modal amplitudes evaluated in the ambient layer at offset z = 0. 

Where:
- `forward_amplitudes` - amplitudes of modes travelling in the forward +z direction in the ambient, this includes the imposed incident mode
- `backward_amplitudes` - amplitudes of modes travelling in the bacward -z direction, these are the reflected modes. 


