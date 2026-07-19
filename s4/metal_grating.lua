-- Reproducing the FMMax metal grating benchmark in S4.

-- This script was developed from the S4 binary grating convergence 
-- (examples/convergence/1d/binary_grating.lua) and extensively 
-- modified to reproduce the FMMax metal_grating.py example. 
--
-- Current modifications include:
--   - FMMax layer stack (ambient, planarization, metal grating, substrate).
--   - FMMax material permittivities and physical dimensions.
--   - Reusable simulation construction function.
--   - Basis sweep corresponding to the FMMax convergence study.
--   - Support for multiple S4 formulation options.
--   - Expanded comments and documentation for learning and benchmarking.

-- Additional benchmark output (reflection extraction, timing and CSV output)
-- will be added in subsequent revisions.

-- In a 1D pattern, the pattern should be specified only with rectangle because
-- it is the only shape that can represent a profile that varies in one direction while 
-- remaining constant in the other. 
-- When S4 detects a 1D lattice, it treats the materials as varying only along x
-- and being infinitely extended along y.

-- Lets you pass Lua code on the command line to set variables.
pcall(loadstring(S4.arg))

-- S4.arg can define form and NG on the command line.
-- If form ("ref", "lan", "fft", "sps", "nv", "cp") is supplied on the command line, use that.
-- Otherwise default to 'fft'.
-- Example: S4 'NG=10; form="fft"' metal_grating.lua
form = form or 'fft'

-- Physical parameters. All lengths use nanometers.
local pitch_nm = 180
local grating_width_nm = 60
local grating_thickness_nm = 80
local planarization_thickness_nm = 20
local wavelength_nm = 500

-- Relative permittivities: {real part, imaginary part}.
local ambient_permittivity = {1.0, 0.0}
local planarization_permittivity = {2.25, 0.0}
local substrate_permittivity = {-7.632, 0.731}

-- Each row pairs an FMMax 2D parallelogramic target with the S4 1D
-- basis having the same number of x-directed harmonics.
-- This is for later comparisons against FMMax results.
-- This does not match matrix size, computational work, or circular truncation.
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

-- Supplying NG runs one basis size instead of the complete sweep.
if NG then
	basis_sweep = {{fmmax_equivalent_terms = NG * NG, s4_num_g = NG}}
end

-- Lua represents a complex number as {real part, imaginary part} rather than have a 
-- built in complex type. The following functions are used to divide complex numbers and compute their magnitude squared.

-- Divide two complex numbers, returning the result as a new complex number.
local function divide_complex(numerator, denominator)
	local denominator_squared =
		denominator[1] * denominator[1] + denominator[2] * denominator[2]
	return {
		(numerator[1] * denominator[1]
			+ numerator[2] * denominator[2]) / denominator_squared,
		(numerator[2] * denominator[1]
			- numerator[1] * denominator[2]) / denominator_squared
	}
end

-- Compute the magnitude squared of a complex number, returning a real number.
local function magnitude_squared(complex_number)
	return complex_number[1] * complex_number[1]
		+ complex_number[2] * complex_number[2]
end

local function create_metal_grating_simulation(requested_num_g, polarization)
	local simulation = S4.NewSimulation()

	-- A zero second lattice vector tells S4 that this is a 1D grating.
	simulation:SetLattice({pitch_nm, 0}, {0, 0})
	simulation:SetNumG(requested_num_g)

	simulation:AddMaterial('Ambient', ambient_permittivity)
	simulation:AddMaterial('Planarization', planarization_permittivity)
	simulation:AddMaterial('Substrate', substrate_permittivity)

	-- Layer order follows the direction of incidence, from top to bottom.
	simulation:AddLayer('Ambient', 0, 'Ambient')
	simulation:AddLayer(
		'Planarization', planarization_thickness_nm, 'Planarization')
	simulation:AddLayer(
		'Grating', grating_thickness_nm, 'Planarization')

	-- S4 rectangles use half-widths, so 60 nm becomes 30 nm.
	-- The y half-width is ignored for a 1D lattice.
	simulation:SetLayerPatternRectangle(
		'Grating', 'Substrate', {0, 0}, 0, {grating_width_nm / 2, 0})
	simulation:AddLayer('Substrate', 0, 'Substrate')

	-- At normal incidence, s is TE and p is TM.
	if polarization == 's' then
		simulation:SetExcitationPlanewave({0, 0}, {1, 0}, {0, 0})
	elseif polarization == 'p' then
		simulation:SetExcitationPlanewave({0, 0}, {0, 0}, {1, 0})
	else
		error("unknown polarization: " .. tostring(polarization))
	end

	-- S4 frequency is 1 / wavelength, not angular frequency.
	simulation:SetFrequency(1 / wavelength_nm)

	-- Keep formulation names used by the original S4 convergence example.
	if form == 'ref' then
		simulation:UseDiscretizedEpsilon(false)
	elseif form == 'lan' then
		simulation:UseLanczosSmoothing(true)
	elseif form == 'fft' then
		simulation:UseDiscretizedEpsilon(true)
		simulation:SetResolution(8)
	elseif form == 'sps' then
		simulation:UseDiscretizedEpsilon(true)
		simulation:UseSubpixelSmoothing()
		simulation:SetResolution(8)
	elseif form == 'nv' then
		simulation:UsePolarizationDecomposition()
		simulation:UseNormalVectorBasis()
		simulation:SetResolution(8)
	elseif form == 'cpx' then
		simulation:UsePolarizationDecomposition()
		simulation:UseJonesVectorBasis()
		simulation:SetResolution(8)
	elseif form == 'pol' then
		simulation:UsePolarizationDecomposition()
		simulation:SetResolution(8)
	else
		error("unknown formulation: " .. tostring(form))
	end

	return simulation
end