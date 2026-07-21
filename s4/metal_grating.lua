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

-- Optional artificial y half-width used by the rectangle representation.
-- Defaults to half the pitch.
local rectangle_y_halfwidth_nm = y_halfwidth_nm or pitch_nm / 2 

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

-- Function to build one simulation at a time
-- basis_sweep then decides how many simulations to run and what number of harmonics to use in each simulation.
local function create_metal_grating_simulation(requested_num_g, polarization)
	-- Creates an empty simulation
	local simulation = S4.NewSimulation()

	-- When SetLattice takes a single argument, it is interpreted as the period of a 1D lattice
	simulation:SetLattice(pitch_nm)
	simulation:SetNumG(requested_num_g)

	simulation:AddMaterial('Ambient', ambient_permittivity)	-- name and permittivity
	simulation:AddMaterial('Planarization', planarization_permittivity)
	simulation:AddMaterial('Substrate', substrate_permittivity)

	-- Layer order follows the direction of incidence, from top to bottom.
	simulation:AddLayer('Ambient',	--name
	 					0,			--thickness
						'Ambient') 	-- material
	simulation:AddLayer('Planarization', planarization_thickness_nm, 'Planarization')
	simulation:AddLayer('Grating', grating_thickness_nm, 'Planarization')

	-- S4 rectangles use half-widths, so 60 nm becomes 30 nm.
	-- The y half-width is ignored for a 1D lattice.
	simulation:SetLayerPatternRectangle('Grating', -- layer to pattern
										'Substrate', -- material of rectangle
										{0, 0}, -- center of rectangle relative to center of unit cell (origin)
										0, -- angle
										{grating_width_nm / 2, rectangle_y_halfwidth_nm	}) --half-widths in x and y
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

-- Solve one simulation and extract the complex reflection coefficient for the 0th order reflected wave
local function solve_and_extract_reflection(requested_num_g, polarization)
	local simulation = create_metal_grating_simulation(requested_num_g, polarization)

	-- Returns the 1-based Lua index of the (0,0) diffraction order.
	local zero_order = simulation:GetDiffractionOrder(0, 0)

	-- Get the forward incident amplitude and backward reflected amplitude
	-- in the uniform ambient layer. Requesting these amplitudes causes S4 to
	-- solve the complete layer stack
	local forward_amplitudes, backward_amplitudes = simulation:GetAmplitudes('Ambient', 0)

	-- SetNumG gives an upper bound; GetNumG returns the actual number of harmonics
	-- used in the simulation.
	local actual_num_g = simulation:GetNumG()

	-- S4 stores the two polarisation amplitudes in a single array
	-- with the first half being s-polarization and the second half being p-polarization.
	local amplitude_index = zero_order

	if polarization == 'p' then
		amplitude_index = zero_order + actual_num_g
	end

	local incident_amplitude = forward_amplitudes[amplitude_index]
	local reflected_amplitude = backward_amplitudes[amplitude_index]

	-- Compute the complex reflection coefficient:
	-- ratio of reflected to incident amplitude
	local reflection_coefficient = divide_complex(reflected_amplitude, incident_amplitude)

	return reflection_coefficient, actual_num_g

end

-- Convergence study
local function run_convergence_study()
	print("fmmax_equivalent_terms,s4_num_g,form,"
		.. "r_s_real, r_s_imag, R_s,"
		.. "r_p_real, r_p_imag, R_p"
	)
	-- Loop over the basis sweep, solving for each number of harmonics and extracting the reflection coefficients.
	for _, basis in ipairs(basis_sweep) do
		local r_s, s_num_g =
			solve_and_extract_reflection(basis.s4_num_g,'s')
		local r_p, p_num_g = 
			solve_and_extract_reflection(basis.s4_num_g,'p')
	
	-- Compute the reflectance (magnitude squared of the reflection coefficient)
		local reflectance_s = magnitude_squared(r_s)
		local reflectance_p = magnitude_squared(r_p)

		print(string.format("%d,%d,%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f",
			basis.fmmax_equivalent_terms,
			basis.s4_num_g,
			form,
			r_s[1], r_s[2], reflectance_s,
			r_p[1], r_p[2], reflectance_p
		))
	end
end

run_convergence_study()