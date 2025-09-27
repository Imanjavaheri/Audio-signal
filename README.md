# Audio Signal Direction-of-Arrival Estimation

This repository evaluates direction-of-arrival (DoA) estimation for multiple acoustic sources captured by a linear microphone array. The MATLAB implementation compares a classical Delay-and-Sum (DAS) beamformer with the subspace-based MUSIC algorithm to illustrate their relative resolution capabilities and the effects of array spacing.

## Repository Contents
- `Audio_Project.m` – MATLAB script that synthesizes array data, applies DAS and MUSIC beamformers, and visualizes the spatial spectra with the corresponding angle estimates.
- `Report.pdf` – Technical report summarizing the array design, methodology, and comparative findings for different source separations and microphone spacings.

## Prerequisites
- MATLAB with base plotting and signal-processing capabilities.

## Quick Start
1. Open `Audio_Project.m` in MATLAB and adjust the scenario parameters—number of sensors `N`, operating frequency `f`, inter-element spacing `dx`, and source directions `theta_true`—to match your experiment.
2. Run the script to generate synthetic array data, sweep the look-direction grid, and compute the DAS and MUSIC spectra.
3. Review the command-window output for the estimated angles and inspect the generated figures to compare the beamformer responses.

## Implementation Overview
### Delay-and-Sum (DAS)
Models each source as a narrowband plane wave, applies phase delays across the array, adds complex Gaussian noise, and forms the beam by coherently summing the delayed microphone signals across a scanning grid. Peaks in the normalized spectrum reveal the DoA estimates, which are highlighted in the plots and printed to the console.

### MUSIC
Collects multiple noisy snapshots, estimates the spatial covariance matrix, performs an eigen-decomposition to separate signal and noise subspaces, and evaluates the pseudo-spectrum across the scan grid via the noise-projection criterion. The normalized spectrum produces sharp peaks at the estimated source directions that are annotated in the figures and printed for comparison with DAS.

## Key Findings from the Report
- For moderately separated sources, both DAS and MUSIC resolve the targets, with MUSIC providing sharper responses.
- When sources are 1° apart, DAS produces a merged peak, whereas MUSIC maintains distinct maxima thanks to its subspace formulation.
- Designing the array for 5000 Hz with roughly 2° resolution motivates a dense aperture; half-wavelength spacing (`d = λ/2`) balances alias-free operation and angular resolution.
- Increasing the spacing beyond `λ/2` introduces grating lobes in DAS and can degrade MUSIC performance, underscoring the importance of careful array layout.

## Customization Tips
- Toggle between single- and multi-source experiments by editing `theta_true`, or experiment with different noise levels (`sigma_noise`) and snapshot counts (`L`) to explore robustness.
- Modify the scan resolution (`theta_scan`) or peak-detection settings to study the effect on detection confidence and false alarms.
- Adjust microphone spacing `dx` to replicate the spacing experiments discussed in the report and observe how aliasing impacts the algorithms.

## Testing
⚠️ Not run (documentation-only update)
