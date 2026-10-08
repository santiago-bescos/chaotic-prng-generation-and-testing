# Chaotic PRNG Generation and Statistical Testing

A reproducible Python pipeline for generating pseudorandom bitstreams from chaotic dynamical systems, post-processing them with extraction methods, and evaluating their statistical quality using the NIST SP 800-22 test suite.

This repository was developed as my Final Degree Project in Mathematical Engineering.

## Project Overview

The project explores whether chaotic maps can be used as a practical source of pseudorandom binary sequences.

The pipeline supports:

- Bitstream generation from the Logistic, Tent and Hénon maps
- Post-processing using SHA-2, SHA-3 and Von Neumann extraction
- Statistical evaluation with NIST SP 800-22
- Comparison with ChaCha20 as a cryptographic baseline
- Analysis of chaotic behaviour through attractors, bifurcation diagrams and Lyapunov exponents
- Investigation of numerical precision and floating-point resolution
- Export of generated sequences in several formats for further testing

The emphasis of the project is on reproducibility, traceability and structured comparison between generators and post-processing methods.

## Tech Stack

- **Python**
- **NumPy**
- **pandas**
- **Jupyter Notebook**
- **NIST SP 800-22**
- **SHA-2 / SHA-3**
- **ChaCha20**

## Pipeline

The workflow can be summarised as:

```text
Chaotic map
    ↓
Raw numerical sequence
    ↓
Binarisation
    ↓
Raw bitstream
    ↓
Post-processing
    ├── SHA-2
    ├── SHA-3
    └── Von Neumann extractor
    ↓
Export
    ↓
NIST SP 800-22 testing
    ↓
Result parsing and statistical analysis
```

## Supported Generators

### Logistic Map

The Logistic map is analysed through:

- Orbit generation
- Attractor behaviour
- Bifurcation diagrams
- Lyapunov exponents
- Transient behaviour
- Binarisation-threshold analysis

### Tent Map

The Tent map is used as another one-dimensional chaotic generator and follows the same general generation and evaluation pipeline.

### Hénon Map

The Hénon map extends the analysis to a two-dimensional chaotic system.

The project includes:

- Orbit generation
- Phase-space attractor visualisation
- Lyapunov exponent estimation
- Threshold analysis

### ChaCha20

ChaCha20 is included as a cryptographic reference generator for comparison with the chaos-based approaches.

## Statistical Evaluation

Generated bitstreams are evaluated using the **NIST SP 800-22 Statistical Test Suite**.

The repository includes tools to:

- Parse NIST result files
- Extract individual test outcomes
- Calculate pass ratios
- Organise results in pandas DataFrames
- Compare generators and post-processing variants
- Build summary tables and test-family comparisons

The purpose of these tests is to evaluate the statistical properties of the generated sequences. Passing statistical randomness tests does not by itself establish cryptographic security.

## Post-processing

Several variants can be produced from each raw chaotic bitstream:

- `raw` — unprocessed output
- `sha2` — SHA-2-based extraction
- `sha3` — SHA-3-based extraction
- `vn` — Von Neumann extraction

This makes it possible to compare the effect of different post-processing strategies on the statistical quality of the generated sequences.

## Numerical Precision Analysis

Because chaotic systems are sensitive to initial conditions but are implemented using finite-precision arithmetic, the project also investigates floating-point resolution.

Utilities are included to examine:

- `float64` spacing around specific values
- The number of representable values within an interval
- Approximate information capacity associated with floating-point intervals

## Repository Structure

```text
chaos_prng/
├── analysis/
│   ├── henon_analysis.py
│   ├── logistic_analysis.py
│   ├── nist_sts.py
│   └── tent_analysis.py
│
├── generators/
│   └── chaotic_bits.py
│
├── notebooks/
│   ├── analysis_logistic.ipynb
│   ├── analysis_henon.ipynb
│   └── nist_results_analysis.ipynb
│
├── nist_results/
│
├── utils/
│   └── floating_space.py
│
├── export_helpers.py
├── io_bits.py
├── export_chacha_bits.py
├── main_export_demo.py
└── mass_export_bits.py
```

## Main Components

### `generators/`

Contains the chaotic bitstream generators and a unified interface for generating sequences from the supported maps.

### `analysis/`

Contains functions for studying the behaviour of the chaotic systems and for processing statistical-test results.

Examples include:

- Attractor visualisation
- Bifurcation analysis
- Lyapunov exponent calculation
- Threshold analysis
- NIST result parsing and comparison

### `notebooks/`

Contains Jupyter notebooks used for exploratory analysis, visualisation and interpretation of results.

### `nist_results/`

Contains outputs generated from the NIST SP 800-22 testing pipeline.

### Export utilities

The project supports exporting generated bitstreams as:

- ASCII `0/1`
- Packed binary
- `uint32`

Manifest files can also be produced to keep track of generated datasets and their configurations.

## Project Goals

The main goals of the project were to:

1. Build a modular and reproducible chaos-based bitstream generation pipeline.
2. Compare several chaotic maps and parameter configurations.
3. Evaluate the impact of post-processing methods on sequence quality.
4. Analyse the generated sequences using established statistical randomness tests.
5. Compare chaos-based outputs with a cryptographic reference generator.
6. Study implementation issues such as floating-point precision and reproducibility.

## Notes

This repository is intended as an academic and experimental project focused on pseudorandomness, chaotic systems and statistical testing.

The statistical results produced by NIST SP 800-22 should be interpreted as evidence about statistical behaviour, not as proof that a generator is cryptographically secure.
