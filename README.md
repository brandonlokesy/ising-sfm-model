# Simulating an injection-locked VCSEL Ising computer

## How to use the files
- The file `3bit_solve_example.py` contains the script required to solve the 3-bit Ising problem found in the paper. The Hamiltonian to be solved can be selected by changing the matrices `A` and `AHWP`.
- The file `3bit_data_process.py` determines the dominant polariztion state from the files generated from the script `3bit_solve_example.py` and writes it to a CSV file. The CSV file can be used to determine the counts of each final Ising state
- The file `master_injection.py` sweeps the injected power and the frequency detuning of a master laser into a single VCSEL