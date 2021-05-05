# QiPRNG *(pronounced "chee"-PRNG)*
A DT**Q**W-**i**nspired **PRNG**. The output sequence is a string of the least significant bits from the measurement probabilities corresponding to a DTQW determined by an arbitrary Hamiltonian. The construction of the DTQW follows Childs' 2010 paper [On the relationship between continuous-and discrete-time quantum walk](https://link.springer.com/article/10.1007/s00220-009-0930-1).

For reference on the statistical tests, see [NIST's page](https://www.nist.gov/publications/statistical-test-suite-random-and-pseudorandom-number-generators-cryptographic).

# Usage Instructions

The directory structure must be initialized before any data is generated. To do this, run ```mkdir data data/binary_data data/stats_output``` from the ```QiPRNG``` directory.

If using the [C implementations](https://csrc.nist.gov/Projects/Random-Bit-Generation/Documentation-and-Software) of the NIST SP 800-22 statistical suite, navigate to the ```src``` directory and compile the library with ```cc -fPIC -shared -o sp800.so ./sp800_22_tests_c/src/*```.

Data generation can then be invoked with ```python DataProcessing.py```, or ```sbatch generate.sh``` if running on a cluster.
