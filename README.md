## Classifier Chain Network
This repository contains the replication files for the paper:

D.J.W. Touw and M. van de Velden (2024). Classifier Chain Networks for Multi-Label Classification.

This repository contains a snapshot of the implementation of the classifier chain network as it was at the time of obtaining the results presented in the paper. For the latest version, please refer to [CCNPy](https://github.com/djwtouw/CCNPy).

## Installation
Supported Python version: Python 3.12.

The implementation of the classifier chain network depends on the Eigen C++ linear algebra library. Download Eigen from https://eigen.tuxfamily.org (version 3.4.0, available at https://gitlab.com/libeigen/eigen/-/releases/3.4.0). Copy the directory with the source code (called `Eigen`, located in `eigen-3.4.0`) into `packages/eccn/cpp/include`. In the terminal, run `pip install -r requirements.txt` from the root directory and all required packages will be installed.

## Repository Structure
The contents of the directories are as follows:
- `applications`: contains the scripts and data for the application presented in the paper.
- `models`: implementations of various models used in the analyses and comparisons. Typically, these consist of simple wrappers around existing code to streamline the analysis (for instance, to have similar function calls to train different models).
- `packages`: contains the `eccn` directory, which contains a Python package with C++ implementatation of the classifier chain network; the `scikit-multilearn` directory, which contains source code retrieved from https://github.com/scikit-multilearn/scikit-multilearn, with minor modifications for compatibility (see next section); and the `rbrl` directory, which contains source code retrieved from https://github.com/GuoqiangWoodrowWu/RBRL.
- `simulations` contains all scripts to obtain the results presented in the paper. There is a configuration file which specifies the different data generating processes, a testing script to test all models, and scripts that run the simulations.
- `utils`: additional useful functions.

## Scikit-Multilearn
The `scikit-multilearn` package was downloaded from https://github.com/scikit-multilearn/scikit-multilearn. The following adjustment was made for compatibility.

To make sure the multi-label k-nearest neighbors algorithm works, line 129 in `packages/scikit-multilearn/skmultilearn/adapt/mlknn.py` was changed from
```
self.knn_ = NearestNeighbors(self.k, n_jobs=self.n_jobs)
```
into
```
self.knn_ = NearestNeighbors(n_neighbors=self.k, n_jobs=self.n_jobs)
```

## RBRL
The Matlab source files for this method were downloaded from https://github.com/GuoqiangWoodrowWu/RBRL. In order to use the method implemented in this software package, a Matlab installation in required and the Python package `matlabengine`. The required version of `matlabengine` in this repository is 24.2.1, which is compatible with Matlab2024b. The required version in `requirements.txt` may need to be modified depending on the version of Matlab.

## License
This project uses the Eigen C++ library for linear algebra operations, which is licensed under the [MPL2.0](https://www.mozilla.org/en-US/MPL/2.0/) license. For more information about Eigen, including documentation and usage examples, please visit the [Eigen website](https://eigen.tuxfamily.org/).

This project contains the source code of `scikit-multilearn`, which is licensed under the [2-Clause BSD License](https://opensource.org/license/bsd-2-clause). For more information about scikit-multilearn, including documentation and usage examples, please visit the [scikit-multilearn website](http://scikit.ml/).

This project is licensed under the GNU General Public License version 3 (GPLv3). The GPLv3 is a widely-used free software license that requires derivative works to be licensed under the same terms. The full text of the GPLv3 can be found in the `LICENSE` file.
