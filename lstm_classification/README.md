### Settings

* training settings can be edited at top of \__main__.py file

### Dependencies

* <a href="https://www.python.org/downloads/release/python-343/">python 3.4</a>
* <a href="http://deeplearning.net/software/theano/install.html">theano 0.7</a>
* <a href="http://www.scipy.org/install.html">numpy/scipy</a>
* <a href="http://keras.io/#installation">keras</a>
* <a href="http://docs.h5py.org/en/latest/build.html#install">h5py</a>
* <a href="https://radimrehurek.com/gensim/install.html">gensim</a>

tested on Ubuntu 14.04.2, NVIDIA GeForce GTX 870M, Driver Version: 346.72, Cuda 7.0, Bumblebee 3.2.1

### Run

```bash
cd confusion-words
python3 lstm_classification
```

or with bumblebee for NVIDIA optimus support on Linux:

```bash
cd confusion-words
optirun python3 lstm_classification
```

### Results

* results for applying classification model to confusion set will be printed
* weights (HDF5 format) and errors (pickled python list) will be saved after each training epoch
* saved files can be loaded for continuing training or to apply model without training