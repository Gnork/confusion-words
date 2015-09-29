### Settings

* training settings can be edited at top of \__main__.py file

### Dependencies

* <a href="https://www.python.org/downloads/release/python-343/">python 3.4</a>
* <a href="http://deeplearning.net/software/theano/install.html">theano 0.7</a>
* <a href="http://www.scipy.org/install.html">numpy/scipy</a>
* <a href="https://radimrehurek.com/gensim/install.html">gensim</a>

tested on Ubuntu 14.04.2, NVIDIA GeForce GTX 870M, Driver Version: 346.72, Cuda 7.0, Bumblebee 3.2.1

### Run

```bash
cd confusion-words
python3 lstm_word2vec_language_model
```

or with bumblebee for NVIDIA optimus support on Linux:

```bash
cd confusion-words
optirun python3 lstm_word2vec_language_model
```

### Results

* results for generating sequences will be printed
* weights and cost function errors will be pickled to files in regular intervals during training
* saved files can be loaded for continuing training or to apply model without training

### Acknowledgements

* Christian Herta for LSTM tutorial: http://christianherta.de/lehre/dataScience/machineLearning/neuralNetworks/LSTM.php (code snippet inspired by tutorial marked in \__main__.py)