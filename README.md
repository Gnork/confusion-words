# Confusion Words

The python code in this repository is part of my master's thesis.
* the code is distributed under MIT license (for more information see LICENSE file)
* every subfolder contains an individual algorithm
* for detailed instructions see README.md files in subfolders
* for further explanations see master's thesis "Evaluation computerlinguistischer Verfahren zur Erkennung von Confusion-Word-Fehlern" (Christoph Jansen, August 2015, HTW Berlin)

## Dependencies

All algorithms can be trained on Brown corpus. Run prepare_data.py to automatically download corpus to home directory and generate word2vec model.

* <a href="https://www.python.org/downloads/release/python-343/">python 3.4</a>
* <a href="http://www.nltk.org/install.html">nltk</a>
* <a href="https://radimrehurek.com/gensim/install.html">gensim</a>

```bash
cd confusion-words
python3 prepare_data.py
```