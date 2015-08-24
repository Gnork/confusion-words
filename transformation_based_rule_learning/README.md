### Settings

* training settings can be edited at top of \__main__.py file

### Dependencies

* <a href="https://www.python.org/downloads/release/python-343/">python 3.4</a>

tested on Ubuntu 14.04.2

### Run

```bash
cd confusion-words
python3 transformation_based_rule_learning
```
### Results

* program creates timestamped subfolder for each training
* subfolder results contains result files:
	* experiment.json (contains settings used in experiment)
	* rules.pickle (python list with rule objects, can be unpickled)
	* training.csv (contains statistical results)