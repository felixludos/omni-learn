# omni-learn

This repo contains a framework for various machine learning and simulation projects.

## Installation

Run `pip install .` while in this dir to install package.

### Environment Variables

It is strongly encouraged to set a few environment variables when using this library:

`OMNILEARN_SAVE_DIR="$HOME/trained_nets"` - set this to an absolute path to a directory you have write access to, this is where scripts will save output (by default).

`OMNILEARN_DATA_DIR="$HOME/local_data"` - set this to an absolute path to a directory with some datasets (best to also have write access to allow automatic downloading of some datasets).



## Execution

For an example of how to use this library, see the `mnist/` dir.

An example execution from inside the `mnist/` dir (requires the environment variables above to be set):

`python project.py model --name test-mnist`