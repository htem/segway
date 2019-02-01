# segway
Common scripts for segmentation

## Setup

Add these to your ~/.bashrc:

`module load gcc/6.2.0 boost/1.62.0 python/3.6.0 cuda/9.0` \
`alias activatedaisy='source /home/tmn7/daisy/bin/activate'`

### Quickstart

Before running any segmentation job, activate daisy env: `activatedaisy`


## Manual setup for a new python virtual environment

_You only have to do this if you really want your own environment_

Follow this direction for a new Python env:

https://packaging.python.org/guides/installing-using-pip-and-virtualenv/

Then install these packages with PIP

`pip install cython numpy scipy tornado mahotas pymongo tensorflow-gpu`

Then go into repos/ and install these packages:

`daisy`, `gunpowder`, `zarr`, and `waterz`
