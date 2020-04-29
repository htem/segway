# segway
Common scripts for segmentation

## Load tmn7's maintained Python 3 environment

Before running any segmentation job, activate daisy env: `activatedaisy`

### On Orchestra cluster:

Add these to your ~/.bashrc:

`module load gcc/6.2.0 boost/1.62.0 python/3.6.0 cuda/9.0` \
`alias activatedaisy='source /home/tmn7/daisy/bin/activate'`

### On Lee Lab's local computers (gpu0/gpu1/dwalin/balin):

`alias activatedaisy='source /n/groups/htem/users/tmn7/envs/ubuntu180402/bin/activate'`


## Manual setup for a new python virtual environment

_You only have to do this if you want to have your own environment_

Follow this direction for a new Python env:

https://packaging.python.org/guides/installing-using-pip-and-virtualenv/

Then install these packages with PIP

`pip install cython numpy scipy tornado mahotas pymongo tensorflow-gpu`

Then go into repos/ and install these packages:

`daisy`, `gunpowder`, `zarr`, and `waterz`
