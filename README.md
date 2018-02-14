# eGSIM
A web service for selecting and testing  ground shaking models in Europe (eGSIM), developed
in the framework of the  Thematic Core Services for Seismology of EPOS-IP
(European Plate Observing  System-Implementation Phase)

## Installation

Disclaimer: this is a temporary tutorial tested with MacOS (El Capitan) and Ubuntu 16.04. 

### Requirements (Ubuntu specific, in Mac should not be an issue, otherwise `brew isntall` instead of `apt-get install`):
```bash
brew doctor  # pre-requisite
brew update # pre-requisite
brew install gcc
sudo apt-get install git python3-venv python3-pip python3-dev
```

### Activate virtualenv (links TBD).
[Pending: doc TBD] Three options:
  1. python-venv (for python>=3.5): suggested as it does not issues the 'matplotlib installed as a framework ...' problem
  2. python-virtualenv
  3. virtualenvwrapper (our choice)

*FROM NOW ON virtualenv is activated! EVERYTHING WILL BE INSTALLED ON YOUR "copy" of pyhton WITH no mess-up with the OS python distribution*

### Upgrade pip and setuptools:
pip install -U pip setuptools

### Install
pip install -r ./requirements.txt


## Old stuff (ignore, will be removed soon):

### ~~First install numpy~~

*Note: this is maybe not necessary as it turned out we needed gcc first (see above). However it's harmless to do it now*

```bash
pip install numpy
pip install scipy
```

### ~~Install oq-engine~~:

Full ref here: https://github.com/gem/oq-engine/blob/master/doc/installing/development.md

clone repository:
```bash
mkdir ../oq-engine # or whatever you want
cd ../oq-engine/  # or whatever you specified above
git clone https://github.com/gem/oq-engine.git .
```
install as editable (this should make git-pull in the repository enough to have the newest version):
```bash
pip install -e .
```

### ~~Install gmpe-smtk~~:

Full ref here: https://github.com/GEMScienceTools/gmpe-smtk

clone repository:
```bash
mkdir ../gmpe-smtk # or whatever you want
cd ../gmpe-smtk/  # or whatever you specified above
git clone https://github.com/GEMScienceTools/gmpe-smtk.git .
```
install as editable (this should make git-pull in the repository enough to have the newest version):
```bash
pip install -e .
```


## Run locally:

Pending: update doc

