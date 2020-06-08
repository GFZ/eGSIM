# eGSIM
A web service for selecting and testing  ground shaking models in Europe (eGSIM), developed
in the framework of the  Thematic Core Services for Seismology of EPOS-IP
(European Plate Observing  System-Implementation Phase)

## Installation (development)

Disclaimer: this is a temporary tutorial tested with MacOS (El Capitan) and Ubuntu 16.04. 

### Requirements (Ubuntu specific, in Mac should not be an issue, otherwise `brew install` instead of `apt-get install`):
```bash
brew doctor  # pre-requisite
brew update # pre-requisite
brew install gcc
sudo apt-get install git python3-venv python3-pip python3-dev
```

Please use Python 3.7+. Check with ```python --version```: If it's Python2, then use ```python3 --version```. If it's not 3.7+, then you need to install Python3.7 **along with** (i.e., not replacing) the current default python3 installed on your computer.
From now on, each `python` command refers to the path of the Python3.7 distribution you have (i.e., you might need to type e.g. `/opt/lib/python3.7` or something similar, instead of `python` or `python3`)


### Activate virtualenv (links TBD).
[Pending: doc TBD] Three options:
  1. python-venv (for python>=3.5): Please use this option as it does not issues the 'matplotlib installed as a framework ...' problem
  2. python-virtualenv
  3. virtualenvwrapper (our choice)

*FROM NOW ON virtualenv MUST be activated! EVERYTHING WILL BE INSTALLED ON YOUR "copy" of pyhton with no conflicts with the OS python distribution*


### Clone repository

```bash
git clone https://github.com/rizac/eGSIM.git
git clone https://github.com/rizac/gmpe-smtk.git
```


### Install

Move to egsim directory and type:

```bash
pip install --upgrade pip setuptools && pip install -r ./requirements.txt
cd ../gmpe-smtk  # or wherever smtk is cloned to, see above
pip install -e .  # -e necessary only in dev mode
cd ../egsim  # or wherever egsim is cloned to, see above
```


### Test

Normal test (x=stop at first error, v*=increase verbosity):
```bash
pytest -xvvv --ds=egsim.settings_debug ./tests/
```

Test with coverage
```bash
pytest -xvvv --ds=egsim.settings_debug --cov=./egsim/ --cov-report=html ./tests/
```

### Setup project data

#### Flatfile (ESM):
This procedure should be executed for all flatfiles to be included in the application.
For testing purposes, we will use ESM flatfile (2018) only.
Download ESM flatfile from https://esm.mi.ingv.it//flatfile-2018/flatfile.php  (ESM_flatfile_2018)
Unzip it and from within the same directory, copy the file:
```bash
cp ESM_flatfile_2018/ESM_flatfile_SA.csv ./ESM_flatfile_2018_SA.csv
```
Called $FLATFILE_PATH the full path of the CSV file just copied, now parse it into the ESM database, the database will be a HDF5 file inside the /media/ directory of the egsim repository (git ignores that directory):
```bash
export DJANGO_SETTINGS_MODULE="egsim.settings_debug";python manage.py gmdb_esm $FLATFILE_PATH
```

#### Migrate (setup django db)
From within the egsim folder (check that manage.py is therein):
```bash
export DJANGO_SETTINGS_MODULE="egsim.settings_debug";python manage.py migrate
```

#### Create db  (setup egsim tables inside django db)
From within the egsim folder (check that manage.py is therein):
```bash
export DJANGO_SETTINGS_MODULE="egsim.settings_debug";python manage.py initdb
```

## Run:
```bash
export DJANGO_SETTINGS_MODULE="egsim.settings_debug";python manage.py runserver
```
and pen your browser (or use the API) at the URL address on the console 


## Installation (production)

Please refer to 'deploy.html' (dynamic web page, open it in your browser of choice)


## Maintenance

### Github packages security issues / dependencies alert:

Security alerts on Githib should be solved by [upgrading the dependencies](#dependencies-upgrade), as
most of the require packages are OpenQuake dependency and thus it's safer to keep everything consistent.
However, if a security alert has to be fixed:

Open `requirements.txt` and change the version of the package to be upgraded.

Run `pip install -r requirements.txt`

Run [tests](#test)

And proceed also on the server, if you have an installed version.

Note that you might get errors such as:

```bash
ERROR: openquake-engine 3.5.0 has requirement django<2.1,>=1.10, but you'll have django 2.2.10 which is incompatible.
```

which is the reason why it's safer to upgrade everything consistently. However, those messages seem to be
more warnings than errors (the installation seems not to be interrupted): in case, as always, run tests and check
if everything works.


### Dependencies upgrade

Dependencies upgrade should be done when really necessary, as it usually requires several fixes in egsim
and sometime in smtk, too. To upgrade dependencies, create a new virtual environment, and then:

```bash
pip install --upgrade pip setuptools && pip install openquake-engine

# Move to the smtk directory (see section Clone-repository)
git pull && pip install -e .

# Run tests
pip freeze > ./requirements.txt
```

*Important*: open `requirements.txt` and **comment the line with gmpe-smtk**,
because it must be installed AFTER openquake (see (#installation))


