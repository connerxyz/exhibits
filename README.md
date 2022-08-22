## Development setup

We need to setup an isolated Python environment with the right version of Python and all the necessary depdendencies. We
use `conda` to do this. Execute the following steps in order.

### What is `conda`?

From [here](https://docs.conda.io/en/latest/).

> Conda as a package manager helps you find and install packages. If you need a package that requires a different
> version of Python, you do not need to switch to a different environment manager, because conda is also an environment
> manager. With just a few commands, you can set up a totally separate environment to run that different version of
> Python, while continuing to run your usual version of Python in your normal environment.

### Install `conda`

- Go [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/#regular-installation)
- Select your operating system
- Select "Miniconda installer for ..."
- Download the installer and follow the instructions

### Test the install

Open a command line interface and enter `conda`. You should somethiing like below.

```
$ conda
usage: conda [-h] [-V] command ...

conda is a tool for managing and deploying applications, environments and packages.

Options:

positional arguments:
  command
    clean        Remove unused packages and caches.
    config       Modify configuration values in .condarc. This is modeled
                 after the git config command. Writes to the user .condarc
                 file (/Users/U6020643/.condarc) by default.
    create       Create a new conda environment from a list of specified
                 packages.
    help         Displays a list of available conda commands and their help
                 strings.
    info         Display information about current conda install.
    install      Installs a list of packages into a specified conda
    ...
```

### Initialize the conda environment

Enter the following to specify a Python environment with the right Python version.

```
conda create --yes -n connerxyz python=3.8
```

### Activate the environment

```
conda activate connerxyz
```

### Clone the repository

```
git clone https://github.com/connerxyz/connerxyz.git
```

### Install the dependencies

Move into the repository and install the depdendencies.

```
cd connerxyz
pip install -r requirements.txt
```

### Start the deveopment server

If you can run `makefiles`, you can do this.

```
make dev
```

Otherwise you can execute

```
export FLASK_APP=cxyz
export FLASK_ENV=development
export FLASK_RUN_PORT=5001
flask run
```

From here you should be able to open http://localhost:5001 and view the site. Make changes to the code and these should
be reflected in the browser.
