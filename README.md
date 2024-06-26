# WaveFunctionSimPython
Python implementation of a simulation using a numeric approximation of the time-dependent Schrödinger equation.

---

## Requirements
CUDA-Enabled NVIDIA GPU  
CUDA Toolkit v12.x  
Python 3.11 or newer  

## How to install

- Download Python from https://www.python.org/downloads/ and install it on your computer.  
On Linux systems alternatively you can use APT:
```
    sudo apt update
    sudo apt install python3
```
- Download the CUDA Toolkit from https://developer.nvidia.com/cuda-downloads and install it on your computer.
- On Linux systems with no physical display connected, make sure to also install X virtual framebuffer by `sudo apt install xvfb`.
- Download and install Git version control system from https://git-scm.com/downloads.
- Open a command-line and clone this repository by entering  
`git clone https://github.com/TheFlyingPiano99/WaveFunctionSimPython.git`
- Enter the cloned folder by `cd WaveFunctionSimPython`
- Create a Python virtual environment for this application using `python3 -m venv .venv`
- Activate the environment by running one of the `activate.*` scripts under the Scripts folder in the newly created `.venv` folder.  
  On Linux systems use:
  ```
  source .venv/bin/activate
  ```
  (To deactivate later, use the `deactivate` command)
- While the virtual environment is active (This can be seen from the presence of the `(.venv)` label at the beginning of the prompt.), upgrade pip: `python -m pip install --upgrade pip`
- Install the required packages by typing: `python -m pip install -r requirements.txt`
This might take a while.
- After everything is installed, run the application by:
  ```
  python wavefunctionsim.py
  ```
- After the plugins are installed, you can also use the `run_script.sh` (Linux/MacOS) or `run_script.bat` (Windows)
to skip the step of manually activating the virtual environment.

## Generating documentation
To generate code docuemntation from the root directory of the project run:
```
sphinx-build -M html docs/source/ docs/build/
```

## How to use
### Configuration
The simulator uses a configuration file located under `<project folder>/config/`
This file uses the Tom's Obvious Minimal Language.
If you want to learn about TOML, browse https://toml.io/en/.
Read the provided configuration files in `<project folder>/config/` and under `<project folder>/archive/final_configs/` to discover the available options.

### Running the simulation
If you start the application, it will check the content of the `<project folder>/cache/` folder.
If the configuration file hasn't changed since the last run, and there are cached datasets under the `cache` folder, it will try to use these cached files to speed up the initialization phase.
Always read the prompts of the program!

### Accessing the simulation results
The output images and videos get written into the `<project folder>/output/` folder.
You can also use the `wfreader.py` utility script located in the root directory of the project.
Run the following script to view the output:
```
python wfreader.py [<path to output folder>] [-h] [-p]
```
If no output folder is specified, the default `output` folder will be used.
The optional `-h` flag prints help text.
The optional `-p` flag enables the visualization of the predicted expected location and standard deviation of a Gauss WP in free space.
This can be used to check the differences between the analytic model and the numeric approximation. 


