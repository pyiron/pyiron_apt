# pyiron_apt
APT analysis with pyiron(_workflows)

Collection of nodes/pyiron jobs to collaboratively develop APT workflows.
Part of [IUC09](https://nfdi-matwerk.de/project/structure/use-cases/iuc09) NFDI-MatWerk

### Getting started

Create an environment with conda:

```
conda env create -f environment.yml
```

After creating the environment, activate it by

```
conda activate pyiron_apt
```

And start jupyter lab

```
jupyter lab
```

### Trying the notebooks

- First download data using `01_download_data.ipynb`, or use your own dataset.
- Composition Space workflow: `02_workflow_compositionspace.ipynb` contains Composition space workflow.
- Paraprobe workflow: `03_workflow_paraprobe.ipynb` 
- Mixed workflow: combine both tools `04_mixed_workflow.ipynb`

TIP: Press Tab to use auto-complete to browse through the available options.

The nodes used in the workflow are available in the `nodes` folder.
To add new nodes, and for a tutorial on how to do this, see the [`pyiron_workflow`](https://github.com/pyiron/pyiron_workflow) repository, especially [quickstart](https://github.com/pyiron/pyiron_workflow/blob/main/notebooks/quickstart.ipynb) and [deepdive](https://github.com/pyiron/pyiron_workflow/blob/main/notebooks/deepdive.ipynb).

### Fixing and reporting bugs

- For adding nodes, open a pull request to this repo.
- To fix bugs, please open an issue, and add a PR to this repo.

### Compositing the scientific workflow

- Please see above.

### TO-DO

- [Enhancement] Set up actions and such with pyiron module template
- [Bug] `paraprobe_transcoder.py` has wrong imports; which needs to be fixed in the conda package. For the moment, the working file is included here.
