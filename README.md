# NUTRIG

NUTRIG develops novel autonomous radio-triggering techinques for GRAND. It consists of a [first-level trigger](https://pos.sissa.it/470/060/) (FLT) at the antenna level, and a [second-level trigger](https://pos.sissa.it/470/061/) (SLT) at the array level. This GitHub project focuses primarily on the template-fitting FLT method. See [here](https://github.com/grand-mother/NUTRIG1.git) for the CNN FLT method software and [here]() for SLT software.

## Overview

This project contains the source code for the following elements of the NUTRIG project:

- creation of a database for offline FLT and SLT studies;
- selection of templates for the template-fitting FLT;
- development of the template-fitting FLT method.

## Setup

This code relies heavily on the [`GRANDlib` package](https://github.com/grand-mother/grand.git). Therefore, it is advised to use the dedicated `conda` environment as explained [here](https://github.com/grand-mother/grand/blob/master/env/conda/readme.md). Before running `nutrig` scripts, you typically need to activate the `conda` environment and setup both `GRANDlib` and `nutrig`:

```bash
>>> conda activate grandlib_conda
>>> source /path/to/grandlib/env/setup.sh
>>> source /path/to/nutrig/env/setup.sh
```
