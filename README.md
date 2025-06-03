# NUTRIG

NUTRIG develops novel autonomous radio-triggering techinques for GRAND. It consists of a first-level trigger (FLT) at the antenna level, and a second-level trigger (SLT) at the array level. This GitHub project focuses primarily on the template-fitting FLT method. See link for the CNN FLT method and link for SLT software.

## Overview

This project contains the source code for the following elements of the NUTRIG project:

- creation of a database for offline FLT and SLT studies;
- selection of templates for the template-fitting FLT;
- development of the template-fitting FLT method.

## Setup

This code relies heavily on the `GRANDlib` package. Therefore, it is advised to use the dedicated `conda` environment as explained here. Before running `nutrig` scripts, you typically need to activate the `conda` environment and setup both `GRANDlib` and `nutrig`:

```bash
>>> conda activate grandlib_conda
>>> source /path/to/grandlib/env/setup.sh
>>> source /path/to/nutrig/env/setup.sh
```
