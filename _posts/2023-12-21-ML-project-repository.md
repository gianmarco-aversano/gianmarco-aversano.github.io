---
title: "Ultimate guide for a Machine Learning repository."
categories:
  - blog
tags:
  - machine learning
  - ml
  - python
  - repository
---

Alright so if you landed here it's because you want to set up a new repository for a machine learning (ML) project. And probably are not sure how to do it.

As Python is the most popular programming language for ML, we'll use that, which also menas that we need to set up everything in a way that also respects the Python development best practices. So if you see a `setup.py` file, that's bad. It's 2023 at the time of writing. Evolve.

You may want to check out Cookiecutter, which comes with templates to set up new Python projects. You can even create your own. But let's start from zero here.

Also beware that I'm writing this piece under the hypothesis that you are on Linux/Mac. If you're on Windows, sorry. Maybe next time.

Most of the code here can be found at: [my-template](https://github.com/svnv-svsv-jm/init-new-project).

<!-- By the end, your project will look something like:

```bash
>> tree .
.
├── Dockerfile
├── LICENSE
├── Makefile
├── README.md
├── docker-compose.yml
├── examples
├── experiments
│   └── README.md
├── mypy.ini
├── pylintrc
├── pyproject.toml
├── pytest.ini
├── scripts
│   ├── docker-installation-steps.sh
│   ├── entrypoint.sh
│   ├── git-clean.sh
│   └── pytest.sh
├── src
│   ├── README.md
│   └── project_name
│       ├── __init__.py
│       └── config.py
└── tests
    ├── __init__.py
    ├── conftest.py
    ├── e2e
    │   └── __init__.py
    ├── integration
    │   └── __init__.py
    └── unit
        └── __init__.py
``` -->

## Pre-requisites

First off, create a new folder and go into it.

```bash
mkdir new-cool-ml-prok
cd new-cool-ml-prok
```

Neat.

Now open this folder with VSCode. We don't use PyCharm over here.

## Virtual environment

We now need a virtual environment. Check out Pyenv and never go back to anything else. Make sure that it is correctly installed and that you have the following lines:

```bash
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
```

at the end of your `~/.bashrc` (Linux/Ubuntu) or `~/.bash_profile` (Mac) or `~/.zshrc` / `~/.zprofile` (normal people). If you're not using Oh-My-Zsh, please ask yourself some serious questions.

Now, create a virtual environment with a desired Python version:

```bash
pyenv install <version> # desired python version, something like 3.10.10 or 3.12.0
pyenv virtualenv <version> <some-name> # e.g. pyenv install 3.10.10 cool-proj
pyenv shell <some-name>
```

In VSCode, open any `.py` file, then in bottom bar (usually on the bottom right) you should be able to select a Python interpreter. Select the environment you just created. If you can't see it, start typing its name, or restart VSCode.

## Getting started

We need to create the project's metadata. So install Poetry:

```bash
pip install --upgrade pip
pip install poetry
poetry init
```

You'll be prompted for some project metadata, such as project name, etc. You can also just smash "Enter" and leave almost everything blank. Poetry will create the `pyproject.toml` file, which is very important. It contains all the project's information.

This file should look something like this:

```toml
[tool.poetry]
name = "project-name" # choose a nice project name
version = "0.1.0" # select a version number
description = "Description." # please describe it
authors = ["Name <address@email.com>"]
license = "LICENSE" # make sure this file exists
readme = "README.md" # make sure this file exists
packages = [{ include = "project_name", from = "src" }] # read belows
include = ["*.py", "src/**/*.json", "src/**/*.toml"] # on this later
exclude = ["test/*"] # on this later

[build-system]
requires = ["poetry-core>=1.0.0", "cython"]
build-backend = "poetry.core.masonry.api"
```

Rather self-explicative. Now create the following files:

- `src/project_name/__init__.py` (package creation file);
- `src/README.md` (you can place any guidelines for how other developers can contribute to your project here).

At this point, your repository looks something like this:

```bash
>> tree .
.
├── LICENSE
├── README.md
├── pyproject.toml
├── src
    ├── README.md
    └── project_name
        └── __init__.py
```

The `README.md` file should contain installation instructions for your package, and how it can be used. Don't be shy to provide examples and/or links to other documentation. Without any of these two things, you may have coded the best thing ever, but it'll be USELESS.

## Dependencies

This is what your `pyproject.toml` should look like:

```toml
[tool.poetry]
name = "project-name" # choose a nice project name
version = "0.1.0" # select a version number
description = "Description." # please describe it
authors = ["Name <address@email.com>"]
license = "LICENSE" # make sure this file exists
readme = "README.md" # make sure this file exists
packages = [{ include = "project_name", from = "src" }] # read belows
include = ["*.py", "src/**/*.json", "src/**/*.toml"] # on this later
exclude = ["test/*"] # on this later

[build-system]
requires = ["poetry-core>=1.0.0", "cython"]
build-backend = "poetry.core.masonry.api"

# Specify Python version(s) and real dependencies in this section
[tool.poetry.dependencies]
python = ">=3.8,<3.11"
jupyter = "*"
jupyterlab_server = "*"
jupyterlab = "*"
pyrootutils = "*"
loguru = "*"

# Here, specify development dependencies, which won't be part of the actual final dependency list
# but that you need, well, to develop your project
[tool.poetry.dev-dependencies]
black = { extras = ["jupyter"], version = "*" }
flake8 = "*"
ipython = "*"
isort = "*"
mypy = "*"
pylint = "*"
pytest = "*"
pytest-cov = "*"
pytest-mock = "*"
pytest-pylint = "*"
pytest-mypy = "*"
pytest-testmon = "*"
pytest-xdist = "*"
nbmake = "*"
```

And now let me show you why we need Poetry and not plain `pip`. Poetry lets you specify different depndency versions, and different sources (the flag `--extra-url` rings a bell?) for each dependency.

Imagine we want to install Keras and PyTorch, but we have a Mac, and our friends have Windows and/or Linux. Some of us have a GPU, others don't. These things mean each person will need a different version of these two popular ML packages, from different sources.

How to solve this? As follows:

```toml
[tool.poetry.dependencies]
python = ">=3.8,<3.11"
# ... other dependencies ...
tensorflow-io-gcs-filesystem = [
    { version = "<0.32.0", platform = "win32" },
    { version = "*", platform = "linux" },
    { version = "*", platform = "darwin" },
]
keras = "*"
torch = [
    { version = "^2.0.0", source = "pytorch", platform = "linux" },
    { version = "^2.0.0", source = "pypi", platform = "darwin" },
]

# ... more stuff ...

[[tool.poetry.source]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cu121"
priority = "explicit" # means this URL will be checked for only for the packages where it is explicitly specified
```

What happens here is that we install `tensorflow-io-gcs-filesystem<0.32.0` if we are on Windows (Tensorflow's higher versions do not support Windows at the time of writing), otherwise we install any (`"*"`) version.

Now PyTorch. This package can be painful to install. This is what usually work: install the desired version from PyPi if we are on Mac, install it from `"https://download.pytorch.org"` if we are on Linux. In our example, we chose the GPU version for CUDA 12.1 (see `"/whl/cu121"`).

Now that we have declared our desired dependencies, we need to resolve them. For this, run:

```bash
poetry lock
```

which will produce a `poetry.lock` file.

### requirements.txt

Why not the `requirements.txt`? Poetry finds a platform-independent dependency resolution. If you do `pip install -r requirements.txt` and then `pip freeze > requirements.txt`, you end up with what worked on YOUR MACHINE. When you do `pip freeze > requirements.txt`, you cannot know if `pip` will run successfully on another machine. So please forget about it.

## Testing

We've declared a ton of dependencies in the TOML file. Let's use them. Especially PyTest.

Crate the following file `tests/conftest.py`:

```python
# tests/conftest.py
"""This file is run by PyTest as first file.
Define testing "fixtures" here.
"""
import pytest, os
import typing as ty
import pyrootutils

# Using pyrootutils, we find the root directory of this project and make sure it is our working directory
root = pyrootutils.setup_root(
    search_from=".",
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
    cwd=True,
)

# Example of a fixture, which are values we can pass to all tests
@pytest.fixture(scope="session")
def data_path() -> str:
    """Path where to find data. Reading this value from an environment variable if defined."""
    return os.environ.get("DATA_LOC", ".data")
```

PyTest will load this file before running the tests. We have also called `pyrootutils.setup_root`, which helps us find the root directory of this project, and set that as current working directory.

Now we need to create the tests. The structure of the `tests/` directory should mimic the structure of the `src/` directory. So it is easy to find the test of a specific file in `src/`. Let's create a function, then test it.

We are going to create a neural network, which we will then train. Here, we will only define a general Multi-Layer Perceptron network that can be trained on both continuous tabular data or image data. Here, however, we will only create the neural network architecture, no training procedure will be defined. On that later.

Create this file: `src/project_name/nn.py`

```python
# src/project_name/nn.py

# Use the `__all__` keyword so not to export everything when people import this module
__all__ = ["MLP", "fc_block"]

# stop printing, use this logger, you'll see
from loguru import logger

# always use typing, everything should be clear and explicit
# or not even you will understand your code
from typing import List, Union, Any, Sequence, Tuple, Dict

# now the data science stuff
import numpy as np
import torch

def fc_block(
    in_features: int,
    out_features: int,
    normalize: bool = True,
    batch_norm_eps: float = 0.5,
    leaky_relu: bool = False,
    negative_slope: float = 0.0,
    dropout: bool = False,
) -> List[torch.nn.Module]:
    """Creates a small fully-connected neural block.
    Rather than hardcoding, we can create a general block.
    Each block is just a `torch.nn.Linear` module plus ReLU, normalization, etc.

    Args:
        in_features (int):
            Input dimension.
        out_features (int):
            Output dimension.
        normalize (bool, optional):
            Whether to use Batch 1D normalization. Defaults to True.
        negative_slope (float, optional):
            Negative slope for Leaky ReLU layers. Defaults to 0.0.
        batch_norm_eps (float, optional):
            Epsilon for Batch 1D normalization. Defaults to 0.5.
        dropout (bool, optional):
            Whether to add a Dropout layer.
    Returns:
        List[torch.nn.Module]:
            List of torch modules, to be then turned into a `torch.nn.Sequential` module.
    """
    layers: List[torch.nn.Module] = []
    layers.append(torch.nn.Linear(in_features, out_features))
    if normalize:
        layers.append(torch.nn.BatchNorm1d(out_features, batch_norm_eps))  # type: ignore
    if leaky_relu:
        layers.append(torch.nn.LeakyReLU(negative_slope, inplace=True))  # type: ignore
    else:
        layers.append(torch.nn.ReLU())
    if dropout:
        layers.append(torch.nn.Dropout())
    return list(layers)

class MLP(torch.nn.Module):
    """MLP network. Avoid hardcoding and create a general network.
    By having generalized constructor,
    """

    def __init__(
        self,
        in_features: int,
        out_features: Union[int, Sequence[int]],
        hidden_dims: Sequence[int] = None,
        hidden_size: int = None,
        n_layers: int = 3,
        last_activation: torch.nn.Module = None,
        **kwargs: Any, # inputs for the function above
    ) -> None:
        """
        Args:
            in_features (int):
                Input dimension or shape.
            out_features (Union[int, Sequence[int]]):
                Output dimension or shape. In case you're working with images,
                you may want to pass the image shape: e.g. (C,H,W), which
                stands for (number of color channgels, height in pixels, width in pixels).
            hidden_dims (Sequence[int], optional):
                Sequence of hidden dimensions. Defaults to [].
            hidden_size (int):
                Hidden layers' dimensions. Use either this and `n_layers` or `hidden_dims`.
            n_layers (int):
                Number of hidden layers. Use this in conjunction with `hidden_size` parameter.
            last_activation (torch.nn.Module, optional):
                Last activation for the MLP. Defaults to None.
            **kwargs (optional):
                See function :func:`~fc_block`
        """
        super().__init__()
        # Sanitize
        in_features = int(in_features) # cast to int
        # We now need to create a list of int values
        # If hidden_dims is not provided, we check if hidden_size is
        # If also hidden_size is not provided, we initialize hidden_dims to default value
        # If it is, then we use it with n_layers to create a list of int values
        if hidden_dims is None:
            if hidden_size is None:
                hidden_dims = []
            else:
                hidden_dims = [hidden_size] * n_layers
        else:
            for i, h in enumerate(hidden_dims):
                hidden_dims[i] = int(h)  # type: ignore
        # We now need to make sure that out_features is a list of int
        # As we allow users to input also just an int
        if isinstance(out_features, int):
            out_features = [out_features]
        else:
            for i, h in enumerate(out_features):
                out_features[i] = int(h)  # type: ignore
        self.out_features = out_features
        out_shape = [out_features] if isinstance(out_features, int) else out_features
        # Set up: we create now the list of torch.nn.Modules
        layers = []
        layers_dims = [in_features, *hidden_dims]
        if len(hidden_dims) > 0:
            for i in range(0, len(layers_dims) - 1):
                layers += fc_block(layers_dims[i], layers_dims[i + 1], **kwargs)
            layers.append(torch.nn.Linear(layers_dims[-1], int(np.prod(out_shape))))
        else:
            layers.append(torch.nn.Linear(in_features, int(np.prod(out_shape))))
        if last_activation is not None:
            layers.append(last_activation)
        # Here is our final model
        self.model = torch.nn.Sequential(*layers)
        logger.debug(f"Initialized {self.model}")

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Econdes input tensor to output tensor of predefined shape (see above)."""
        logger.trace(f"input_tensor: {input_tensor.size()}")
        output_tensor: torch.Tensor = self.model(input_tensor)
        logger.trace(f"output_tensor: {output_tensor.size()}")
        return output_tensor
```

That was a lot of code.

As you may also have noticed, there are some `logger.trace()` and `logger.debug()` statements in there. Rather then put in a lot of `print` statements when debugging, and then having to delete them all when we're done. We can leverage Python's logger with the following advantages:

- the print message will also contain the line of code they are coming from;
- they can be left there, you will only see what they print when running in debug mode.

Now, should we test this MLP module? Let's go. Create this file:

```python
# tests/project_name/test_nn.py
import pytest, sys
from loguru import logger
import typing as ty

import torch

from project_name.nn import MLP

def test_mlp_module() -> None:
    """Check network can be initialized, and outputs tensors of expected shape."""
    # Create a neural network consuming inputs of size 100,
    # return a tensor of size 2, going from 100 to 50 to 25 to 10 to 2
    mlp = MLP(100, 2, [50, 25, 10])
    # Create a random input tensor of size 100
    x = torch.rand((100,))
    # Run the MLP, and check output size
    o = mlp(x)
    assert o.numel() == 2, f"Wrong size, expected {2}, got {o.size()}"

if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    pytest.main([__file__, "-x", "-s", "--mypy", "--pylint"])
```

Here is our test. What if we want more of it? Let's parameterize it.

```python
# tests/project_name/test_nn.py
import pytest, sys
from loguru import logger
import typing as ty

import torch

from project_name.nn import MLP

@pytest.mark.parametrize(
    "in_features, out_features, hidden_dims",
    [
        (100, 2, [25, 25, 10]),
        (25, 5, [100]),
    ]
)
def test_mlp_module() -> None:
    """Check network can be initialized, and outputs tensors of expected shape."""
    # Create a neural network
    mlp = MLP(in_features, out_features, hidden_dims)
    # Create a random input tensor of size in_features
    x = torch.rand((in_features,))
    # Run the MLP, and check output size
    o = mlp(x)
    assert o.numel() == out_features, f"Wrong size, expected {out_features}, got {o.size()}"

if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    pytest.main([__file__, "-x", "-s", "--mypy", "--pylint"])
```

Now this is gonna run more than once (twice), each time with a different set of inputs.

To run it, you can simply run this fine `python tests/project_name/test_nn.py`.

## Training

We have tested our neural net behaves as expected. But all it does is just transform an input of a certain shape, to an output of another shape. We now need to train it to solve a task. But we have not defined a training loop. We have to.

Unlike Keras, which is a high-level libray relying on Tensorflow for the Tensor operations, PyTorch is a low-level libray for Deep Learning. The dualism Keras vs PyTorch makes zero sense. While I think you still must learn plain PyTorch, plain PyTorch requires a lot of coding, especially if you want to develop a model that is also easy to use and re-train for other people. Unless you're really experienced, you'd better off using high-level libraries that come with pre-defined building blocks and a clear API that makes your code easy to use for the others.

As said, all we've done is define a MLP architecture, there is no information about how to train it. So now we are going to define a training loop, and we are going to attach all the training procedures to MLP model itself. This way, other people can use it very simply and clearly, by just calling a `.fit()` method.

> PyTorch is my favorite Deep Learning framework, but people's coding skills are, in general, good enough to have fun with PyTorch, but not good enough to produce usable code with it... and plain PyTorch does not focus on code sharing and reproducibilty. And it should not. So we'll use something else. We'll use Lightning.

Let's create a class that not only implements our MLP, but also defines its training loop using Lightning.

Create this file.

```python
# src/project_name/classifier.py
__all__ = ["Classifier"]

from loguru import logger
import typing as ty

import torch
import lightning.pytorch as pl
from torchmetrics import Accuracy, Metric

from .nn import MLP


class Classifier(pl.LightningModule):
    """General classifier, using our MLP module."""

    def __init__(
        self,
        num_classes: int,
        loss: str = "nll",
        lr: float = 1e-2,
        **kwargs: ty.Any,
    ) -> None:
        super().__init__()
        # we create our MLP
        kwargs["out_features"] = num_classes
        self.layers = MLP(**kwargs)
        self.num_classes = num_classes
        # but also a learning rate
        self.lr = lr
        # and a loss function
        task = "multiclass" if num_classes > 2 else "binary"
        self.loss: torch.nn.Module
        if isinstance(loss, str):
            # consider using LogSoftmax with NLLLoss instead of Softmax with CrossEntropyLoss
            if loss.lower() in ["nll", "nllloss", "nl_loss"]:
                self.loss = torch.nn.NLLLoss()
            elif loss.lower() in ["bce", "bceloss", "bce_loss"]:
                self.loss = torch.nn.BCELoss()
                task = "binary"
            else:
                self.loss = torch.nn.NLLLoss()
        elif isinstance(loss, torch.nn.Module):
            self.loss = loss
        else:
            raise TypeError(f"Unrecognized input for loss: {type(loss)}.")
        # we can also define some useful metrics for reporting while training our model
        self.accuracy: Metric = Accuracy(task, num_classes=num_classes)

    def configure_optimizers(self) -> dict:
        """Here is our optimization configuration."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {
            "optimizer": optimizer,
            "scheduler": scheduler,
        }

    def forward(self, x: ty.Union[torch.Tensor, ty.Tuple[torch.Tensor, ty.Any]]) -> torch.Tensor:
        """
        Args:
            x (ty.Union[torch.Tensor, ty.Tuple[torch.Tensor, ty.Any]]):
                Input data. Can be either a tuple of tensors, or a tensor.

        Returns:
            torch.Tensor:
                Output tensor.
        """
        x = x[0] if isinstance(x, (tuple, list)) else x
        assert torch.is_tensor(x), f"x must be a tensor but found of type {type(x)}"  # type: ignore
        x_vectorized = x.view(x.size(0), -1)
        output: torch.Tensor = self.layers(x_vectorized)
        return output

    def training_step(  # type: ignore  # pylint: disable=arguments-differ
        self,
        batch: ty.Tuple[torch.Tensor, torch.Tensor],
        batch_nb: int,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        """
        Args:
            batch (ty.Tuple[torch.Tensor, torch.Tensor]):
                Tuple of (input, label) tensors.
            batch_nb (int):
                Batch ID.

        Returns:
            torch.Tensor:
                Value of the loss function.
        """
        # get data: x (input) and y (label)
        x, y = batch
        output = self(x) # this is our forward pass defined above
        # now we evaluate the loss
        loss: torch.Tensor = self.loss(output, y)
        # we also log useful metrics to monitor our training
        with torch.no_grad():
            preds = torch.argmax(output.detach(), dim=1)
            self.accuracy.update(preds, y)
            self.log("loss/train", loss, prog_bar=True)
            self.log("acc/train", self.accuracy, prog_bar=True)
        return loss
```

We may also define a `validation_step()` and a `test_step()`.

Now, this `Classifier` class not only defined our `MLP` architecture, but also how to train it. To a certain extent, while plain PyTorch is for tensor operations and neural netowrks (in terms of plain achitecture), Lightning allows us to create tasks.

> TO BE CONTINUED
