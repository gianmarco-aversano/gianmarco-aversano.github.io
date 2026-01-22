---
title: "Ultimate guide for a Machine Learning repository"
categories:
  - blog
tags:
  - machine learning
  - ml
  - ai
  - python
  - repository
---

<!-- [my-template](https://github.com/svnv-svsv-jm/init-new-project) -->

> DISCALIMER: This post was written before the era of LLMs and is based off a repository that was written long, long ago. I have recently updated it, but smells from the past may still be around.

If you landed here it's because you want to set up a new repository for a machine learning (ML) project. And probably are not sure how to do it.

During my career, I've had to chance to learn different tools. Nothing too crazy, I try to follow basic conventions and best practices. But I've realized that, for many, none of this is evident. And I don't want them to struggle like I did, so here I am sharing the solutions I've learned.

As Python is the most popular programming language for ML, we'll use that, which also means that we need to set up everything in a way that also respects Python development best practices.

Let's start from zero here.

> Beware that I'm writing this piece under the hypothesis that you are on Linux/Mac. If you're on Windows, just install [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install): check out [this guide](https://www.omgubuntu.co.uk/how-to-install-wsl2-on-windows-10), too.

The target audience for this post is a little all over the place, you'll find things that are easier and things that are less. Hopefully, I've been clear enough, but you should have at least some familiarity with Python, and know what a YAML file is, what Docker (roughly) is, etc.


<!-- By the end, your project will look something like:

```bash
>> tree .
.
├── Dockerfile
├── LICENSE
├── justfile
├── uv.lock
├── contributing.md
├── README.md
├── docker-compose.yml
├── examples
│   └──notebook.ipynb
├── experiments
│   └── README.md
├── pyproject.toml
├── scripts
│   ├── docker-installation-steps.sh
│   ├── entrypoint.sh
│   ├── git-clean.sh
│   └── pytest.sh
├── src
│   └── project_name
│       ├── __init__.py
└── tests
    ├── conftest.py
    ├── e2e
    ├── integration
    └── unit
``` -->

## Pre-requisites

First off, create a new folder and go into it.

```bash
mkdir new-cool-ml-prok
cd new-cool-ml-prok
```

Neat.

Now open this folder with [VSCode](https://code.visualstudio.com/), which is recommended.

You may also want to install the following VSCode extensions:

- [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python): pretty mandatory. This should also automatically install Pylance.
- [Mypy](https://marketplace.visualstudio.com/items?itemName=ms-python.mypy-type-checker): not only this will force you to code in a readable way, but will often spot bugs early while you're still coding.
- [Ruff](https://github.com/astral-sh/ruff): it will spot, while coding, violations of Python coding best practices. It will help you improve your code quality. It will also format your code.
- [TOML](https://marketplace.visualstudio.com/items?itemName=tamasfe.even-better-toml): in order to have well-colored `.toml` files while editing them (not really needed).

Also, in VSCoce settings, activate the `"Editor: Word Wrap"` option, and other similar ones. This will allow you to visualize correctly even long lines of code.

## Virtual environment

We now need a virtual environment. Check out [uv](https://github.com/astral-sh/uv) and never go back to anything else. Make sure that it is correctly installed.

Now, create a virtual environment with a desired Python version:

```bash
uv python install 3.12
uv venv .venv --python=3.12
source .venv/bin/activate
```

In VSCode, open any `.py` file, then in bottom bar (usually on the bottom right) you should be able to select a Python interpreter. Select the environment you just created. If you can't see it, start typing its name, or restart VSCode.

## Getting started

We need to create the project's metadata. We need to create the `pyproject.toml` file, which is very important. It contains all the project's information.

This file should look something like this:

```toml
# ----------------
# Project
# ----------------
[project]
name = "project-name" # choose a nice project name
version = "0.1.0" # select a version number
description = "Description." # please describe it
authors = ["Name <address@email.com>"]
license = "LICENSE" # make sure this file exists
readme = "README.md" # make sure this file exists

# ----------------
# Build System
# ----------------
[build-system]
requires = ["uv_build>=0.9.9,<0.10.0"]
build-backend = "uv_build"
```

Rather self-explicative. Now create the following files:

- `src/project_name/__init__.py` (package creation file);
- `contributing.md` (you can place any guidelines for how other developers can contribute to your project here).

At this point, your repository looks something like this:

```bash
>> tree .
.
├── LICENSE
├── README.md
├── contributing.md
├── pyproject.toml
├── src
    └── project_name
        └── __init__.py
```

The `README.md` file should contain installation instructions for your package, and how it can be used. Don't be shy to provide examples and/or links to other documentation. Without any of these two things, you may have coded the best thing ever, but it'll be USELESS.

## Dependencies

This is what your `pyproject.toml` should look something like:

```toml
# ----------------
# Build System
# ----------------
[build-system]
requires = ["uv_build>=0.9.9,<0.10.0"]
build-backend = "uv_build"

# ----------------
# Project
# ----------------
[project]
name = "project_name"
version = "0.1.0"
description = ""
authors = [{ name = "Gianmarco Aversano" }]
requires-python = ">=3.10,<3.12"
readme = "README.md"
license = "LICENSE"
dependencies = [
    "pyrootutils",
    "loguru",
    "lightning>=2.0.9.post0,<3",
    "torch",
]

[dependency-groups]
dev = [
    "mypy",
    "ruff",
    "pytest",
    "pytest-cov",
    "pytest-mock",
    "pytest-pylint",
    "pytest-mypy",
    "pytest-testmon",
    "pytest-xdist",
    "nbmake",
]

[tool.uv]

[[tool.uv.index]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cu121"
explicit = true

[tool.uv.sources]
torch = [
    { index = "pytorch", marker = "sys_platform == 'linux'" },
    { index = "pypi", marker = "sys_platform == 'darwin'" },
]
```

And now let me show you why we need `uv` and not plain `pip`. `uv` lets you specify different depndency versions, and different sources (the flag `--extra-url` rings a bell?) for each dependency.

Imagine we want to install PyTorch, but we have a Mac, and our friends have Windows and/or Linux. Some of us have a GPU, others don't. These things mean each person will need a different version of these two popular ML packages, from different sources.

How to solve this? Look above!

PyTorch. This package can be painful to install. This is what usually works: install the desired version from PyPi if we are on Mac, install it from `"https://download.pytorch.org"` if we are on Linux. In our example, we chose the GPU version for CUDA 12.1 (see `"/whl/cu121"`).

Now that we have declared our desired dependencies, we need to resolve them. For this, run:

```bash
uv lock
```

which will produce a `uv.lock` file. This file is our dependency solution. Now, to install the dependencies, run:

```bash
uv sync
```

You will see stuff being installed, but also upgrade or downgraded or uninstalled. This is cool and this command will always sync the dependencies you have currently installed in your virtual environment with the ones declared in the `pyproject.toml` file. This is not supported by plain `pip install -r requirements.txt`.

### ~~requirements.txt~~

Why not the `requirements.txt`? `uv` finds a platform-independent dependency resolution. If you do `pip install -r requirements.txt` and then `pip freeze > requirements.txt`, you end up with what worked on YOUR MACHINE. You cannot know if `pip` will run successfully on another machine. So please forget about it.

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

# Example of a fixture, which are values we can pass to all tests
@pytest.fixture(scope="session")
def resources_path() -> str:
    """Path where to resources for the tests."""
    return os.environ.get("RESOURCES_LOC", "tests/res")
```

PyTest will load this file before running the tests. We have also called `pyrootutils.setup_root`, which helps us find the root directory of this project, and set that as current working directory.

In this file, you can create "fixture", that is variables that can be automatically passed to any test you want. Here, we defined a `data_path` fixture, telling our tests where they can find data, and a `resources_path`, telling our tests where to find resources that can be needed for the tests (text file, images, etc.). You will see later that now we can create tests and if they request an input argument with the same name, PyTest will give it to them (e.g. `def test_blala(data_path: str)`).

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
from typing import Any, Sequence

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
) -> list[torch.nn.Module]:
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
        list[torch.nn.Module]:
            List of torch modules, to be then turned into a `torch.nn.Sequential` module.
    """
    layers: list[torch.nn.Module] = []
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
    Have generalized constructor.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int | Sequence[int],
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
            
            out_features (int | Sequence[int]):
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
    pytest.main([__file__, "-x", "-s"])
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
        (100, 2, [25, 25, 10]), # Run number 0
        (25, 5, [100]), # Run number 1
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
    pytest.main([__file__, "-x", "-s"])
```

Now this is gonna run more than once (twice), each time with a different set of inputs.

To run it, you can simply run this fine `python tests/project_name/test_nn.py`.

### TDD

Please use Test-Driven Development. What is it?

Before touching what's under `src/`, create a simple test that you'd like the new code to pass. The test must be clear and functions as both documentation and proof that that code that you will write will work.

So, write a (small) test, then write enough code to pass it. If the test passes, you cannot write more source code and are forced to improve/extend the test before doing so.

This is a very short introduction to TDD, but if you stick to these principles you'll already write code that "just works". This will also force you to make the right code design choices.

> You can also play a game in order to learn TDD: pair up with a partner code, one of you write a test and the other has to write the code to pass it. The tester will see that there will be multiple source code solutions that can (by)pass their test(s). This will teach both of you how to write tests.

## Training

We have tested our neural net behaves as expected. But all it does is just transform an input of a certain shape, to an output of another shape. We now need to train it to solve a task. But we have not defined a training loop. We have to.

Unlike Keras, which is a high-level libray for the Tensor operations, PyTorch is a low-level libray for Deep Learning. The dualism Keras vs PyTorch makes zero sense. While I think you still must learn plain PyTorch, plain PyTorch requires a lot of coding, especially if you want to develop a model that is also easy to use and re-train for other people. Unless you're really experienced, you'd better off using high-level libraries that come with pre-defined building blocks and a clear API that makes your code easy to use for the others.

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

We may also define a `validation_step()` and a `test_step()`, which will be the same but with `loss/train` replaced by `loss/val` and `loss/test`. Same for `acc/train`.

Now, this `Classifier` class not only defines our `MLP` architecture, but also shows how to train it. To a certain extent, while plain PyTorch is for tensor operations and neural netowrks (in terms of plain achitecture), Lightning allows us to create tasks: the training procedure and inference step of a neural network.

### Testing the training procedure of a neural network

Of course, we can also test that our `Classifier` trains correctly. Here, we will not check that we train the best classifier ever, we will just make sure that the code runs fine, both for training and for inference, and that the loss decreases during training.

Create this file:

```python
# tests/project_name/test_classifier.py
import pytest, sys
from loguru import logger
import typing as ty  # pylint: disable=unused-import

import torch  # pylint: disable=unused-import
import lightning.pytorch as pl
from lightning.pytorch.tuner import Tuner

from project_name.datasets import MNISTDataModule
from torchvision.datasets import MNIST
import torchvision.transforms as tfs

from project_name.models import Classifier

# Remember the "data_path" fixture? Here we use it
# PyTest will pass it to any test that requests it
def test_mnist_classifier(data_path: str) -> None:
    """Test Classifier model can be trained."""
    transforms: tfs.Compose = tfs.Compose(
            [
                tfs.ToTensor(),
                tfs.Normalize((0.1307,), (0.3081,)),
            ]
        )
    dataset = MNIST(self.data_dir, train=True, transform=transforms)
    # datamodule
    datamodule = MNISTDataModule(data_path)
    # model
    model = Classifier()
    # check code runs ok
    pl.Trainer(fast_dev_run=True).fit(model, datamodule)
    # trainer
    trainer = pl.Trainer()
    # metrics before training
    outputs = trainer.validate(model, loader)[0]
    logger.info(outputs)
    loss_start = outputs["loss/val"]
    # find best learning rate
    Tuner(trainer).lr_find(
        model,
        datamodule=datamodule,
        max_lr=1.0,
        min_lr=1e-12,
        update_attr=True,
    )
    # train
    trainer.fit(model, datamodule)
    # metrics after training
    outputs = trainer.validate(model, loader)[0]
    logger.info(outputs)
    loss_end = outputs["loss/val"]
    # test metrics have improved
    logger.info(f"Loss: {loss_end:.3f} < {loss_start:.3f}")
    assert loss_end < loss_start

if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    pytest.main([__file__, "-x", "-s"])
```

Of course, you can test for more, you can also try to overfit one batch. For now, let's keep it like this.

## Running experiments: training models, evaluate them, etc.

Now that we have at least one model available, we can actually train it on a dataset, save it, then load it again and evaluate it. All of these steps can be further developed with little effort now that our model is also tested.

Of course, we could write a notebook and/or a script that imports our model and trains it on some dataset. The problem with this is repoducibility and experiment tracking:

- We want to make sure that the same script runs for different combinations of hyper-parameters, while still remembering what values we chose for them in each run.
- Do hyper-parameter optimization (HPO) out of the box.
- If we can also visualize what's going on while the model is training, that'd be nice.

While I think everyone should learn how to use [MLFlow](https://mlflow.org/docs/latest/index.html) and what it does, I think there is another tool and complements it, which is [Hydra](https://hydra.cc/docs/intro/).

Hydra allows you to create configuration files for your ML experiments. For example, you can create the following file that configures the hyper-parameters for a specific Python class (in our case, it will be ethe `Classifier` class):

```yaml
_target_: project_name.models.Classifier
in_features: 15
num_classes: 2
hidden_dims: [256, 256, 256, 256]
```

We chose some default values. The values are random, we'd need to change them according to whatever we need to run. They can also be overriden on the fly when we run an experiment, or you can manually edit them and run the experiment again.

Either way, Hydra will automatically the configuration being used by the current experiment in a log folder, so you can always go back to it and find it (and run the same experiment again if you want).

For example, take a look at this:

```bash
experiments_logs/<model-name>/<dataset-name>/fit/multiruns/2023-03-22/09-27-28/0
├── .hydra
│   ├── config.yaml
│   ├── hydra.yaml
│   └── overrides.yaml
├── mlflow
│   ├── 0
│   │   └── meta.yaml
│   └── 1
│       ├── 481411259582403785c073586554050d
│       │   ├── meta.yaml
│       │   ├── metrics
│       │   │   ├── accuracy
│       │   │   │   ├── train
│       │   │   │   └── val
│       │   │   ├── auroc
│       │   │   │   ├── train
│       │   │   │   └── val
│       │   │   ├── epoch
│       │   │   ├── loss
│       │   │   │   ├── train
│       │   │   │   └── val
│       │   │   ├── lr-Adam
│       │   │   └── recall
│       │   │       ├── train
│       │   │       └── val
│ # ... The MLFlow stuff is huge, cutting it here
└── tensorboard
    ├── checkpoints
    │   ├── epoch=32-step=3432.ckpt
    │   ├── epoch=34-step=3640.ckpt
    │   ├── epoch=35-step=3744.ckpt
    │   └── last.ckpt
    ├── events.out.tfevents.1679477262.machine.1.0
    └── hparams.yaml
```

With the correct Hydra configuration, I was able to have Hydra creating all of this for each experiment that I ran. Let's break it down.

- `experiments_logs/<model-name>/<dataset-name>/fit/multiruns/2023-03-22/09-27-28/0`: I was able to save my experiments in an `experiments_logs` folder, where then I would go: `<model-name>/<dataset-name>/<fit-or-evaluate>/multiruns/<date>/<time>/<run-id>`, which helped me log as much as I could about each experiment. Why the "multiruns"? You'll see below.
- `.hydra/`: this folder contains the configuration that we used for the run in `config.yaml`, some Hydra-specific configuraiton only in `hydra.yaml`, and any overriden parameter information in `overrides.yaml`.
- `mlflow/`: MLFlow collects a lot of stuff, that it then needs to visualize everything correctly. Many of the things it contains is redundant.
- `tensorboard/`: I was also using [Tensorboard](https://www.tensorflow.org/tensorboard), too. And I was saving my `checkpoints/` in that folder.

As said, Hydra lets you do HPO. Which means that you can set up your config as follows:

```yaml
# @package _global_

defaults: # You can load config from other files, too.
  - extras: default.yaml
  - paths: default.yaml
  - hydra: default.yaml
  - callbacks: default.yaml
  - logger: default.yaml
  - datamodule: mnist.yaml
  - model: classifier.yaml
  - trainer: auto.yaml
  - override hydra/sweeper: optuna
  - override hydra/sweeper/sampler: tpe
  # - override hydra/launcher: ray
  - _self_

hydra:
  mode: MULTIRUN
  sweeper:
    direction: minimize
    n_trials: 30
    n_jobs: 1
    params:
      model.latent_dim: interval(4, 64)
      model.weight_decay: interval(0.001, 0.5)
      model.num_layers: interval(1, 8)
      model.hidden_size: interval(32, 256)
      model.heads: interval(2, 8)
optimize_metric: loss/train

stage: fit
tag: classifier/${get_data_name:${datamodule}}/${stage}
```

Here, you tell Hydra to go `mode: MULTIRUN`, which means it has to create multiple runs of the same experiment, but each time try a different combination of values for the parameters listed under `params:`.
Besides, with the line `- override hydra/sweeper: optuna`, we tell Hydra to use [Optuna](https://optuna.org/), which means that Hydra won't try HP values randomly or do a Cartesian exploration, but will perform Bayesian Optimization (BO).

As you can see, we also indicate `direction: minimize`, meaning that BO will choose the next HP config based on an estimation of where it thinks it will find a better value of the metric we want to optimizer for.

In my config, I indicate this metric as `optimize_metric: loss/train`, but this was a custom keyworkd that I created.

All in all, what this configuraion does is to train the `Classifier` multiple times, each time with a different set of HP values, with the objective of minimizing the final training loss. It will also save and log each run, so that you can re-run it, and tell you which run was the best one.

### The experiment scripts

The above configuration needs to be tied to a Python script, that can consume the configuration and start the training. Place this script in a `experiments/` folder. This script can be something like:

```python
import typing as ty
import pyrootutils
import os
import hydra
from omegaconf import OmegaConf, DictConfig

# Module that contains a basic PyTorch Lightning training loop
from my_project.pipeline import runner
# Hydra/OmegaConf resolvers, see below
from my_project.resolvers import get_data_name, get_model_name, to_int

ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

# I install Hydra/OmegaConf resolvers to create experiment tags
# based on the model I train and the dataset I choose
OmegaConf.register_new_resolver("get_data_name", get_data_name)
OmegaConf.register_new_resolver("get_model_name", get_model_name)
OmegaConf.register_new_resolver("to_int", to_int)

@hydra.main(
    version_base=None,
    config_path=os.path.join(ROOT, "configs"),
    config_name="test",  # change using the flag `--config-name`
)
def main(cfg: DictConfig = None) -> ty.Optional[float]:
    """Train model. You can pass a different configuration from the command line as follows:
    >>> python main.py --config-name <name>
    """
    assert cfg is not None
    # The runner reads the configuration, runs the training and returns
    # a "pipeline" object, just a wrapper around what the runner does
    # so that I can then grab the logged metrics and return the one
    # we want to run HPO for
    pipeline = runner.run(cfg)
    # Grab the metric (e.g. "optimize_metric: loss/train", see above)
    output = pipeline.get_metric_to_optimize()
    return output

if __name__ == "__main__":
    """You can pass a different configuration from the command line as follows:
    >>> python main.py --config-name <name>
    """
    main()
```

The training script (the content of that `runner` module) can look like anything you want, as long as you're able to read the configuration and return the HPO metric.

## Examples and tutorials

As your project grows bigger, you may want to create an `examples/` or `tutorials/` folder some notebooks inside, showcasing the important functionalities of your code.

You should then paste a link to this folder in the repo's `README.md`. (As general guideline, your `README.md` should mention everything (directly or linking to it). Everything that is not somehow mentioned there, does not exist.)

You can also test these notebooks! So you're sure that they run smoothly, as they will probably be the first thing people landing on your repository will try out.

To test them, make sure you have installed `pytest` and `pytest-testmon`, then run:

```bash
pytest --testmon --nbmake --overwrite "./examples"
```

## CI/CD

You may want to use a CI/CD pipeline to automate important steps such as: testing, lint checks, creating releases, creating documentation, publishing your project to Pypi, etc.

This is different depending on whether you're using Gitlab or Github.

On Gitlab, all you have to do is to create a `..gitlab-ci.yml` file at the root directory of your project, then populate this file with keywords that Gitlab understands.

For example:

```yaml
pytest:
  parallel:
    matrix:
      - IMAGE: ["python:3.10", "python:3.11", "python:3.12"]
  image: $IMAGE
  stage: test
  only:
    - merge_requests
  before_script:
    - apt-get update -qy # Update package lists
    - apt-get install -y <anything-you-may-need>
    - pip install uv
    - just install
  script:
    - just test
```

As you can see, this job will run our tests for two Python versions.

But actually, you cannot really see it as the most important commands are hidden behind `just` recipes:

- `just install` to install the project's in the virtual environment;
- `just test` to run the tests.

Using a `justfile` is not strictly mandatory, but it does simplify things, as these installation and test commands may be long and tedious. You'd rather avoid having to write them multiple times. Plus, if for example the installation process changes, you have to remember all the places where it is coded and update them all.

By writing these processes under a `just` recipe, and then calling these recipes rather than those long commands, you can code faster and are less error prone.

For the sake of this example, the following `justfile` is needed:

```justfile
set shell := ["bash", "-c"]
set dotenv-load

default:
    @just --list


# ----------------
# default settings
# ----------------
# project
PROJECT_NAME := "project_name"
EXAMPLE_DIR := "./examples"
LOGS_DIR := "./logs"
# python
PYTHON_EXEC := "uv run"
PYTHONVERSION := "3.12"
ENVNAME := "venv"
COV_FAIL_UNDER := "100"
# docker
IMAGE := PROJECT_NAME


# -----------
# utilities
# -----------
init-directories:
    mkdir -p {{LOGS_DIR}}


# -----------
# install project's dependencies
# -----------
install:
    uv sync

lock:
    uv lock


# -----------
# testing
# -----------
init-tests: init-directories

mypy:
    {{PYTHON_EXEC}} mypy --cache-fine-grained tests
    {{PYTHON_EXEC}} mypy --cache-fine-grained src

ruff:
    {{PYTHON_EXEC}} ruff check --fix .
    {{PYTHON_EXEC}} ruff format .

unit-test: init-tests
    {{PYTHON_EXEC}} pytest -m "not integtest" -x --testmon --junitxml=unit-tests.xml --cov=src/ --cov-fail-under {{COV_FAIL_UNDER}} --cov-report xml:unit-tests-cov.xml

integ-test: init-tests
    {{PYTHON_EXEC}} pytest -m "integtest" -x --testmon --junitxml=integ-tests.xml --cov=src/ --cov-report xml:integ-tests-cov.xml

nbmake: init-tests
    {{PYTHON_EXEC}} pytest --nbmake --overwrite {{EXAMPLE_DIR}}

test: ruff mypy unit-test nbmake

tests: test
```

> The `.env` file should not be committed.

For example, someone may want to replace the `PYTHON_EXEC` variable's value with `poetry run` or `pyenv exec` or `python3 -m`. Whatever floats their boat.

## Docker

Docker is also an important element in a ML repo. Providing a Docker container to run your experiments further helps faciliate reproducibility.

A good enough Docker image for a ML repository may look like this:

```Dockerfile
# Dockerfile
FROM ghcr.io/astral-sh/uv:ubuntu

# Create workdir and copy dependency files
RUN mkdir -p /workdir
COPY . /workdir

# Change shell to be able to easily activate virtualenv
SHELL ["/bin/bash", "-c"]
WORKDIR /workdir

# Install project
RUN apt-get update -qy  &&\
    apt-get install -y apt-utils gosu curl &&\
    curl -sSf https://just.systems/install.sh | bash -s -- --to /usr/local/bin
RUN uv venv .venv --python=3.13 &&\
    source .venv/bin/activate &&\
    just install

# TensorBoard
EXPOSE 6006
# Jupyter Notebook
EXPOSE 8888

# Set entrypoint and default container command
ENTRYPOINT ["/workdir/scripts/entrypoint.sh"]
```

It basically does the same steps as in the CI/CD. Plus, the following:

- exposes some ports that we may use (see below);
- uses a script as entrypoint.

Let's see why.

### docker-compose

We can leverage `docker-compose` to create useful services/containers for your project:

```yaml
# docker-compose.yaml
version: "3.8"

x-common-variables: &common-variables
  LOCAL_USER_ID: ${LOCAL_USER_ID}
  LOCAL_USER: ${LOCAL_USER}

services:
  dev-container:
    image: ${IMAGE}
    container_name: dev-${UNIQUE-0}
    entrypoint: /workdir/scripts/entrypoint.sh
    # Overrides default command so things don't shut down after the process ends.
    command: /bin/sh -c "while sleep 1000; do :; done"
    volumes:
      - ./:/workdir
    environment:
      <<: *common-variables
    runtime: nvidia
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
              device_ids: ["1"]
        limits:
          cpus: 1
          memory: 32G

  notebook:
    image: ${IMAGE}
    container_name: notebook-${UNIQUE-0}
    entrypoint: /workdir/scripts/entrypoint.sh
    command: /${PROJECT_NAME}/bin/python -m jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''
    ports: #server:container
      - ${PORT_JUPY-8888}:8888
    volumes:
      - ./:/workdir
    environment:
      <<: *common-variables
    runtime: nvidia
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
              device_ids: ["1"]
        limits:
          cpus: 1
          memory: 8G

  tensorboard:
    image: ${IMAGE}
    container_name: tensorboard-${UNIQUE-0}
    command: /${PROJECT_NAME}/bin/tensorboard --logdir=. --port=6006 --host 0.0.0.0
    ports:
      - ${PORT_TB-6007}:6006 #server:container
    volumes:
      - ./:/workdir
    environment: *common-variables
    deploy:
      resources:
        limits:
          cpus: 1
          memory: 8G

  mlflow:
    image: ${IMAGE}
    container_name: mlflow-${UNIQUE-0}
    command: bash -c "source /${PROJECT_NAME}/bin/activate && mlflow ui --host 0.0.0.0 --port 5000 --backend-store-uri ${MLFLOW_BACKEND_STORE_URI-file://workdir/lightning_logs}"
    ports:
      - ${PORT_MLFLOW-5002}:5000 #server:container
    volumes:
      - ./:/workdir
    environment: *common-variables
    deploy:
      resources:
        limits:
          cpus: 1
          memory: 8G
```

This long `docker-compose.yaml` file creates:

- A dev container: [see here](https://code.visualstudio.com/docs/devcontainers/containers).
- A Jupyter notebook container, to run code directly on the project's Docker image.
- A Tensorboard and MLFlow container, for ML experiment tracking. These two needs access to a port, which is why we exposed some ports in the `Dockerfile`.

The `docker-compose.yaml` file also gives approriate resources to the containers, and access to GPU (assuming you have one).

Now, what is that entrypoint script?

As you want to run code inside the container, while being able to edit the code and have it updated instantly, in the container, we may want to mount the project's folder on the contaier (at `/workdir`). This is useful also because as we run scripts in the container, those scripts will produce some output files.

This has an issue though. A permission-related one. The container does not have yourself as user. And it should not. It may run as `root` or any other user. When files are created from inside the container, they will not belong to you, but to the container's user.

This will cause painful issues. There are some solutions, but none is as elegant and truly efficient as the following.

You may have noticed that in the `Dockerfile` we `apt-get install -y gosu`. What our entrypoint script does, is to create a new user on the fly, when the container is run. The user that it creates will be the user running the container. Then, it will execute whatever it has to, using `gosu`, under your user ID.

Let's take a look at the entrypoint script:

```bash
#!/bin/bash
# This script is supposed to be run in the Docker image of the project
set -ex
# Add local user: either use the LOCAL_USER_ID if passed in at runtime or fallback
# export $(grep -v '^#' .env | xargs)
DEFAULT_USER=$(whoami)
DEFAULT_ID=$(id -u)
echo "DEFAULT_USER=${DEFAULT_USER}"
USER="${LOCAL_USER:${DEFAULT_USER}}"
USER_ID="${LOCAL_USER_ID:${DEFAULT_ID}}"

echo "USER: $USER -- UID: $USER_ID"
# umask 022 # by default, all newly created files have open permissions
VENV=/venv
ACTIVATE="source $VENV/bin/activate"

# If $USER is empty, pretend to be root
if [[ $USER = "" ]] || [[ -z $USER ]]; then
    USER="$DEFAULT_USER"
    USER_ID="$DEFAULT_ID"
fi

# Check who we are and based on that decide what to do
if [[ $USER = "root" ]]; then
    # If root, just install
    bash -c "$ACTIVATE || echo 'Something went wrong.'"
else
    # If not root, create user (and give them root powers?)
    useradd --shell /bin/bash -u $USER_ID -o -c "" -m $USER
    # echo "$USER ALL=(ALL:ALL) NOPASSWD: ALL" >> /etc/sudoers
    # echo "$USER ALL=(ALL:ALL) NOPASSWD: ALL" | tee /etc/sudoers.d/$USER
    sudo -H -u $USER bash -c 'echo "Running as USER=$USER, with UID=$UID"'
    sudo -H -u $USER bash -c "echo \"$ACTIVATE\" >> \$HOME/.bashrc"
fi

exec gosu $USER "$@"
```

Now, when running docker commands, you can do something like:

```bash
export LOCAL_USER=$(whoami)
export LOCAL_USER_ID=$(id -u)
docker run --rm --network=host --volume $(PWD):/workdir \
    -e LOCAL_USER -e LOCAL_USER_ID \
    -t <image-name> bash <my-command>
```

We pass our username and user ID to the container (the flags `-e LOCAL_USER_ID -e LOCAL_USER`), which will be consumed by the entrypoint script to create this user (ourselves) in the container.

## Conclusions

With this set up, you should now know enough to be able to properly set up your Machine Learning project and have a fruitful collaboration with your fellows.

As this is just a guide, with probably no code snippet that runs out of the box, I also recommend you to take a look at [my working template](https://github.com/svnv-svsv-jm/init-new-project).
