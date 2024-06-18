# sionna-vispy

A [VisPy](https://github.com/vispy/vispy)
backend to preview
[Sionna](https://github.com/NVlabs/sionna) scenes
that works both inside and **outside** Jupyter Notebook.

<p align="center">
  <img alt="Example VisPy canvas" width="480" src="https://github.com/jeertmans/sionna-vispy/assets/27275099/40905bb3-8851-43bf-9374-35a331fa0b59">
</p>

This library consists of two parts:

1. a VisPy-based `InteractiveDisplay` to replace `sionna.rt.previewer`;
2. and a `patch()` context manager that dynamically replaces
  the old pythreejs previewer with the new VisPy previewer.

## Installation

For the best out-of-the-box experience, we recommend
installing via Pip with `recommended` extras:

```bash
pip install sionna-vispy[recommended]
```

This will install this package, as well as PySide6 and jupyter-rfb,
so that `scene.preview(...)` works inside and **outside** Jupyter Notebooks.

Alternatively, you can install sionna-vispy with:

```bash
pip install sionna-vispy
```

And later install your preferred
[VisPy backend(s)](https://vispy.org/installation.html).

## Usage

The VisPy scene previewer works both
inside and **outside** Jupyter Notebooks.

First, you need to import the package (import order does not matter):

```python
import sionna_vispy
```

Next, the usage of `patch` depends on the environment,
see the subsections.

> [!NOTE]
> If `with patch():` is called before any
> call to `scene.preview(...)`,
> then you only need to call
> `patch()` once.

### Inside Notebooks[^1]

Very simply, rewrite any

```python
scene.preview(...)
```

with the following:

```python
with sionna_vispy.patch():
    canvas = scene.preview()

canvas
```

> [!WARNING]
> `canvas` must be the return variable
> of the cell, because the `with` context
> does not return an instance of
> `InteractiveDisplay`.

[^1]: Note that you need `jupyter_rfb` to work inside Jupyter Notebooks.

### Outside Notebooks

Canvas need to be *shown* and the VisPy application
must be started to open a window:

```python
with sionna_vispy.patch():
    canvas = scene.preview(...)

canvas.show()
canvas.app.run()
```

## How it works

This package replaces the pythreejs previewer with some
VisPy implementation by
[*monkey-patching*](https://docs.python.org/3/library/unittest.mock.html#unittest.mock.patch)
`sionna.rt.scene.InteractiveDisplay`.

Additionally, `patch()` will (by default) look at
any existing `sionna.rt.scene.Scene` class instance in the local
namespace of the callee, and temporarily replace any
existing preview widget to make sure to use the new previewer. You can
opt-out of this by calling `patch(patch_existing=False)` instead.

## Design goals

This package aims to be a very minimal replacement to the pythreejs
previewer, with maximum compatibility.

As a result, **it does not aim to provide any additional feature**.

Instead, it aims at providing a very similar look to that of
pythreejs, with all the nice features that come with VisPy.

## Contributing

This project welcomes any contribution, and especially:

+ bug fixes;
+ graphical improvements to closely match the original pythreejs previewers;
+ or documentation typos.

As stated above, new features are not expected to be added, unless they are also
added to the original pythreejs previewer.
