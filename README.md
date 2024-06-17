# sionna-vispy

A [VisPy](https://github.com/vispy/vispy)
backend to preview
[Sionna](https://github.com/NVlabs/sionna) scenes.

## Installation

Simply, use pip:

```bash
pip install sionna-vispy[recommended]
```

or

```bash
pip install sionna-vispy
```

to...

## Usage

Before:

```python
scene.preview(...)
```

After:

```python
with sionna_vispy.patch():
    canvas = scene.preview(...)
    canvas.show()
    canvas.app.run()
```

or simply:

```python
with sionna_vispy.patch():
    scene.preview()
```

inside notebooks[^1].

[^1]: Note that you need `jupyter_rfb` to work inside Jupyter Notebooks.
