# usage

install with

```sh
pip install git+https://github.com/haykh/hamr-read.git
```

merge chunked files into `.hamr` using the built-in command line tool:

```sh
hamr_convert --input=<path_to_raw_data> --frame_min=0 --frame_max=1000
```

or equivalently via a python interface

```python
import hamr_read as hr

hr.convert(input="<path_to_raw_data>", frame_min=0, frame_max=1000)
```

read the merged data into an `xarray` dataset:

```python
import hamr_read as hr

data = hr.Data("merged.01000.harm")
```

or you may use raw numpy arrays

```python
import hamr_read as hr

t, a, gam, data = hr.DataNumpy("merged.01000.harm")
```

## accessing the data with `xarray`

`xarray` supports fancy data reduction, e.g.,

```python
(data.rho * np.sin(data.th)).mean(["th", "phi"]).plot()
```

and also 2D plotting (poloidal plane):

```python
data.rho.polar.pcolor(plane="r-th", phi=0.25, norm=mpl.colors.LogNorm(0.1, 200))
# or equivalently
data.rho.sel(phi=0.25, method="nearest").polar.pcolor(norm=mpl.colors.LogNorm(0.1, 200))
```

or in the equatorial plane

```python
data.rho.polar.pcolor(plane="r-phi", norm=mpl.colors.LogNorm(0.1, 200))
```