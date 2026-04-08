from typing import Any
import re
import numpy as np
import xarray as xr


class polar_accessor:
    def __init__(self, xarray_obj) -> None:
        self._obj = xarray_obj

    def pcolor(self, **kwargs) -> Any:
        """
        Plots a pseudocolor plot of 2D polar data on a rectilinear projection.

        Parameters
        ----------
        ax : Axes object, optional
            The axes on which to plot. Default is the current axes.
        plane : str, optional
            The plane to plot ["r-th", "r-phi"]. Default is "r-th".
        phi : float, optional
            The value of phi to plot if `plane` is "r-th". Default is None.
        cbar_size : str, optional
            The size of the colorbar. Default is "5%".
        cbar_pad : float, optional
            The padding between the colorbar and the plot. Default is 0.05.
        cbar_position : str, optional
            The position of the colorbar. Default is "right".
        cbar_ticksize : int or float, optional
            The size of the ticks on the colorbar. Default is None.
        title : str, optional
            The title of the plot. Default is None.
        invert_x : bool, optional
            Whether to invert the x-axis. Default is False.
        invert_y : bool, optional
            Whether to invert the y-axis. Default is False.
        ylabel : str, optional
            The label for the y-axis. Default is "y".
        xlabel : str, optional
            The label for the x-axis. Default is "x".
        label : str, optional
            The label for the plot. Default is None.

        Returns
        -------
        matplotlib.collections.Collection
            The pseudocolor plot.

        Raises
        ------
        AssertionError
            If `ax` is a polar projection or if time is not specified or if data is not 2D polar.

        Notes
        -----
        Additional keyword arguments are passed to `pcolormesh`.
        """

        import matplotlib.pyplot as plt
        from matplotlib import colors
        from matplotlib import tri
        import matplotlib as mpl
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        ax = kwargs.pop("ax", plt.gca())
        if len(self._obj.dims) == 2:
            plane_to_plot = "r-th" if "th" in self._obj.dims else "r-phi"
        else:
            plane_to_plot = kwargs.pop("plane", "r-th")

        cbar_size = kwargs.pop("cbar_size", "5%")
        cbar_pad = kwargs.pop("cbar_pad", 0.05)
        cbar_pos = kwargs.pop("cbar_position", "right")
        cbar_orientation = (
            "vertical" if cbar_pos == "right" or cbar_pos == "left" else "horizontal"
        )
        cbar_ticksize = kwargs.pop("cbar_ticksize", None)
        title = kwargs.pop("title", None)
        invert_x = kwargs.pop("invert_x", False)
        invert_y = kwargs.pop("invert_y", False)
        ylabel = kwargs.pop(
            "ylabel", "$z / r_g$" if plane_to_plot == "r-th" else "$y / r_g$"
        )
        xlabel = kwargs.pop("xlabel", "$x / r_g$")
        label = kwargs.pop("label", None)

        assert ax.name != "polar", "`ax` must be a rectilinear projection"
        assert "t" not in self._obj.dims, "Time must be specified"

        phi_plane = kwargs.pop("phi", None)
        if plane_to_plot == "r-th":
            if len(self._obj.dims) == 3:
                assert (
                    phi_plane is not None
                ), "Phi value must be specified for r-th plane"
                self._obj = self._obj.sel(phi=phi_plane, method="nearest")
            else:
                assert (
                    "phi" not in self._obj.dims
                ), f"Phi dimension must be specified for r-th plane: found {self._obj.dims}"
        elif plane_to_plot == "r-phi":
            if len(self._obj.dims) == 3:
                self._obj = self._obj.sel(th=np.pi / 2, method="nearest")
            else:
                assert (
                    "th" not in self._obj.dims
                ), f"Theta dimension must be specified for r-phi plane: found {self._obj.dims}"
        else:
            raise ValueError(f"Invalid plane: {plane_to_plot}")

        ax.grid(False)
        if type(kwargs.get("norm", None)) is colors.LogNorm:
            cm = kwargs.get("cmap", "viridis")
            cm = mpl.colormaps[cm]
            cm.set_bad(cm(0))
            kwargs["cmap"] = cm

        vals = self._obj.values.T.flatten()
        vals = np.concatenate((vals, vals))

        rs = np.append(self._obj.coords["r_min"], self._obj.coords["r_max"][-1])
        nr = len(rs)

        if plane_to_plot == "r-phi":
            phs = np.append(
                self._obj.coords["phi_min"],
                self._obj.coords["phi_max"][-1],
            )
            nph = len(phs)
            rs, phs = np.meshgrid(rs, phs)
            rs = rs.flatten()
            phs = phs.flatten()
            points_1 = np.arange(nph * nr).reshape(nph, -1)[:-1, :-1].flatten()
            points_2 = np.arange(nph * nr).reshape(nph, -1)[:-1, 1:].flatten()
            points_3 = np.arange(nph * nr).reshape(nph, -1)[1:, 1:].flatten()
            points_4 = np.arange(nph * nr).reshape(nph, -1)[1:, :-1].flatten()

            x, y = rs * np.cos(phs), rs * np.sin(phs)
        elif plane_to_plot == "r-th":
            ths = np.append(
                self._obj.coords["th_min"],
                self._obj.coords["th_max"][-1],
            )
            nth = len(ths)
            rs, ths = np.meshgrid(rs, ths)
            rs = rs.flatten()
            ths = ths.flatten()
            points_1 = np.arange(nth * nr).reshape(nth, -1)[:-1, :-1].flatten()
            points_2 = np.arange(nth * nr).reshape(nth, -1)[:-1, 1:].flatten()
            points_3 = np.arange(nth * nr).reshape(nth, -1)[1:, 1:].flatten()
            points_4 = np.arange(nth * nr).reshape(nth, -1)[1:, :-1].flatten()

            x, y = rs * np.sin(ths), rs * np.cos(ths)
        else:
            raise ValueError(f"Invalid plane: {plane_to_plot}")

        if invert_x:
            x = -x
        if invert_y:
            y = -y
        triang = tri.Triangulation(
            x,
            y,
            triangles=np.concatenate(
                [
                    np.array([points_1, points_2, points_3]).T,
                    np.array([points_1, points_3, points_4]).T,
                ],
                axis=0,
            ),
        )
        ax.set(
            aspect="equal",
            xlabel=xlabel,
            ylabel=ylabel,
        )
        im = ax.tripcolor(triang, vals, rasterized=True, shading="flat", **kwargs)
        if cbar_pos is not None:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes(cbar_pos, size=cbar_size, pad=cbar_pad)
            _ = plt.colorbar(
                im,
                cax=cax,
                label=self._obj.name if label is None else label,
                orientation=cbar_orientation,
            )
            if cbar_orientation == "vertical":
                axis = cax.yaxis
            else:
                axis = cax.xaxis
            axis.set_label_position(cbar_pos)
            axis.set_ticks_position(cbar_pos)
            if cbar_ticksize is not None:
                cax.tick_params("both", labelsize=cbar_ticksize)
        if title is not None:
            ax.set_title(title)
        return im


def DataNumpy(filename):
    with open(filename, "rb") as f:
        line = f.readline().decode("ascii")
        line = re.sub(r"np\.\w+\(([^)]+)\)", r"\1", line)
        header = line.split()

        t = float(header[0])
        N1, N2, N3 = int(header[1]), int(header[2]), int(header[3])
        # not really usable, i guess...
        # x1start, x2start, x3start = float(header[4]), float(header[5]), float(header[6])
        # dx1, dx2, dx3 = float(header[7]), float(header[8]), float(header[9])
        a, gam = float(header[10]), float(header[11])

        data = np.fromfile(f, dtype=np.float32).reshape(N1, N2, N3, 16)

    fields = [
        "x1",
        "x2g",
        "ph",
        "r",
        "h",
        "ph2",
        "rho",
        "ug",
        "uu0",
        "uu1",
        "uu2",
        "uu3",
        "bu0",
        "bu1",
        "bu2",
        "bu3",
    ]
    return t, a, gam, {name: data[..., i] for i, name in enumerate(fields)}


import xarray as xr


@xr.register_dataarray_accessor("polar")
class PolarPlotAccessor(polar_accessor):
    pass


def Data(filename):
    t, a, gam, data = DataNumpy(filename)

    def centers_to_edges_1d(c):
        e = np.empty(len(c) + 1)
        e[1:-1] = 0.5 * (c[:-1] + c[1:])
        e[0] = c[0] - (c[1] - c[0]) / 2
        e[-1] = c[-1] + (c[-1] - c[-2]) / 2
        return e

    r_coord = data["r"][:, 0, 0]
    th_coord = data["h"][0, :, 0]
    phi_coord = data["ph"][0, 0, :]

    re_coord = centers_to_edges_1d(r_coord)
    rmin_coord = re_coord[:-1]
    rmax_coord = re_coord[1:]
    the_coord = centers_to_edges_1d(th_coord)
    the_coord[0] = 0.0
    the_coord[-1] = np.pi
    thmin_coord = the_coord[:-1]
    thmax_coord = the_coord[1:]
    phie_coord = centers_to_edges_1d(phi_coord)
    phimin_coord = phie_coord[:-1]
    phimax_coord = phie_coord[1:]
    return xr.Dataset(
        {
            "rho": (["r", "th", "phi"], data["rho"]),
            "ug": (["r", "th", "phi"], data["ug"]),
            "uu0": (["r", "th", "phi"], data["uu0"]),
            "uu1": (["r", "th", "phi"], data["uu1"]),
            "uu2": (["r", "th", "phi"], data["uu2"]),
            "uu3": (["r", "th", "phi"], data["uu3"]),
            "bu0": (["r", "th", "phi"], data["bu0"]),
            "bu1": (["r", "th", "phi"], data["bu1"]),
            "bu2": (["r", "th", "phi"], data["bu2"]),
            "bu3": (["r", "th", "phi"], data["bu3"]),
        },
        coords={
            "r": ("r", data["r"][:, 0, 0]),
            "th": ("th", data["h"][0, :, 0]),
            "phi": ("phi", data["ph"][0, 0, :]),
            "r_min": ("r", rmin_coord),
            "r_max": ("r", rmax_coord),
            "th_min": ("th", thmin_coord),
            "th_max": ("th", thmax_coord),
            "phi_min": ("phi", phimin_coord),
            "phi_max": ("phi", phimax_coord),
        },
        attrs={"t": t, "a": a, "gam": gam},
    )
