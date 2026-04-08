import argparse
import numpy as np

from .pp import PostProcessor


def convert(
    input,
    frame_min,
    frame_max,
    output="merged",
    frame_stride=1,
    r_min=1.31224989992,
    r_max=105.0,
    low_res=1,
    low_res_r=-1,
    low_res_th=-1,
    low_res_ph=-1,
):
    lowres1 = low_res_r if low_res_r > 0 else low_res
    lowres2 = low_res_th if low_res_th > 0 else low_res
    lowres3 = low_res_ph if low_res_ph > 0 else low_res

    pp = PostProcessor(
        input_dir=input,
        lowres1=lowres1,
        lowres2=lowres2,
        lowres3=lowres3,
        axisym=1,
        interpolate_var=1,
        do_box=1,
        r_min=r_min,
        r_max=r_max,
        theta_min=-1.0 * np.pi / 180.0,
        theta_max=181.0 * np.pi / 180.0,
        phi_min=-1.0 * np.pi / 180.0,
        phi_max=361.0 * np.pi / 180.0,
        export_raytracing_RAZIEH=0,
        export_raytracing_GRTRANS=0,
        DISK_THICKNESS=0.1,
        check_files=0,
    )

    for i in range(frame_min, frame_max + 1, frame_stride):
        print(f"Merging {i} into {output}.{i:05d}.harm ...")

        pp.rblock_new(i)
        pp.rpar_new(i)
        pp.rgdump_griddata()
        pp.rdump_griddata(i)
        pp.misc_calc(calc_bu=1, calc_bsq=1)

        x2g = 0.5 * pp.x2 + 0.5
        foo = np.array([1, 1, 0.5, 1], dtype=np.float32)[
            :, np.newaxis, np.newaxis, np.newaxis, np.newaxis
        ]
        uug = foo * pp.uu
        bug = foo * pp.bu

        with open(f"{output}.{i:05d}.harm", "wb") as f:
            dx1 = pp.x1[0, 1, 0, 0] - pp.x1[0, 0, 0, 0]
            dx2g = x2g[0, 0, 1, 0] - x2g[0, 0, 0, 0]
            dx3 = pp.ph[0, 0, 0, 1] - pp.ph[0, 0, 0, 0]

            header = [
                pp.t,
                pp.x1.shape[1],
                pp.x1.shape[2],
                pp.x1.shape[3],
                pp.x1[0, 0, 0, 0] - 0.5 * dx1,
                x2g[0, 0, 0, 0] - 0.5 * dx2g,
                pp.ph[0, 0, 0, 0] - 0.5 * dx3,
                dx1,
                dx2g,
                dx3,
                pp.a,
                pp.gam,
                pp.r[0, 0, 0, 0],
                1.0,
                8,
            ]
            header_str = (
                " ".join(
                    [
                        repr(float(entry) if hasattr(entry, "__float__") else int(entry))
                        for entry in header
                    ]
                )
                + "\n"
            )
            f.write(bytes(header_str, "ascii"))

            data = np.array(
                [
                    pp.x1[0],
                    x2g[0],
                    pp.ph[0],
                    pp.r[0],
                    pp.h[0],
                    pp.ph[0],
                    pp.rho[0],
                    pp.ug[0],
                    uug[0, 0],
                    uug[1, 0],
                    uug[2, 0],
                    uug[3, 0],
                    bug[0, 0],
                    bug[1, 0],
                    bug[2, 0],
                    bug[3, 0],
                ],
                dtype=np.float32,
            ).transpose(1, 2, 3, 0)
            data.tofile(f)

        print(f"Finished merging {i} into {output}.{i:05d}.harm")


def main():
    parser = argparse.ArgumentParser(
        description="Convert HAMR simulation dumps to .harm files."
    )
    parser.add_argument("--input", required=True, help="base name of hamr files to read")
    parser.add_argument(
        "--output", default="merged", help="base name of the *.harm files to write"
    )
    parser.add_argument(
        "--frame_min", type=int, required=True, help="minimum file number to process, inclusive"
    )
    parser.add_argument(
        "--frame_max", type=int, required=True, help="maximum file number to process, inclusive"
    )
    parser.add_argument("--frame_stride", type=int, default=1, help="stride in file numbers")
    parser.add_argument("--r_min", type=float, default=1.31224989992)
    parser.add_argument("--r_max", type=float, default=105.0)
    parser.add_argument("--low_res", type=int, default=1)
    parser.add_argument("--reduced", type=int, default=1)
    parser.add_argument("--low_res_r", type=int, default=-1)
    parser.add_argument("--low_res_th", type=int, default=-1)
    parser.add_argument("--low_res_ph", type=int, default=-1)
    args = parser.parse_args()

    convert(
        input=args.input,
        output=args.output,
        frame_min=args.frame_min,
        frame_max=args.frame_max,
        frame_stride=args.frame_stride,
        r_min=args.r_min,
        r_max=args.r_max,
        low_res=args.low_res,
        low_res_r=args.low_res_r,
        low_res_th=args.low_res_th,
        low_res_ph=args.low_res_ph,
    )
