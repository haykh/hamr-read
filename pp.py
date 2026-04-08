import os
import numpy as np
import hamr_read as pp_c

mytype = np.float32
np.seterr(divide="ignore")


class PostProcessor:
    # AMR field indices (constant across all instances)
    AMR_ACTIVE = 0
    AMR_LEVEL = 1
    AMR_REFINED = 2
    AMR_COORD1 = 3
    AMR_COORD2 = 4
    AMR_COORD3 = 5
    AMR_PARENT = 6
    AMR_CHILD1 = 7
    AMR_CHILD2 = 8
    AMR_CHILD3 = 9
    AMR_CHILD4 = 10
    AMR_CHILD5 = 11
    AMR_CHILD6 = 12
    AMR_CHILD7 = 13
    AMR_CHILD8 = 14
    AMR_NBR1 = 15
    AMR_NBR2 = 16
    AMR_NBR3 = 17
    AMR_NBR4 = 18
    AMR_NBR5 = 19
    AMR_NBR6 = 20
    AMR_NODE = 21
    AMR_POLE = 22
    AMR_GROUP = 23
    AMR_CORN1 = 24
    AMR_CORN2 = 25
    AMR_CORN3 = 26
    AMR_CORN4 = 27
    AMR_CORN5 = 28
    AMR_CORN6 = 29
    AMR_CORN7 = 30
    AMR_CORN8 = 31
    AMR_CORN9 = 32
    AMR_CORN10 = 33
    AMR_CORN11 = 34
    AMR_CORN12 = 35
    AMR_TIMELEVEL = 36
    AMR_LEVEL1 = 110
    AMR_LEVEL2 = 111
    AMR_LEVEL3 = 112
    AMR_NBR1_3 = 113
    AMR_NBR1_4 = 114
    AMR_NBR1_7 = 115
    AMR_NBR1_8 = 116
    AMR_NBR2_1 = 117
    AMR_NBR2_2 = 118
    AMR_NBR2_3 = 119
    AMR_NBR2_4 = 120
    AMR_NBR3_1 = 121
    AMR_NBR3_2 = 122
    AMR_NBR3_5 = 123
    AMR_NBR3_6 = 124
    AMR_NBR4_5 = 125
    AMR_NBR4_6 = 126
    AMR_NBR4_7 = 127
    AMR_NBR4_8 = 128
    AMR_NBR5_1 = 129
    AMR_NBR5_3 = 130
    AMR_NBR5_5 = 131
    AMR_NBR5_7 = 132
    AMR_NBR6_2 = 133
    AMR_NBR6_4 = 134
    AMR_NBR6_6 = 135
    AMR_NBR6_8 = 136
    AMR_NBR1P = 161
    AMR_NBR2P = 162
    AMR_NBR3P = 163
    AMR_NBR4P = 164
    AMR_NBR5P = 165
    AMR_NBR6P = 166

    def __init__(
        self,
        input_dir,
        lowres1=1,
        lowres2=1,
        lowres3=1,
        axisym=0,
        interpolate_var=1,
        do_box=1,
        r_min=1.31224989992,
        r_max=105.0,
        theta_min=-1.0 * np.pi / 180.0,
        theta_max=181.0 * np.pi / 180.0,
        phi_min=-1.0 * np.pi / 180.0,
        phi_max=361.0 * np.pi / 180.0,
        export_raytracing_RAZIEH=0,
        export_raytracing_GRTRANS=0,
        DISK_THICKNESS=0.1,
        check_files=0,
    ):
        self.input_dir = input_dir
        self.lowres1 = lowres1
        self.lowres2 = lowres2
        self.lowres3 = lowres3
        self.axisym = axisym
        self.interpolate_var = interpolate_var
        self.do_box = do_box
        self.r_min = r_min
        self.r_max = r_max
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.phi_min = phi_min
        self.phi_max = phi_max
        self.export_raytracing_RAZIEH = export_raytracing_RAZIEH
        self.export_raytracing_GRTRANS = export_raytracing_GRTRANS
        self.DISK_THICKNESS = DISK_THICKNESS
        self.check_files = check_files

    def _p(self, *parts):
        """Join path parts relative to input_dir."""
        return os.path.join(self.input_dir, *parts)

    def rblock_new(self, dump):
        """Read block/AMR structure from dump directory."""
        dump_grid = self._p("dumps%d" % dump, "grid")
        gdump_grid = self._p("gdumps", "grid")
        if os.path.isfile(dump_grid):
            fin = open(dump_grid, "rb")
            nmax = np.fromfile(fin, dtype=np.int32, count=1, sep="")[0]
            NV = 36
        elif os.path.isfile(gdump_grid):
            fin = open(gdump_grid, "rb")
            size = os.path.getsize(gdump_grid)
            nmax = np.fromfile(fin, dtype=np.int32, count=1, sep="")[0]
            NV = (size - 1) // nmax // 4
        else:
            raise FileNotFoundError("Cannot find grid file for dump %d" % dump)

        self.nmax = nmax
        self.block = np.zeros((nmax, 200), dtype=np.int32, order="C")
        self.n_ord = np.zeros((nmax,), dtype=np.int32, order="C")

        gd = np.fromfile(fin, dtype=np.int32, count=NV * nmax, sep="")
        gd = gd.reshape((NV, nmax), order="F").T
        self.block[:, 0:NV] = gd
        if NV < 170:
            self.block[:, self.AMR_LEVEL1] = gd[:, self.AMR_LEVEL]
            self.block[:, self.AMR_LEVEL2] = gd[:, self.AMR_LEVEL]
            self.block[:, self.AMR_LEVEL3] = gd[:, self.AMR_LEVEL]

        if os.path.isfile(dump_grid):
            i = 0
            for n in range(nmax):
                if self.block[n, self.AMR_ACTIVE] == 1:
                    self.n_ord[i] = n
                    i += 1

        fin.close()

    def rpar_new(self, dump):
        """Read simulation parameters from dump directory."""
        path = self._p("dumps%d" % dump, "parameters")
        if not os.path.isfile(path):
            raise FileNotFoundError("Rpar error: %s not found" % path)

        with open(path, "rb") as fin:

            def read_f64():
                return np.fromfile(fin, dtype=np.float64, count=1, sep="")[0]

            def read_i32():
                return np.fromfile(fin, dtype=np.int32, count=1, sep="")[0]

            self.t = read_f64()
            self.n_active = read_i32()
            self.n_active_total = read_i32()
            self.nstep = read_i32()
            self.Dtd = read_f64()
            self.Dtl = read_f64()
            self.Dtr = read_f64()
            self.dump_cnt = read_i32()
            self.rdump_cnt = read_i32()
            self.dt = read_f64()
            self.failed = read_i32()

            self.bs1 = read_i32()
            self.bs2 = read_i32()
            self.bs3 = read_i32()
            _nmax = read_i32()  # nmax from params, already set from rblock_new
            self.nb1 = read_i32()
            self.nb2 = read_i32()
            self.nb3 = read_i32()

            self.startx1 = read_f64()
            self.startx2 = read_f64()
            self.startx3 = read_f64()
            self._dx1 = read_f64()
            self._dx2 = read_f64()
            self._dx3 = read_f64()
            self.tf = read_f64()
            self.a = read_f64()
            self.gam = read_f64()
            self.cour = read_f64()
            self.Rin = read_f64()
            self.Rout = read_f64()
            self.R0 = read_f64()
            self.density_scale = read_f64()

            for _ in range(13):
                trash = read_i32()
            trash = read_i32()

            self.P_NUM = int(trash >= 1000)
            if self.P_NUM:
                trash -= 1000
            self.TWO_T = int(trash >= 100)
            if self.TWO_T:
                trash -= 100
            self.RESISTIVE = int(trash >= 10)
            if self.RESISTIVE:
                trash -= 10
            self.RAD_M1 = int(trash >= 1)

            read_i32()  # trailing int

            # Recompute grid spacing from physical parameters
            self._dx1 = (np.log(self.Rout) - np.log(self.Rin)) / (self.bs1 * self.nb1)
            fractheta = -self.startx2
            self._dx2 = 2.0 * fractheta / (self.bs2 * self.nb2)
            self._dx3 = 2.0 * np.pi / (self.bs3 * self.nb3)

            self.nb = self.n_active_total
            self.rhor = 1 + (1 - self.a**2) ** 0.5

            self.NODE = np.copy(self.n_ord)
            self.TIMELEVEL = np.copy(self.n_ord)
            self.REF_1 = 1
            self.REF_2 = 1
            self.REF_3 = 1
            self.flag_restore = 0

            size = os.path.getsize(path)
            if size >= 66 * 4 + 3 * self.n_active_total * 4:
                for n in range(self.n_active_total):
                    self.n_ord[n] = read_i32()
                    self.TIMELEVEL[n] = read_i32()
                    self.NODE[n] = read_i32()
            elif size >= 66 * 4 + 2 * self.n_active_total * 4:
                self.flag_restore = 1
                for n in range(self.n_active_total):
                    self.n_ord[n] = read_i32()
                    self.TIMELEVEL[n] = read_i32()

        if self.export_raytracing_RAZIEH == 1:
            if (
                self.bs1 % self.lowres1 != 0
                or self.bs2 % self.lowres2 != 0
                or self.bs3 % self.lowres3 != 0
                or ((self.lowres1 & (self.lowres1 - 1) == 0) and self.lowres1 != 0) != 1
                or ((self.lowres2 & (self.lowres2 - 1) == 0) and self.lowres2 != 0) != 1
                or ((self.lowres3 & (self.lowres3 - 1) == 0) and self.lowres3 != 0) != 1
            ):
                print("For raytracing block size needs to be divisible by lowres!")
            if self.interpolate_var == 0:
                print(
                    "Warning: variable interpolation is highly recommended for raytracing!"
                )

    def rgdump_griddata(self):
        """Read grid geometry from gdumps, assembling into a single uniform grid."""
        n_ord = self.n_ord
        block = self.block

        ACTIVE1 = int(np.max(block[n_ord, self.AMR_LEVEL1])) * self.REF_1
        ACTIVE2 = int(np.max(block[n_ord, self.AMR_LEVEL2])) * self.REF_2
        ACTIVE3 = int(np.max(block[n_ord, self.AMR_LEVEL3])) * self.REF_3

        full1 = int(self.nb1 * (1 + self.REF_1) ** ACTIVE1 * self.bs1)
        full2 = int(self.nb2 * (1 + self.REF_2) ** ACTIVE2 * self.bs2)
        full3 = int(self.nb3 * (1 + self.REF_3) ** ACTIVE3 * self.bs3)

        if (
            full1 % self.lowres1 != 0
            or full2 % self.lowres2 != 0
            or full3 % self.lowres3 != 0
        ):
            print("Incompatible lowres settings in rgdump_griddata")

        gridsizex1 = full1 // self.lowres1
        gridsizex2 = full2 // self.lowres2
        gridsizex3 = full3 // self.lowres3

        self._dx1 = self._dx1 * self.lowres1 / (1.0 + self.REF_1) ** ACTIVE1
        self._dx2 = self._dx2 * self.lowres2 / (1.0 + self.REF_2) ** ACTIVE2
        self._dx3 = self._dx3 * self.lowres3 / (1.0 + self.REF_3) ** ACTIVE3

        if self.do_box == 1:
            i_min = max(
                int(
                    np.int32(
                        (np.log(self.r_min) - (self.startx1 + 0.5 * self._dx1))
                        / self._dx1
                    )
                )
                + 1,
                0,
            )
            i_max = min(
                int(
                    np.int32(
                        (np.log(self.r_max) - (self.startx1 + 0.5 * self._dx1))
                        / self._dx1
                    )
                )
                + 1,
                gridsizex1,
            )
            j_min = max(
                int(
                    np.int32(
                        (
                            (2.0 / np.pi * self.theta_min - 1.0)
                            - (self.startx2 + 0.5 * self._dx2)
                        )
                        / self._dx2
                    )
                )
                + 1,
                0,
            )
            j_max = min(
                int(
                    np.int32(
                        (
                            (2.0 / np.pi * self.theta_max - 1.0)
                            - (self.startx2 + 0.5 * self._dx2)
                        )
                        / self._dx2
                    )
                )
                + 1,
                gridsizex2,
            )
            z_min = max(
                int(
                    np.int32(
                        (self.phi_min - (self.startx3 + 0.5 * self._dx3)) / self._dx3
                    )
                )
                + 1,
                0,
            )
            z_max = min(
                int(
                    np.int32(
                        (self.phi_max - (self.startx3 + 0.5 * self._dx3)) / self._dx3
                    )
                )
                + 1,
                gridsizex3,
            )

            gridsizex1 = i_max - i_min
            gridsizex2 = j_max - j_min
            gridsizex3 = z_max - z_min

            if j_max < j_min or i_max < i_min or z_max < z_min:
                print("Bad box selection")
        else:
            i_min, i_max = 0, gridsizex1
            j_min, j_max = 0, gridsizex2
            z_min, z_max = 0, gridsizex3

        self.nx = gridsizex1
        self.ny = gridsizex2
        self.nz = gridsizex3
        self.bs1new = gridsizex1
        self.bs2new = gridsizex2
        self.bs3new = gridsizex3
        self.gridsizex1 = gridsizex1
        self.gridsizex2 = gridsizex2
        self.gridsizex3 = gridsizex3
        self.i_min = i_min
        self.i_max = i_max
        self.j_min = j_min
        self.j_max = j_max
        self.z_min = z_min
        self.z_max = z_max

        self.x1 = np.zeros(
            (1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order="C"
        )
        self.x2 = np.zeros(
            (1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order="C"
        )
        self.x3 = np.zeros(
            (1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order="C"
        )
        self.r = np.zeros(
            (1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order="C"
        )
        self.h = np.zeros(
            (1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order="C"
        )
        self.ph = np.zeros(
            (1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order="C"
        )

        if self.axisym:
            self.gcov = np.zeros(
                (4, 4, 1, gridsizex1, gridsizex2, 1), dtype=mytype, order="C"
            )
            self.gcon = np.zeros(
                (4, 4, 1, gridsizex1, gridsizex2, 1), dtype=mytype, order="C"
            )
            self.gdet = np.zeros(
                (1, gridsizex1, gridsizex2, 1), dtype=mytype, order="C"
            )
            self.dxdxp = np.zeros(
                (4, 4, 1, gridsizex1, gridsizex2, 1), dtype=mytype, order="C"
            )
        else:
            self.gcov = np.zeros(
                (4, 4, 1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order="C"
            )
            self.gcon = np.zeros(
                (4, 4, 1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order="C"
            )
            self.gdet = np.zeros(
                (1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order="C"
            )
            self.dxdxp = np.zeros(
                (4, 4, 1, gridsizex1, gridsizex2, gridsizex3), dtype=mytype, order="C"
            )

        if self.check_files == 1:
            expected = (
                9 * self.bs1 * self.bs2 * self.bs3
                + (self.bs1 * self.bs2 * 49) * self.axisym
                + (self.bs1 * self.bs2 * self.bs3 * 49) * (self.axisym == 0)
            ) * 8
            for n in range(self.n_active_total):
                path = self._p("gdumps", "gdump%d" % n_ord[n])
                if not os.path.isfile(path) or os.path.getsize(path) != expected:
                    print("Gdump file %d doesn't exist or has wrong size" % n_ord[n])

        size = os.path.getsize(self._p("gdumps", "gdump%d" % n_ord[0]))
        flag = 1 if size == 58 * self.bs3 * self.bs2 * self.bs1 * 8 else 0

        pp_c.rgdump_griddata(
            flag,
            self.interpolate_var,
            self.input_dir,
            self.axisym,
            n_ord,
            self.lowres1,
            self.lowres2,
            self.lowres3,
            self.nb,
            self.bs1,
            self.bs2,
            self.bs3,
            self.x1,
            self.x2,
            self.x3,
            self.r,
            self.h,
            self.ph,
            self.gcov,
            self.gcon,
            self.dxdxp,
            self.gdet,
            block,
            self.nb1,
            self.nb2,
            self.nb3,
            self.REF_1,
            self.REF_2,
            self.REF_3,
            int(np.max(block[n_ord, self.AMR_LEVEL1])),
            int(np.max(block[n_ord, self.AMR_LEVEL2])),
            int(np.max(block[n_ord, self.AMR_LEVEL3])),
            self.startx1,
            self.startx2,
            self.startx3,
            self._dx1,
            self._dx2,
            self._dx3,
            self.export_raytracing_RAZIEH,
            i_min,
            i_max,
            j_min,
            j_max,
            z_min,
            z_max,
        )

    def rdump_griddata(self, dump):
        """Read fluid dump data, assembling into a single uniform grid."""
        gx1 = self.gridsizex1
        gx2 = self.gridsizex2
        gx3 = self.gridsizex3
        n_ord = self.n_ord
        block = self.block

        self.rho = np.zeros((1, gx1, gx2, gx3), dtype=mytype, order="C")
        self.ug = np.zeros((1, gx1, gx2, gx3), dtype=mytype, order="C")
        self.uu = np.zeros((4, 1, gx1, gx2, gx3), dtype=mytype, order="C")
        self.B = np.zeros((4, 1, gx1, gx2, gx3), dtype=mytype, order="C")
        self.bsq = np.zeros((1, gx1, gx2, gx3), dtype=mytype, order="C")

        if self.export_raytracing_RAZIEH:
            Rdot = np.zeros((1, gx1, gx2, gx3), dtype=mytype, order="C")
        else:
            Rdot = np.zeros((1, 1, 1, 1), dtype=mytype, order="C")

        if self.RAD_M1:
            E_rad = np.zeros((1, gx1, gx2, gx3), dtype=mytype, order="C")
            uu_rad = np.zeros((4, 1, gx1, gx2, gx3), dtype=mytype, order="C")
        else:
            E_rad = np.copy(self.ug)
            uu_rad = np.copy(self.uu)

        E = (
            np.zeros((4, 1, gx1, gx2, gx3), dtype=mytype, order="C")
            if self.RESISTIVE
            else self.B
        )

        if self.TWO_T:
            TE = np.zeros((1, gx1, gx2, gx3), dtype=mytype, order="C")
            TI = np.zeros((1, gx1, gx2, gx3), dtype=mytype, order="C")
        else:
            TE = self.rho
            TI = self.rho

        photon_number = (
            np.zeros((1, gx1, gx2, gx3), dtype=mytype, order="C")
            if self.P_NUM
            else self.rho
        )

        flag = 1 if os.path.isfile(self._p("dumps%d" % dump, "new_dump")) else 0

        pp_c.rdump_griddata(
            flag,
            self.interpolate_var,
            np.int32(self.RAD_M1),
            np.int32(self.RESISTIVE),
            self.TWO_T,
            self.P_NUM,
            self.input_dir,
            dump,
            self.n_active_total,
            self.lowres1,
            self.lowres2,
            self.lowres3,
            self.nb,
            self.bs1,
            self.bs2,
            self.bs3,
            self.rho,
            self.ug,
            self.uu,
            self.B,
            E,
            E_rad,
            uu_rad,
            TE,
            TI,
            photon_number,
            self.gcov,
            self.gcon,
            self.axisym,
            n_ord,
            block,
            self.nb1,
            self.nb2,
            self.nb3,
            self.REF_1,
            self.REF_2,
            self.REF_3,
            int(np.max(block[n_ord, self.AMR_LEVEL1])),
            int(np.max(block[n_ord, self.AMR_LEVEL2])),
            int(np.max(block[n_ord, self.AMR_LEVEL3])),
            self.export_raytracing_RAZIEH,
            self.DISK_THICKNESS,
            self.a,
            self.gam,
            Rdot,
            self.bsq,
            self.r,
            self.startx1,
            self.startx2,
            self.startx3,
            self._dx1,
            self._dx2,
            self._dx3,
            self.x1,
            self.x2,
            self.x3,
            self.i_min,
            self.i_max,
            self.j_min,
            self.j_max,
            self.z_min,
            self.z_max,
        )

        self.bs1new = gx1
        self.bs2new = gx2
        self.bs3new = gx3

        if self.do_box == 1:
            self.startx1 += self.i_min * self._dx1
            self.startx2 += self.j_min * self._dx2
            self.startx3 += self.z_min * self._dx3

        self.nb2d = self.nb
        self.nb = 1
        self.nb1 = 1
        self.nb2 = 1
        self.nb3 = 1

    def misc_calc(self, calc_bu=1, calc_bsq=1, calc_eu=0, calc_esq=0):
        """Compute derived magnetic/electric field quantities."""
        self.bu = (
            np.copy(self.uu)
            if calc_bu
            else np.zeros((1, 1, 1, 1, 1), dtype=self.rho.dtype)
        )
        eu = (
            np.copy(self.uu)
            if calc_eu
            else np.zeros((1, 1, 1, 1, 1), dtype=self.rho.dtype)
        )
        self.bsq = (
            np.copy(self.rho)
            if calc_bsq
            else np.zeros((1, 1, 1, 1), dtype=self.rho.dtype)
        )
        esq = (
            np.copy(self.rho)
            if calc_esq
            else np.zeros((1, 1, 1, 1), dtype=self.rho.dtype)
        )

        E = (
            np.zeros(
                (4, 1, self.bs1new, self.bs2new, self.bs3new), dtype=mytype, order="C"
            )
            if self.RESISTIVE
            else self.B
        )

        pp_c.misc_calc(
            self.bs1new,
            self.bs2new,
            self.bs3new,
            self.nb,
            self.axisym,
            self.uu,
            self.B,
            E,
            self.bu,
            eu,
            self.gcov,
            self.bsq,
            esq,
            calc_bu,
            calc_eu,
            calc_bsq,
            calc_esq,
        )
