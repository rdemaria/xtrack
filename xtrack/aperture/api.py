import re
import numpy as np
import matplotlib.pyplot as plt

from .structures import Profile


class ApertureAPI:
    @classmethod
    def from_aperture_data(cls, aperture_data):
        model = aperture_data._model
        profiles = ApertureProfiles(model.profiles, model.profile_names)
        types = ApertureTypes(model.types, model.type_names, profiles)
        models = {"b1": Model(aperture_data.line, model.type_positions, types)}
        halo_params = aperture_data.halo_params
        return cls(profiles=profiles, types=types, models=models, halo_params=halo_params)

    def __init__(self, profiles=None, types=None, models=None, halo_params=None):
        self.profiles = profiles
        self.types = types
        self.models = models
        self.halo_params = halo_params

    def __getattr__(self, item):
        if item in self.models:
            return self.models[item]
        else:
            raise AttributeError(f"{item} not found in ApertureAPI")

    def __repr__(self):
        return f"<ApertureAPI: {len(self.profiles.profile_names)} profiles, {len(self.types.type_names)} types, {len(self.models)} model{'s' if len(self.models) != 1 else ''}>"


class ApertureProfiles:
    def __init__(self, profile_data, profile_names):
        self.profile_data = profile_data
        self.profile_names = profile_names
        self.profile_dict = {name: i for i, name in enumerate(self.profile_names)}

    def __getitem__(self, key):
        return ProfileAPI(key, self.profile_data[self.profile_dict[key]])

    def _by_index(self, index):
        return ProfileAPI(self.profile_names[index], self.profile_data[index])

    def __repr__(self):
        return f"<ApertureProfiles: {len(self.profile_names)} profiles>"

    def search(self, regexp):
        pattern = re.compile(regexp)
        return [name for name in self.profile_names if pattern.search(name)]


class ProfileAPI:
    def __init__(self, name, data):
        self.name = name
        self.data = data

    def __getattr__(self, item):
        if item in self._get_attr_names():
            return getattr(self.data.shape, item)
        else:
            raise AttributeError(f"{item} not found in Profile")

    def __repr__(self):
        data = self.data

        return f"<Profile: {self.name!r}: {data.shape} tol={data.tol_r}, {data.tol_x}, {data.tol_y}>"

    def __setattr__(self, name, value):
        if name in ["name", "data"]:
            super().__setattr__(name, value)
        elif name in self._get_attr_names():
            setattr(self.data.shape, name, value)
        else:
            raise AttributeError(f"{name} not found in Profile")

    def _get_attr_names(self):
        return [ff.name for ff in self.data.shape._fields]

    def get_polygon(self, n_points=100, method="uniform"):
        out = np.empty((n_points, 2))
        ## This will use uniform sampling in the parameter space, which may not be ideal for all shapes, but is a good start
        if method == "uniform":
            self.data.build_polygon_for_profile(out, n_points)
        else:
            raise ValueError(f"Unknown method {method}")
        return out

    def plot(self, n_points=100, ax=None, filename=None, method="uniform"):
        x, y = self.get_polygon(n_points, method=method).T
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        ax.plot(x, y, "-")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(f"Aperture Profile: {self.name}")
        ax.grid()
        ax.axis("equal")
        fig.tight_layout()
        if filename is not None:
            fig.savefig(filename)
        return ax


class ApertureType:
    def __init__(self, name, data, _profiles):
        self.name = name
        self.data = data
        self._profiles = _profiles

    def __getitem__(self, item):
        return ProfilePosition(self.data.positions[item], self._profiles)

    @property
    def curvature(self):
        return self.data.curvature

    @curvature.setter
    def curvature(self, value):
        self.data.curvature = value

    def __repr__(self):
        curv = self.data.curvature
        positions = self.data.positions
        return (
            f"<ApertureType: {self.name!r} profiles={len(positions)} curvature={curv}>"
        )

    def plot_xy(self, ax=None, filename=None):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        for pos in self.data.positions:
            profile = self._profiles._by_index(pos.profile_index)
            x, y = profile.get_polygon().T
            ax.plot(x, y, "-", label=f"{profile.name} at s={pos.s_position:.3f} m")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(f"Type: {self.name}")
        ax.grid()
        ax.axis("equal")
        ax.legend()
        fig.tight_layout()
        if filename is not None:
            fig.savefig(filename)
        return ax


class ProfilePosition:
    def __init__(self, data, _profiles):
        self.data = data
        self._profiles = _profiles

    @property
    def profile(self):
        return self._profiles._by_index(self.data.profile_index)

    @property
    def x(self):
        return self.data.shift_x

    @property
    def y(self):
        return self.data.shift_y

    @x.setter
    def x(self, value):
        self.data.shift_x = value

    @y.setter
    def y(self, value):
        self.data.shift_y = value

    @property
    def s(self):
        return self.data.s_position

    @s.setter
    def s(self, value):
        self.data.s_position = value

    @property
    def rot_x(self):
        return self.data.rot_x

    @property
    def rot_y(self):
        return self.data.rot_y

    @property
    def rot_s(self):
        return self.data.rot_s

    @rot_x.setter
    def rot_x(self, value):
        self.data.rot_x = value

    @rot_y.setter
    def rot_y(self, value):
        self.data.rot_y = value

    @rot_s.setter
    def rot_s(self, value):
        self.data.rot_s = value

    def __repr__(self):
        attrs = []
        attrs.append(f"s={self.s}")
        if self.x != 0:
            attrs.append(f"x={self.x}")
        if self.y != 0:
            attrs.append(f"y={self.y}")
        if self.rot_x != 0:
            attrs.append(f"rot_x={self.rot_x}")
        if self.rot_y != 0:
            attrs.append(f"rot_y={self.rot_y}")
        if self.rot_s != 0:
            attrs.append(f"rot_s={self.rot_s}")
        attr_str = ", ".join(attrs)

        return f"<{self.profile.name!r} at {attr_str}>"


class ApertureTypes:
    def __init__(self, type_data, type_names, _profiles):
        self.type_data = type_data
        self.type_names = type_names
        self.type_dict = {name: i for i, name in enumerate(self.type_names)}
        self._profiles = _profiles

    def __getitem__(self, key):
        return ApertureType(key, self.type_data[self.type_dict[key]], self._profiles)

    def _by_index(self, index):
        return ApertureType(self.type_names[index], self.type_data[index], self._profiles)

    def __repr__(self):
        return f"<ApertureTypes: {len(self.type_names)} types>"

    def search(self, regexp):
        pattern = re.compile(regexp)
        return [name for name in self.type_names if pattern.search(name)]


class Model:
    def __init__(self, line, type_positions, _types):
        self.line = line
        self.type_positions = type_positions
        self._types = _types

    def __getitem__(self, item):
        return TypePosition(self.type_positions[item], self._types)

    def __repr__(self): 
        name= self.line.name if self.line.name is not None else f"<Line {id(self.line)}>"
        return f"<Model for line {name}, {len(self.type_positions)} type positions>"



class TypePosition:
    def __init__(self, data, _types):
        self.data = data
        self._types = _types

    @property
    def type(self):
        return self._types._by_index(self.data.type_index)

    @property
    def survey_reference_name(self):
        return self.data.survey_reference_name

    @property
    def transformation(self):
        return Transformation(self.data.transformation)

    def __repr__(self):
        return f"<TypePosition: {self.type.name!r} from {self.survey_reference_name!r}>"


class Transformation:
    def __init__(self, data):
        self.data = data

    @property
    def matrix(self):
        return self.data.to_nplike()

    @property
    def xys(self):
        return self.matrix[0, 3], self.matrix[1, 3], self.matrix[2, 3]

    @property
    def rot_xyz(self):
        rot_x= np.arctan2(self.matrix[2, 1], self.matrix[2, 2])
        rot_y= np.arctan2(-self.matrix[2, 0], np.sqrt(self.matrix[0, 0]**2 + self.matrix[1, 0]**2))
        rot_z= np.arctan2(self.matrix[1, 0], self.matrix[0, 0])
        return rot_x, rot_y, rot_z

    def __repr__(self):
        x, y, s = self.xys
        rot_x, rot_y, rot_s = self.rot_xyz
        attrs = []
        if x != 0:
            attrs.append(f"x={x}")
        if y != 0:
            attrs.append(f"y={y}")
        if s != 0:
            attrs.append(f"s={s}")
        if rot_x != 0:
            attrs.append(f"rot_x={rot_x}")
        if rot_y != 0:
            attrs.append(f"rot_y={rot_y}")
        if rot_s != 0:
            attrs.append(f"rot_s={rot_s}")
        if not attrs:
            return "<Transformation: identity>"
        attr_str = ", ".join(attrs)
        return f"<Transformation: {attr_str}>"

class ApertureBuilder:
    def __init__(self):
        self.type_positions = []
        self.profiles={}
        self.types={}

    def new_profile(self, name, shape, tol_r=0, tol_x=0, tol_y=0, **kwargs):
        shape=(shape, kwargs)
        dct={"shape": shape, "tol_r": tol_r, "tol_x": tol_x, "tol_y": tol_y}
        return Profile(dct)

