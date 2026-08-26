################################################################################
# Shape Pack                                                                   #
#                                                                              #
"""Analytical shape definitions used to carve pores from silica blocks."""
################################################################################


from dataclasses import dataclass, field, fields
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import silicams.utils as utils
import silicams.geometry as geometry


def _coerce_vector3(name, value):
    """Normalize a three-dimensional coordinate-like input."""
    if len(value) != 3:
        raise ValueError(f"{name} must contain exactly three coordinates.")
    return tuple(value)


def _serialize_config_value(value):
    """Convert config values into list-based serializable representations."""
    if isinstance(value, tuple):
        return list(value)
    return value


@dataclass(frozen=True)
class ShapeConfig:
    """Shared orientation data for analytical pore shapes.

    Parameters
    ----------
    centroid : tuple
        Shape centroid in Cartesian coordinates.
    central : tuple
        Main axis of the shape in Cartesian coordinates.
    """

    centroid: tuple[float, float, float]
    central: tuple[float, float, float]

    def __post_init__(self):
        """Normalize the stored vectors to three-dimensional tuples."""
        object.__setattr__(self, "centroid", _coerce_vector3("centroid", self.centroid))
        object.__setattr__(self, "central", _coerce_vector3("central", self.central))

    def to_dict(self):
        """Return a serializable dict representation of the configuration.

        Returns
        -------
        config : dict
            Dictionary representation of the configuration values.
        """
        return {
            field_info.name: _serialize_config_value(getattr(self, field_info.name))
            for field_info in fields(self)
        }


@dataclass(frozen=True)
class CylinderConfig(ShapeConfig):
    """Configuration for :class:`Cylinder`.

    Parameters
    ----------
    centroid : tuple
        Shape centroid in Cartesian coordinates.
    central : tuple
        Main axis of the cylinder.
    length : float
        Cylinder length in nanometers.
    diameter : float
        Cylinder diameter in nanometers.
    """

    length: float
    diameter: float


@dataclass(frozen=True)
class SphereConfig(ShapeConfig):
    """Configuration for :class:`Sphere`.

    Parameters
    ----------
    centroid : tuple
        Shape centroid in Cartesian coordinates.
    central : tuple
        Main axis used for orientation.
    diameter : float
        Sphere diameter in nanometers.
    """

    diameter: float


@dataclass(frozen=True)
class CuboidConfig(ShapeConfig):
    """Configuration for :class:`Cuboid`.

    Parameters
    ----------
    centroid : tuple
        Shape centroid in Cartesian coordinates.
    central : tuple
        Main axis used for orientation.
    length : float
        Cuboid length in nanometers.
    width : float
        Cuboid width in nanometers.
    height : float
        Cuboid height in nanometers.
    """

    length: float
    width: float
    height: float


@dataclass(frozen=True)
class ConeConfig(ShapeConfig):
    """Configuration for :class:`Cone`.

    Parameters
    ----------
    centroid : tuple
        Shape centroid in Cartesian coordinates.
    central : tuple
        Main axis of the cone.
    length : float
        Cone length in nanometers.
    diameter_1 : float
        Diameter at the first end of the cone in nanometers.
    diameter_2 : float
        Diameter at the second end of the cone in nanometers.
    """

    length: float
    diameter_1: float
    diameter_2: float


@dataclass(frozen=True)
class ShapeSection:
    """Optional coordinate windows used to assign sites to one shape.

    Parameters
    ----------
    x : tuple, optional
        Inclusive ``(min, max)`` range along the x-axis.
    y : tuple, optional
        Inclusive ``(min, max)`` range along the y-axis.
    z : tuple, optional
        Inclusive ``(min, max)`` range along the z-axis.
    """

    x: tuple[float, float] | None = None
    y: tuple[float, float] | None = None
    z: tuple[float, float] | None = None

    def __post_init__(self):
        """Normalize optional axis ranges to tuples."""
        for axis_name in ("x", "y", "z"):
            axis_range = getattr(self, axis_name)
            if axis_range is None:
                continue
            if len(axis_range) != 2:
                raise ValueError(f"ShapeSection.{axis_name} must contain exactly two bounds.")
            object.__setattr__(self, axis_name, tuple(axis_range))

    @classmethod
    def from_dict(cls, section):
        """Build a section specification from a mapping.

        Parameters
        ----------
        section : dict or ShapeSection or None
            Section mapping using ``x``, ``y``, and ``z`` keys.

        Returns
        -------
        shape_section : ShapeSection
            Normalized section dataclass.
        """
        if section is None:
            return cls()
        if isinstance(section, cls):
            return section
        if not isinstance(section, dict):
            raise TypeError("section must be a ShapeSection, dict, or None.")

        kwargs = {}
        for axis_name in ("x", "y", "z"):
            axis_range = section.get(axis_name)
            kwargs[axis_name] = tuple(axis_range) if axis_range else None
        return cls(**kwargs)

    def to_ranges(self, box):
        """Return the normalized axis-range mapping.

        Parameters
        ----------
        box : list
            Simulation-box dimensions used as defaults for omitted axes.

        Returns
        -------
        ranges : dict
            Axis-index mapping used by the pore allocation code.
        """
        axis_ranges = {0: self.x, 1: self.y, 2: self.z}
        return {
            axis_id: list(bounds) if bounds is not None else [0, box[axis_id]]
            for axis_id, bounds in axis_ranges.items()
        }


@dataclass(frozen=True)
class ShapeSpec:
    """Typed shape entry for geometry-specific builders.

    Parameters
    ----------
    shape_type : str
        Shape identifier such as ``"CYLINDER"`` or ``"SLIT"``.
    shape : Shape
        Configured analytical shape object.
    section : ShapeSection, optional
        Optional section constraint used during interior-site assignment.
    """

    shape_type: str
    shape: "Shape"
    section: ShapeSection = field(default_factory=ShapeSection)


def _coerce_shape_config(inp, config_type):
    """Validate that a typed shape configuration was provided."""
    if isinstance(inp, config_type):
        return inp
    raise TypeError(
        f"{config_type.__name__} input must be provided as a {config_type.__name__} instance."
    )


class Shape():
    """Base class for analytical pore shapes.

    Parameters
    ----------
    inp : ShapeConfig
        Shape configuration dataclass.
    """
    def __init__(self, inp):
        self._config = _coerce_shape_config(inp, self._config_type)
        self._inp = self._config.to_dict()

        # Calculate angle and normal vector for rotation
        self._angle = geometry.angle(self._config.central, geometry.main_axis("z"))
        self._normal = geometry.cross_product(self._config.central, geometry.main_axis("z"))

        # Calculate distance towards central axis start
        self._dist_start = geometry.vector(self._centroid, self._config.centroid)
        self._dist_zero = geometry.vector(
            geometry.rotate(self._config.centroid, self._normal, -self._angle, True),
            self._centroid,
        )


    ##################
    # Helper Methods #
    ##################
    def convert(self, data, to_zero=True):
        """Convert coordinates between local and global shape frames.

        Parameters
        ----------
        data : list
            Coordinate-like input to transform.
        to_zero : bool
            True to transform towards the local reference frame, False to
            transform back into the global frame.

        Returns
        -------
        data : list
            Transformed coordinates.
        """
        # Rotate towards main axis to the zero axis
        data = geometry.rotate(data, self._normal, -self._angle if to_zero else self._angle, True)

        # Translate to zero or to start
        dist = self._dist_zero if to_zero else self._dist_start
        data = [data[i]+dist[i] for i in range(3)]

        return data

    def plot(self, inp=0, num=100, vec=None):
        """Plot the surface, rim, and optionally a normal-vector probe.

        Parameters
        ----------
        inp : float, optional
            Position on the main axis used for the rim slice.
        num : int, optional
            Number of points used for the generated surface grid.
        vec : list, optional
            Surface point used to visualize the normal vector.
        """
        fig = plt.gcf()
        ax = fig.add_subplot(111, projection="3d")

        # Surface
        ax.plot_surface(*self.surf(num=100), alpha=0.7)

        # Rim
        ax.plot3D(*[x[0] for x in self.rim(inp, num)])

        # Normal
        if vec:
            line = [self.convert([0, 0, 0], False),
                    vec,
                    self.convert(self.normal(vec), False)]
            ax.plot3D(*utils.column(line))


    ##########
    # Getter #
    ##########
    def get_inp(self):
        """Return the full shape configuration as a serializable dictionary.

        Returns
        -------
        inp : dict
            Dictionary of all shape inputs.
        """
        return self._config.to_dict()

    def get_config(self):
        """Return the typed shape configuration.

        Returns
        -------
        config : ShapeConfig
            Dataclass describing the configured shape geometry.
        """
        return self._config


class Cylinder(Shape):
    """Cylindrical pore shape.

    Required input keys are ``central``, ``centroid``, ``length``, and
    ``diameter``.

    Parameters
    ----------
    inp : CylinderConfig
        Shape configuration dataclass.
    """
    _config_type = CylinderConfig

    def __init__(self, inp):
        config = _coerce_shape_config(inp, self._config_type)
        # Set centroid
        self._centroid = [0, 0, config.length/2]

        # Call super class
        super(Cylinder, self).__init__(config)


    ############
    # Function #
    ############
    def Phi(self, r, phi, z):
        """Surface function of a cylinder

        .. math::

            \\Phi(r,\\phi,z)=
            \\begin{bmatrix}r\\cos(\\phi)\\\\r\\sin(\\phi)\\\\z\\end{bmatrix}

        with radius :math:`r`, polar angle :math:`\\phi` and cylinder length
        :math:`z`.

        Parameters
        ----------
        r : float
            Radius
        phi : float
            Polar angle
        z : float
            Distance ins z-axis

        Returns
        -------
        pos : list
            Cartesian coordinates for given polar coordinates
        """
        x = np.outer(r, np.cos(phi))
        y = np.outer(r, np.sin(phi))
        z = np.outer(z, np.ones(len(z)))

        return self.convert([x, y, z], False)

    def d_Phi_phi(self, r, phi, z):
        """Derivative of the surface function considering the polar angle

        .. math::

            \\frac{\\partial\\Phi}{\\partial\\phi}(r,\\phi,z)=
            \\begin{bmatrix}-r\\sin(\\phi)\\\\r\\cos(\\phi)\\\\0\\end{bmatrix}

        with radius :math:`r`, polar angle :math:`\\phi` and cylinder length
        :math:`z`.

        Parameters
        ----------
        r : float
            Radius
        phi : float
            Polar angle
        z : float
            Distance ins z-axis

        Returns
        -------
        pos : list
            Cartesian coordinates for given polar coordinates
        """
        x = -r*np.sin(phi)
        y = r*np.cos(phi)
        z = 0

        return [x, y, z]

    def d_Phi_z(self, r, phi, z):
        """Derivative of the surface function considering the z-axis

        .. math::

            \\frac{\\partial\\Phi}{\\partial z}(r,\\phi,z)=
            \\begin{bmatrix}0\\\\0\\\\1\\end{bmatrix}

        with radius :math:`r`, polar angle :math:`\\phi` and cylinder length
        :math:`z`.

        Parameters
        ----------
        r : float
            Radius
        phi : float
            Polar angle
        z : float
            Distance ins z-axis

        Returns
        -------
        pos : list
            Cartesian coordinates for given polar coordinates
        """
        x = 0
        y = 0
        z = 1

        return [x, y, z]


    ############
    # Features #
    ############
    def normal(self, pos):
        """Calculate unit normal vector on surface for a given position

        .. math::

            \\frac{\\partial\\Phi}{\\partial\\phi}(r,\\phi,z)\\times
            \\frac{\\partial\\Phi}{\\partial z}(r,\\phi,z)=
            \\begin{bmatrix}r\\cos(\\phi)\\\\r\\sin(\\phi)\\\\0\\end{bmatrix}

        with radius :math:`r`, polar angle :math:`\\phi` and cylinder length
        :math:`z`.

        Parameters
        ----------
        pos : list
            Position

        Returns
        -------
        normal : list
            Normal vector
        """
        # Initialize
        x, y, z = self.convert(pos)

        # Cartesian to polar
        r = math.sqrt(x**2+y**2)
        phi = geometry.angle_polar([x, y, z])

        # Calculate derivatives
        d_Phi_phi = self.d_Phi_phi(r, phi, z)
        d_Phi_z = self.d_Phi_z(r, phi, z)

        # Calculate normal vector
        return geometry.cross_product(d_Phi_phi, d_Phi_z)

    def is_in(self, pos):
        """Check if given position is inside of shape.

        Parameters
        ----------
        pos : list
            Position

        Returns
        -------
        is_in : bool
            True if position is inside of shape
        """
        # Check if within shape
        if geometry.length(self.normal(pos)) < self._inp["diameter"]/2:
            pos_zero = self.convert(pos)
            return pos_zero[2]>0 and pos_zero[2]<self._inp["length"]
        else:
            return False


    #########
    # Shape #
    #########
    def rim(self, z, num=100):
        """Return x and y values for given z-position.

        Parameters
        ----------
        z : float
            Position on the axis
        num : integer, optional
            Number of points

        Returns
        -------
        positions : list
            x and y arrays of the surface rim on the z-position
        """
        phi = np.linspace(0, 2*np.pi, num)
        r = self._inp["diameter"]/2

        return self.Phi(r, phi, [z])

    def surf(self, num=100):
        """Return x, y and z values for the shape.

        Parameters
        ----------
        num : integer, optional
            Number of points

        Returns
        -------
        positions : list
            x, y and z arrays of the surface rim
        """
        phi = np.linspace(0, 2*np.pi, num)
        r = np.ones(num)*self._inp["diameter"]/2
        z = np.linspace(0, self._inp["length"], num)

        return self.Phi(r, phi, z)


    ##############
    # Properties #
    ##############
    def volume(self):
        """Calculate volume

        .. math::

            V=\\pi r^2l

        with radius :math:`r` and cylinder length :math:`l`.

        Returns
        -------
        volume : float
            Volume
        """
        return math.pi*(self._inp["diameter"]/2)**2*self._inp["length"]

    def surface(self):
        """Calculate inner surface

        .. math::

            S=2\\pi rl

        with radius :math:`r` and cylinder length :math:`l`.

        Returns
        -------
        surface : float
            Inner surface
        """
        return 2*math.pi*self._inp["diameter"]/2*self._inp["length"]


class Sphere(Shape):
    """Spherical pore shape.

    Required input keys are ``central``, ``centroid``, and ``diameter``.

    Parameters
    ----------
    inp : SphereConfig
        Shape configuration dataclass.
    """
    _config_type = SphereConfig

    def __init__(self, inp):
        # Set centroid
        self._centroid = [0, 0, 0]

        # Call super class
        super(Sphere, self).__init__(inp)


    ############
    # Function #
    ############
    def Phi(self, r, theta, phi):
        """Surface function of a sphere

        .. math::

            \\Phi(r,\\phi,\\theta)=
            \\begin{bmatrix}r\\cos(\\phi)\\sin(\\theta)\\\\r\\sin(\\phi)\\sin(\\theta)\\\\r\\cos(\\theta)\\end{bmatrix}

        with radius :math:`r`, polar angle :math:`\\phi` and azimuthal angle
        :math:`\\theta`.

        Parameters
        ----------
        r : float
            Radius
        theta : float
            Azimuth angle
        phi : float
            Polar angle

        Returns
        -------
        pos : list
            Cartesian coordinates for given spherical coordinates
        """
        x = r*np.outer(np.cos(phi), np.sin(theta))
        y = r*np.outer(np.sin(phi), np.sin(theta))
        z = r*np.outer(np.ones(len(phi)), np.cos(theta))

        return self.convert([x, y, z], False)

    def d_Phi_phi(self, r, theta, phi):
        """Derivative of the surface function considering the polar angle
        :math:`\\phi`

        .. math::

            \\frac{\\partial\\Phi}{\\partial\\phi}(r,\\phi,\\theta)=
            \\begin{bmatrix}-r\\sin(\\phi)\\sin(\\theta)\\\\r\\cos(\\phi)\\sin(\\theta)\\\\0\\end{bmatrix}

        with radius :math:`r`, polar angle :math:`\\phi` and azimuthal angle
        :math:`\\theta`.

        Parameters
        ----------
        r : float
            Radius
        theta : float
            Azimuth angle
        phi : float
            Polar angle

        Returns
        -------
        pos : list
            Cartesian coordinates for given spherical coordinates
        """
        x = -r*np.sin(phi)*np.sin(theta)
        y = r*np.cos(phi)*np.sin(theta)
        z = 0

        return [x, y, z]

    def d_Phi_theta(self, r, theta, phi):
        """Derivative of the surface function considering the azimuthal angle
        :math:`\\theta`

        .. math::

            \\frac{\\partial\\Phi}{\\partial\\theta}(r,\\phi,\\theta)=
            \\begin{bmatrix}r\\cos(\\phi)\\cos(\\theta)\\\\r\\sin(\\phi)\\cos(\\theta)\\\\-r\\sin(\\theta)\\end{bmatrix}

        with radius :math:`r`, polar angle :math:`\\phi` and azimuthal angle
        :math:`\\theta`.

        Parameters
        ----------
        r : float
            Radius
        theta : float
            Azimuth angle
        phi : float
            Polar angle

        Returns
        -------
        pos : list
            Cartesian coordinates for given spherical coordinates
        """
        x = r*np.cos(phi)*np.cos(theta)
        y = r*np.sin(phi)*np.cos(theta)
        z = -r*np.sin(theta)

        return [x, y, z]


    ############
    # Features #
    ############
    def normal(self, pos):
        """Calculate unit normal vector on surface for a given position

        .. math::

            \\frac{\\partial\\Phi}{\\partial\\theta}(r,\\phi,\\theta)\\times
            \\frac{\\partial\\Phi}{\\partial\\phi}(r,\\phi,\\theta)=
            \\begin{bmatrix}
            -r^2\\cos(\\phi)\\sin(\\theta)^2\\\\
            r^2\\sin(\\phi)\\sin(\\theta)^2\\\\
            -r^2\\sin(\\theta)\\cos(\\theta)\\left[\\sin(\\phi)^2-\\cos(\\phi)^2\\right]\\\\
            \\end{bmatrix}

        with radius :math:`r`, polar angle :math:`\\phi` and cylinder length
        :math:`z`.

        Parameters
        ----------
        pos : list
            Position

        Returns
        -------
        normal : list
            Normal vector
        """
        # Initialize
        x, y, z = self.convert(pos)

        # Cartesian to polar
        r = math.sqrt(x**2+y**2+z**2)
        theta = geometry.angle_azi([x, y, z])
        phi = geometry.angle_polar([x, y, z])

        # Calculate derivatives
        d_Phi_theta = self.d_Phi_theta(r, theta, phi)
        d_Phi_phi = self.d_Phi_phi(r, theta, phi)

        # Calculate normal vector
        return geometry.cross_product(d_Phi_theta, d_Phi_phi)

    def is_in(self, pos):
        """Check if given position is inside of shape.

        Parameters
        ----------
        pos : list
            Position

        Returns
        -------
        is_in : bool
            True if position is inside of shape
        """
        # Check if within shape
        pos_zero = self.convert(pos)
        if geometry.length(geometry.vector(self._centroid, pos_zero)) < self._inp["diameter"]/2:
            return abs(pos_zero[2])<self._inp["diameter"]/2
        else:
            return False


    #########
    # Shape #
    #########
    def rim(self, phi, num=100):
        """Return x and y values for given polar angle.

        Parameters
        ----------
        phi : float
            Position on the axis
        num : integer, optional
            Number of points

        Returns
        -------
        positions : list
            x and y arrays of the surface rim on the z-position
        """
        r = self._inp["diameter"]/2
        theta = np.linspace(0, 2*np.pi, num)

        return self.Phi(r, theta, [phi])

    def surf(self, num=100):
        """Return x, y and z values for the shape.

        Parameters
        ----------
        num : integer, optional
            Number of points

        Returns
        -------
        positions : list
            x, y and z arrays of the surface rim
        """
        r = self._inp["diameter"]/2
        theta = np.linspace(0, np.pi, num)
        phi = np.linspace(0, 2*np.pi, num)

        return self.Phi(r, theta, phi)


    ##############
    # Properties #
    ##############
    def volume(self):
        """Calculate volume

        .. math::

            V=\\frac43\\pi r^3

        with radius :math:`r`.

        Returns
        -------
        volume : float
            Volume
        """
        return 4/3*math.pi*(self._inp["diameter"]/2)**3

    def surface(self):
        """Calculate inner surface

        .. math::

            S=4\\pi r^2

        with radius :math:`r`.

        Returns
        -------
        surface : float
            Inner surface
        """
        return 4*math.pi*(self._inp["diameter"]/2)**2


class Cuboid(Shape):
    """Rectangular slit-like pore shape.

    Required input keys are ``central``, ``centroid``, ``length``, ``width``,
    and ``height``.

    Parameters
    ----------
    inp : CuboidConfig
        Shape configuration dataclass.
    """
    _config_type = CuboidConfig

    def __init__(self, inp):
        config = _coerce_shape_config(inp, self._config_type)
        # Set centroid
        self._centroid = [config.width/2, config.height/2, config.length/2]

        # Call super class
        super(Cuboid, self).__init__(config)


    ############
    # Function #
    ############
    def Phi(self, x, y, z):
        """Surface function of a cuboid.

        Parameters
        ----------
        x : float
            Width
        y : float
            Height
        z : float
            Length

        Returns
        -------
        pos : list
            Cartesian coordinates for given spherical coordinates
        """
        phi = np.arange(1,10,2)*np.pi/4
        Phi, Theta = np.meshgrid(phi, phi)

        x = x*np.cos(Phi)*np.sin(Theta)
        y = y*np.sin(Phi)*np.sin(Theta)
        z = z*np.cos(Theta)/np.sqrt(2)

        return self.convert([x, y, z], False)


    ############
    # Features #
    ############
    def normal(self, pos):
        """Calculate unit normal vector on surface for a given position.

        Parameters
        ----------
        pos : list
            Position

        Returns
        -------
        normal : list
            Unit normal vector
        """
        # Initialize
        x, y, z = self.convert(pos)

        # Calculate derivatives
        return [0, -1, 0] if y < self._centroid[1] else [0, 1, 0]

    def is_in(self, pos):
        """Check if given position is inside of shape.

        Parameters
        ----------
        pos : list
            Position

        Returns
        -------
        is_in : bool
            True if position is inside of shape
        """
        pos_zero = self.convert(pos)

        return pos_zero[1] > 0 and pos_zero[1] < self._inp["height"]


    #########
    # Shape #
    #########
    def rim(self, z, num=100):
        """Return x and y values for given length.

        Parameters
        ----------
        z : float
            Position on the axis
        num : integer, optional
            Number of points

        Returns
        -------
        positions : list
            x and y arrays of the surface rim on the z-position
        """
        x = self._inp["width"]
        y = self._inp["height"]

        return self.Phi(x, y, z)

    def surf(self, num=100):
        """Return x, y and z values for the shape.

        Parameters
        ----------
        num : integer, optional
            Number of points

        Returns
        -------
        positions : list
            x, y and z arrays of the surface rim
        """
        x = self._inp["width"]
        y = self._inp["height"]
        z = self._inp["length"]

        return self.Phi(x, y, z)


    ##############
    # Properties #
    ##############
    def volume(self):
        """Calculate volume

        .. math::

            V=w\\cdot h\\cdot l

        with width :math:`w`, height :math:`h` and length :math:`l`.

        Returns
        -------
        volume : float
            Volume
        """
        return self._inp["length"]*self._inp["width"]*self._inp["height"]

    def surface(self):
        """Calculate inner surface

        .. math::

            S=2\\cdot(w\\cdot h+w\\cdot l+h\\cdot l)

        with width :math:`w`, height :math:`h` and length :math:`l`.

        Returns
        -------
        surface : float
            Inner surface
        """
        return 2*(self._inp["length"]*self._inp["width"]+self._inp["length"]*self._inp["height"]+self._inp["width"]*self._inp["height"])


class Cone(Shape):
    """Conical transition shape.

    Required input keys are ``central``, ``centroid``, ``length``,
    ``diameter_1``, and ``diameter_2``.

    Parameters
    ----------
    inp : ConeConfig
        Shape configuration dataclass.
    """
    _config_type = ConeConfig

    def __init__(self, inp):
        config = _coerce_shape_config(inp, self._config_type)
        # Set centroid
        self._centroid = [0, 0, config.length/2]

        # Call super class
        super(Cone, self).__init__(config)


    ############
    # Function #
    ############
    def Phi(self, r, phi, z):
        """Surface function of a cone

        .. math::

            \\Phi(r(z),\\phi,z)=
            \\begin{bmatrix}r(z)\\cos(\\phi)\\\\r(z)\\sin(\\phi)\\\\z\\end{bmatrix}

        with polar angle :math:`\\phi` and radius function along the :math:`z`-axis

        .. math::

            r(z)=r_1+\\frac{r_2-r_1}{l-1}(z-1)

        with radii :math:`r_1` and :math:`r_2` and cone length :math:`l`.

        Parameters
        ----------
        r : float
            Radius
        phi : float
            Polar angle
        z : float
            Distance ins z-axis

        Returns
        -------
        pos : list
            Cartesian coordinates for given polar coordinates
        """
        def r_func(z):
            z = np.array(z)
            r_1 = self._inp["diameter_1"]/2
            r_2 = self._inp["diameter_2"]/2
            l = self._inp["length"]
            return r_1+(r_2-r_1)/(l-1)*(z-1)

        x = np.outer(r_func(z), np.cos(phi))
        y = np.outer(r_func(z), np.sin(phi))
        z = np.outer(z, np.ones(len(z)))

        return self.convert([x, y, z], False)

    def d_Phi_phi(self, r, phi, z):
        """Derivative of the surface function considering the polar angle

        .. math::

            \\frac{\\partial\\Phi}{\\partial\\phi}(r(z),\\phi,z)=
            \\begin{bmatrix}-r(z)\\sin(\\phi)\\\\r(z)\\cos(\\phi)\\\\0\\end{bmatrix}

        with polar angle :math:`\\phi` and radius function along the :math:`z`-axis

        .. math::

            r(z)=r_1+\\frac{r_2-r_1}{l-1}(z-1)

        with radii :math:`r_1` and :math:`r_2` and cone length :math:`l`.

        Parameters
        ----------
        r : float
            Radius
        phi : float
            Polar angle
        z : float
            Distance ins z-axis

        Returns
        -------
        pos : list
            Cartesian coordinates for given polar coordinates
        """
        def r_func(z):
            r_1 = self._inp["diameter_1"]/2
            r_2 = self._inp["diameter_2"]/2
            l = self._inp["length"]
            return r_1+(r_2-r_1)/(l-1)*(z-1)

        x = -r_func(z)*np.sin(phi)
        y = r_func(z)*np.cos(phi)
        z = 0

        return [x, y, z]

    def d_Phi_z(self, r, phi, z):
        """Derivative of the surface function considering the z-axis

        .. math::

            \\frac{\\partial\\Phi}{\\partial z}(r(z),\\phi,z)=
            \\begin{bmatrix}r(z)\\cos(\\phi)\\\\r(z)\\sin(\\phi)\\\\1\\end{bmatrix}

        with polar angle :math:`\\phi` and radius function along the :math:`z`-axis

        .. math::

            r(z)=\\frac{r_2-r_1}{l-1}

        with radii :math:`r_1` and :math:`r_2` and cone length :math:`l`.

        Parameters
        ----------
        r : float
            Radius
        phi : float
            Polar angle
        z : float
            Distance ins z-axis

        Returns
        -------
        pos : list
            Cartesian coordinates for given polar coordinates
        """
        def r_func(z):
            r_1 = self._inp["diameter_1"]/2
            r_2 = self._inp["diameter_2"]/2
            l = self._inp["length"]
            return (r_2-r_1)/(l-1)

        x = r_func(z)*np.cos(phi)
        y = r_func(z)*np.sin(phi)
        z = 1

        return [x, y, z]


    ############
    # Features #
    ############
    def normal(self, pos):
        """Calculate unit normal vector on surface for a given position

        .. math::

            \\frac{\\partial\\Phi}{\\partial\\phi}(\\tilde r(z),\\phi,z)\\times
            \\frac{\\partial\\Phi}{\\partial z}(\\hat r(z),\\phi,z)=
            \\begin{bmatrix}\\tilde r(z)\\cos(\\phi)\\\\\\tilde r(z)\\sin(\\phi)\\\\\\tilde r(z)\\hat r(z)\\end{bmatrix}

        with polar angle :math:`\\phi` and radius functions along the :math:`z`-axis

        .. math::

            &\\tilde r(z)=r_1+\\frac{r_2-r_1}{l-1}(z-1)\\\\
            &\\hat r(z)=\\frac{r_2-r_1}{l-1},

        with radii :math:`r_1` and :math:`r_2` and cone length :math:`l`.

        Parameters
        ----------
        pos : list
            Position

        Returns
        -------
        normal : list
            Normal vector
        """
        # Initialize
        x, y, z = self.convert(pos)

        # Cartesian to polar
        r = math.sqrt(x**2+y**2)
        phi = geometry.angle_polar([x, y, z])

        # Calculate derivatives
        d_Phi_phi = self.d_Phi_phi(r, phi, z)
        d_Phi_z = self.d_Phi_z(r, phi, z)

        # Calculate normal vector
        return geometry.cross_product(d_Phi_phi, d_Phi_z)

    def is_in(self, pos):
        """Check if given position is inside of shape.

        Parameters
        ----------
        pos : list
            Position

        Returns
        -------
        is_in : bool
            True if position is inside of shape
        """
        def r_func(z):
            r_1 = self._inp["diameter_1"]/2
            r_2 = self._inp["diameter_2"]/2
            l = self._inp["length"]
            return r_1+(r_2-r_1)/(l-1)*(z-1)


        # Check if within shape
        pos_zero = self.convert(pos)
        length = geometry.length(geometry.cross_product(self._inp["central"], geometry.vector([0, 0, 0], pos_zero)))/geometry.length(self._inp["central"])

        if length < r_func(pos_zero[2]):
            return pos_zero[2]>0 and pos_zero[2]<self._inp["length"]
        else:
            return False


    #########
    # Shape #
    #########
    def rim(self, z, num=100):
        """Return x and y values for given z-position.

        Parameters
        ----------
        z : float
            Position on the axis
        num : integer, optional
            Number of points

        Returns
        -------
        positions : list
            x and y arrays of the surface rim on the z-position
        """
        phi = np.linspace(0, 2*np.pi, num)
        r = self._inp["diameter_1"]/2

        return self.Phi(r, phi, [z])

    def surf(self, num=100):
        """Return x, y and z values for the shape.

        Parameters
        ----------
        num : integer, optional
            Number of points

        Returns
        -------
        positions : list
            x, y and z arrays of the surface rim
        """
        phi = np.linspace(0, 2*np.pi, num)
        r = np.linspace(self._inp["diameter_1"]/2, self._inp["diameter_2"]/2, num)
        z = np.linspace(0, self._inp["length"], num)

        return self.Phi(r, phi, z)


    ##############
    # Properties #
    ##############
    def volume(self):
        """Calculate volume

        .. math::

            V=\\frac{1}{3}\\pi \\left[r_1^2+r_2^2+r_1r_2\\right]l

        with radii :math:`r_1` and :math:`r_2` and cone length :math:`l`.

        Returns
        -------
        volume : float
            Volume
        """
        r_1 = self._inp["diameter_1"]/2
        r_2 = self._inp["diameter_2"]/2
        l = self._inp["length"]
        return 1/3*math.pi*(r_1**2+r_2**2+r_1*r_2)*l

    def surface(self):
        """Calculate inner surface

        .. math::

            S=\\pi (r_1+r_2)l

        with radii :math:`r_1` and :math:`r_2` and cone length :math:`l`.

        Returns
        -------
        surface : float
            Inner surface
        """
        r_1 = self._inp["diameter_1"]/2
        r_2 = self._inp["diameter_2"]/2
        l = self._inp["length"]
        return math.pi*(r_1+r_2)*math.sqrt((r_1-r_2)**2+l**2)
