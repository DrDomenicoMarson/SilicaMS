################################################################################
# Database Class                                                               #
#                                                                              #
"""Lightweight lookup tables used by the public chemistry helpers."""
################################################################################

# Create masses dictionary
masses = {"H":    1.0079,  # Hydrogen
          "He":   4.0026,  # Helium
          "Li":   6.9410,  # Lithium
          "Be":   9.0122,  # Beryllium
          "B":   10.8110,  # Boron
          "C":   12.0107,  # Carbon
          "N":   14.0067,  # Nitrogen
          "O":   15.9994,  # Oxygen
          "F":   18.9984,  # Fluorine
          "Ne":  20.1797,  # Neon
          "Na":  22.9897,  # Sodium
          "Mg":  24.3050,  # Magnesium
          "Al":  26.9815,  # Aluminum
          "Si":  28.0855,  # Silicon
          "P":   30.9738,  # Phosphorus
          "S":   32.0650,  # Sulfur
          "Cl":  35.4530,  # Chlorine
          "K":   39.0983,  # Potassium
          "Ar":  39.9480,  # Argon
          "Ca":  40.0780,  # Calcium
          "Sc":  44.9559,  # Scandium
          "Ti":  47.8670,  # Titanium
          "V":   50.9415,  # Vanadium
          "Cr":  51.9961,  # Chromium
          "Mn":  54.9380,  # Manganese
          "Fe":  55.8450,  # Iron
          "Ni":  58.6934,  # Nickel
          "Co":  58.9332,  # Cobalt
          "Cu":  63.5460,  # Copper
          "Zn":  65.3900,  # Zinc"
          "Ga":  69.7230,  # Gallium
          "Ge":  72.6400,  # Germanium
          "As":  74.9216,  # Arsenic
          "Se":  78.9600,  # Selenium
          "Br":  79.9040,  # Bromine
          "Kr":  83.8000,  # Krypton
          "Rb":  85.4678,  # Rubidium
          "Sr":  87.6200,  # Strontium
          "Y":   88.9059,  # Yttrium
          "Zr":  91.2240,  # Zirconium
          "Nb":  92.9064,  # Niobium
          "Mo":  95.9400,  # Molybdenum
          "Tc":  98.0000,  # Technetium
          "Ru": 101.0700,  # Ruthenium
          "Rh": 102.9055,  # Rhodium
          "Pd": 106.4200,  # Palladium
          "Ag": 107.8682,  # Silver
          "Cd": 112.4110,  # Cadmium
          "In": 114.8180,  # Indium
          "Sn": 118.7100,  # Tin
          "Sb": 121.7600,  # Antimony
          "I":  126.9045,  # Iodine
          "Te": 127.6000,  # Tellurium
          "Xe": 131.2930,  # Xenon
          "Cs": 132.9055,  # Cesium
          "Ba": 137.3270,  # Barium
          "La": 138.9055,  # Lanthanum
          "Ce": 140.1160,  # Cerium
          "Pr": 140.9077,  # Praseodymium
          "Nd": 144.2400,  # Neodymium
          "Pm": 145.0000,  # Promethium
          "Sm": 150.3600,  # Samarium
          "Eu": 151.9640,  # Europium
          "Gd": 157.2500,  # Gadolinium
          "Tb": 158.9253,  # Terbium
          "Dy": 162.5000,  # Dysprosium
          "Ho": 164.9303,  # Holmium
          "Er": 167.2590,  # Erbium
          "Tm": 168.9342,  # Thulium
          "Yb": 173.0400,  # Ytterbium
          "Lu": 174.9670,  # Lutetium
          "Hf": 178.4900,  # Hafnium
          "Ta": 180.9479,  # Tantalum
          "W":  183.8400,  # Tungsten
          "Re": 186.2070,  # Rhenium
          "Os": 190.2300,  # Osmium
          "Ir": 192.2170,  # Iridium
          "Pt": 195.0780,  # Platinum
          "Au": 196.9665,  # Gold
          "Hg": 200.5900,  # Mercury
          "Tl": 204.3833,  # Thallium
          "Pb": 207.2000,  # Lead
          "Bi": 208.9804,  # Bismuth
          "Po": 209.0000,  # Polonium
          "At": 210.0000,  # Astatine
          "Rn": 222.0000,  # Radon
          "Fr": 223.0000,  # Francium
          "Ra": 226.0000,  # Radium
          "Ac": 227.0000,  # Actinium
          "Pa": 231.0359,  # Protactinium
          "Th": 232.0381,  # Thorium
          "Np": 237.0000,  # Neptunium
          "U":  238.0289,  # Uranium
          "Am": 243.0000,  # Americium
          "Pu": 244.0000,  # Plutonium
          "Cm": 247.0000,  # Curium
          "Bk": 247.0000,  # Berkelium
          "Cf": 251.0000,  # Californium
          "Es": 252.0000,  # Einsteinium
          "Fm": 257.0000,  # Fermium
          "Md": 258.0000,  # Mendelevium
          "No": 259.0000,  # Nobelium
          "Lr": 262.0000,  # Lawrencium
          "Rf": 261.0000,  # Rutherfordium
          "Db": 262.0000,  # Dubnium
          "Sg": 266.0000,  # Seaborgium
          "Bh": 264.0000,  # Bohrium
          "Hs": 277.0000}  # Hassium


# Create covalent radii dictionary in nanometers
covalent_radii = {
    "H": 0.031,
    "C": 0.076,
    "N": 0.071,
    "O": 0.066,
    "F": 0.057,
    "P": 0.107,
    "S": 0.105,
    "Cl": 0.102,
    "Si": 0.111,
}


# Create van der Waals radii dictionary in nanometers
vdw_radii = {
    "H": 0.120,
    "C": 0.170,
    "N": 0.155,
    "O": 0.152,
    "F": 0.147,
    "P": 0.180,
    "S": 0.180,
    "Cl": 0.175,
    "Si": 0.210,
}


_PDB_SINGLE_LETTER_ELEMENTS = {"B", "C", "F", "H", "I", "N", "O", "P", "S"}
_PDB_TWO_LETTER_FALLBACKS = {
    "AL": "Al",
    "BR": "Br",
    "CL": "Cl",
    "CU": "Cu",
    "FE": "Fe",
    "LI": "Li",
    "MG": "Mg",
    "NA": "Na",
    "SI": "Si",
    "ZN": "Zn",
}


##################
# Getter Methods #
##################
def get_mass(symbol):
    """Return the atomic mass for a chemical symbol.

    Parameters
    ----------
    symbol : str
        Chemical symbol to look up.

    Returns
    -------
    mass : float
        Atomic mass in :math:`\\frac{g}{mol}`.

    Raises
    ------
    ValueError
        Raised when ``symbol`` is not present in the local mass table.
    """
    if symbol in masses:
        return masses[symbol]

    raise ValueError("DB: Atom name not found.")


def get_element(symbol):
    """Return the chemical element represented by one atom-type token.

    Parameters
    ----------
    symbol : str
        Atom-type or element token.

    Returns
    -------
    element : str
        Normalized chemical element symbol.

    Raises
    ------
    ValueError
        Raised when no supported chemical element can be derived.
    """
    if symbol in masses:
        return symbol

    letters = "".join(char for char in symbol if char.isalpha())
    if not letters:
        raise ValueError("DB: Atom name not found.")

    if len(letters) >= 2:
        candidate = letters[0].upper() + letters[1].lower()
        if candidate in masses:
            return candidate

    candidate = letters[0].upper()
    if candidate in masses:
        return candidate

    raise ValueError("DB: Atom name not found.")


def get_covalent_radius(symbol):
    """Return the covalent radius of one atom type.

    Parameters
    ----------
    symbol : str
        Atom-type or element token.

    Returns
    -------
    radius : float
        Covalent radius in nanometers.

    Raises
    ------
    ValueError
        Raised when the requested element is not present in the local radius
        table.
    """
    element = get_element(symbol)
    if element in covalent_radii:
        return covalent_radii[element]

    raise ValueError("DB: Covalent radius not found.")


def get_pdb_element(atom_name, element_token=""):
    """Infer one chemical element from PDB atom metadata.

    Parameters
    ----------
    atom_name : str
        PDB atom-name field, such as ``"CA"`` or ``"SI1"``.
    element_token : str, optional
        Explicit PDB element-column token. When provided, it takes priority.

    Returns
    -------
    element : str
        Normalized chemical element symbol.

    Raises
    ------
    ValueError
        Raised when no supported chemical element can be derived.
    """
    token = element_token.strip()
    if token:
        return get_element(token)

    stripped_name = atom_name.strip()
    if not stripped_name:
        raise ValueError("DB: Atom name not found.")

    while stripped_name and stripped_name[0].isdigit():
        stripped_name = stripped_name[1:]
    letters = "".join(char for char in stripped_name if char.isalpha())
    if not letters:
        raise ValueError("DB: Atom name not found.")

    letters_upper = letters.upper()
    if len(letters_upper) >= 2 and letters_upper[:2] in _PDB_TWO_LETTER_FALLBACKS:
        return _PDB_TWO_LETTER_FALLBACKS[letters_upper[:2]]

    if letters_upper[0] in _PDB_SINGLE_LETTER_ELEMENTS:
        return letters_upper[0]

    return get_element(letters)


def get_vdw_radius(symbol):
    """Return the van der Waals radius of one atom type.

    Parameters
    ----------
    symbol : str
        Atom-type or element token.

    Returns
    -------
    radius : float
        Van der Waals radius in nanometers.

    Raises
    ------
    ValueError
        Raised when no supported chemical element can be derived or when the
        resolved element is not present in the local van der Waals table.
    """
    element = get_element(symbol)
    if element in vdw_radii:
        return vdw_radii[element]

    raise ValueError("DB: Atom name not found.")
