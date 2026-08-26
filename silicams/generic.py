################################################################################
# Generic Molecules Pack                                                       #
#                                                                              #
"""Factory helpers for small reusable molecule building blocks."""
################################################################################


from silicams.molecule import Molecule


def alkane(length, name="alkane", short="ALK", is_h=True):
    """Build a linear alkane molecule.

    Parameters
    ----------
    length : int
        Number of carbon atoms in the chain.
    name : str, optional
        Molecule name.
    short : str, optional
        Molecule short name.
    is_h : bool, optional
        True to add hydrogens to the carbon skeleton.

    Returns
    -------
    mol : Molecule
        Generated molecule object.
    """
    # Initialize Molecule
    mol = Molecule(name, short)

    # Define bond lengths and angles
    b = {"cc": 0.153, "ch": 0.109}
    a = {"ccc": 30.00, "cch": 109.47}

    # Add carbons
    mol.add("C", [0, 0, 0])

    angle = a["ccc"]
    for i in range(length-1):
        angle *= -1
        mol.add("C", mol.get_num()-1, r=b["cc"], theta=angle)

    # Add hydrogens
    if is_h:
        if length > 1:
            angle = -90
            for i in range(length):
                # Boundary
                if i==0 or i==length-1:
                    for j in range(3):
                        mol.add("H", i, r=b["ch"], theta=angle-30, phi=120*j)
                # Inner
                else:
                    mol.add("H", i, r=b["ch"], theta=angle, phi=a["cch"])
                    mol.add("H", i, r=b["ch"], theta=angle, phi=-a["cch"])

                # Switch orientation
                angle *= -1

        # Methane
        else:
            mol.add("H", 0, r=b["ch"])
            for i in range(3):
                mol.add("H", 0, r=b["ch"], theta=a["cch"], phi=i*120)

    # Move to zero
    mol.zero()

    # Return molecule
    return mol


def alcohol(length, name="alcohol", short="ALC", is_h=True):
    """Build a linear alcohol molecule.

    Parameters
    ----------
    length : int
        Number of carbon atoms in the chain.
    name : str, optional
        Molecule name.
    short : str, optional
        Molecule short name.
    is_h : bool, optional
        True to add hydrogens to the carbon skeleton.

    Returns
    -------
    mol : Molecule
        Generated molecule object.
    """
    # Initialize Molecule
    mol = Molecule(name, short)

    # Define bond lengths and angles
    b = {"cc": 0.153, "ch": 0.109, "co": 0.143, "oh": 0.098}
    a = {"ccc": 30.00, "cch": 109.47, "occ": 30.00, "coh": 109.47}

    # Add carbons
    mol.add("C", [0, 0, 0])

    angle = a["ccc"]
    for i in range(length-1):
        angle *= -1
        mol.add("C", mol.get_num()-1, r=b["cc"], theta=angle)

    # Add hydroxy
    mol.add("O", mol.get_num()-1, r=b["co"], theta=-angle)
    mol.add("H", mol.get_num()-1, r=b["oh"], theta=angle)

    # Add hydrogens
    if is_h:
        if length > 1:
            angle = -90
            for i in range(length):
                # Boundary
                if i==0:
                    for j in range(3):
                        mol.add("H", i, r=b["ch"], theta=angle-30, phi=120*j)
                # Inner
                else:
                    mol.add("H", i, r=b["ch"], theta=angle, phi=a["cch"])
                    mol.add("H", i, r=b["ch"], theta=angle, phi=-a["cch"])

                # Switch orientation
                angle *= -1

        # Methanol
        else:
            for i in range(3):
                mol.add("H", 0, r=b["ch"], theta=a["cch"], phi=i*120)

    # Move to zero
    mol.zero()

    # Return molecule
    return mol


def ketone(length, pos, name="ketone", short="KET", is_h=True):
    """Build a linear ketone molecule.

    Parameters
    ----------
    length : int
        Number of carbon atoms in the chain.
    pos : int
        One-based carbon index that carries the oxygen atom.
    name : str, optional
        Molecule name.
    short : str, optional
        Molecule short name.
    is_h : bool
        True to add hydrogens to the carbon skeleton.

    Returns
    -------
    mol : Molecule
        Generated molecule object.

    Raises
    ------
    ValueError
        Raised when fewer than three carbon atoms are requested.
    """
    # Check input
    if length < 3:
        raise ValueError("Specified length is too small for ketones ...")

    # Initialize Molecule
    mol = Molecule(name, short)

    # Define bond lengths and angles
    b = {"cc": 0.153, "ch": 0.109, "co": 0.123}
    a = {"ccc": 30.00, "cch": 109.47}

    # Add carbons
    mol.add("C", [0, 0, 0])

    angle = a["ccc"]
    for i in range(length-1):
        angle *= -1
        mol.add("C", mol.get_num()-1, r=b["cc"], theta=angle)

    # Add oxygen
    angle = -90 if pos % 2 == 0 else 90
    mol.add("O", pos-1, r=b["co"], theta=angle)

    # Add hydrogens
    if is_h:
        angle = -90
        for i in range(length):
            # Boundary
            if i==0 or i==length-1:
                for j in range(3):
                    mol.add("H", i, r=b["ch"], theta=angle-30, phi=120*j)
            # Inner
            elif not i == pos-1:
                mol.add("H", i, r=b["ch"], theta=angle, phi=a["cch"])
                mol.add("H", i, r=b["ch"], theta=angle, phi=-a["cch"])

            # Switch orientation
            angle *= -1

    # Move to zero
    mol.zero()

    # Return molecule
    return mol


def tms(name="tms", short="TMS", separation=30, is_si=True, is_hydro=True):
    """Build a trimethylsilyl-like surface fragment.

    Parameters
    ----------
    name : str, optional
        Molecule name.
    short : str, optional
        Molecule short name.
    separation : float, optional
        Angular separation of the methyl substituents.
    is_si : bool, optional
        True to keep a terminal silicon atom, False to cap the terminus with a
        carbon atom instead.
    is_hydro : bool, optional
        True to add hydrogens to the methyl groups.

    Returns
    -------
    mol : Molecule
        Generated molecule object.

    Notes
    -----
    This helper returns only the molecular coordinates used during slit
    attachment. It does not provide a ready functionalized-slit topology
    bundle. Full functionalized slit topology export additionally requires an
    explicit :class:`silicams.slit.SilaneTopologyConfig` whose flat ITP uses the
    same atom names as the returned molecule and already satisfies the active
    silica-model charge target for the base ``T3`` fragment.
    """
    # Initialize molecule
    mol = Molecule(name, short)

    # Check silicon
    si = "Si" if is_si else "Ci"
    sio = "sio" if is_si else "co"
    sic = "sic" if is_si else "cc"

    # Define bond lengths and angles
    b = {"sio": 0.155, "sic": 0.186, "ch": 0.109, "co": 0.143, "cc": 0.153}

    # Build silyl chain
    mol.add(si, [0, 0, 0])
    mol.add("O", 0, r=b[sio])
    mol.add(si, 1, r=b[sio])

    # Add methyl
    for i in range(3):
        mol.add("C", 2, r=b[sic], theta=separation+10, phi=120*i)

    if is_hydro:
        # Add hydrogens
        for i in range(3, 5+1):
            for j in range(3):
                mol.add("H", i, r=b["ch"], theta=separation, phi=120*j)

        # If not silicon ending
        for i in range(3):
            if not is_si:
                mol.add("H", 0, r=b["ch"], theta=180-separation, phi=120*i)

    # Move to zero
    mol.zero()

    # Return molecule
    return mol


def silanol(name="sl", short="SL"):
    """Build a silanol surface fragment.

    Parameters
    ----------
    name : str, optional
        Molecule name.
    short : str, optional
        Molecule short name.

    Returns
    -------
    mol : Molecule
        Generated molecule object.
    """
    # Initialize molecule
    mol = Molecule(name, short)

    # Define bonds lengths
    b = {"sio": 0.164, "oh": 0.098}

    # Build molecule
    mol.add("Si",[0, 0, 0])
    mol.add("O", 0, r=b["sio"])
    mol.add("H", 1, r=b["oh"])

    # Move to zero
    mol.zero()

    # Return molecule
    return mol
