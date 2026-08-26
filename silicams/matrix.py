################################################################################
# Matric Class                                                                 #
#                                                                              #
"""Connectivity-matrix helper used during silica-system preparation."""
################################################################################


class Matrix:
    """Store and edit grid connectivity during silica-system preparation.

    Although the search can be parallelized, still multiple iterations are
    needed to cover the surface preparations. Additionally, due to machine
    inaccuracy there is the risk of bonds not being detected as such, leading
    to artefacts. Also, it is not possible to ensure that all bonds were found
    when deleting atoms, because all systems are shaped differently. Therefore,
    another optimization, or rather supporting algorithm, was implemented to
    bypass these issues.

    The idea was reducing the number of iterations to a single search by
    creating a connectivity matrix of all grid atoms. The result is a dictionary
    that has atoms :math:`1\\dots n` as keys and their corresponding value is a
    list of bonded atoms :math:`1\\dots m`

    .. math::

        \\boldsymbol{C}=
        \\begin{Bmatrix}
            a_1:&\\begin{bmatrix}a_{1,1}&a_{1,2}&\\dots&a_{1,m_1}\\end{bmatrix}\\\\
            a_2:&\\begin{bmatrix}a_{2,1}&a_{2,2}&\\dots&a_{2,m_2}\\end{bmatrix}\\\\
            \\vdots&\\vdots\\\\
            a_n:&\\begin{bmatrix}a_{n,1}&a_{n,2}&\\dots&a_{n,m_n}\\end{bmatrix}\\\\
        \\end{Bmatrix}

    Using this implementation, it is no longer required to physically delete
    atoms when carving out a structure, it is enough to remove binding partners
    from the matrix. For example, conditions for the surface preparation only
    need to consider the number of bonds remaining in each entry and thereby
    determine whether an atom needs to be removed or not, resulting into a
    negligible computational effort scaling linear with the number of atoms

    .. math::

      \\mathcal{O}(n).

    In addtition, for further processing, the original amount of bonds is saved
    in the same dictionary.

    Parameters
    ----------
    bonds : list
        Pairwise bond information for the source grid.
    """
    def __init__(self, bonds):
        # Create bond dictionary
        self._matrix = {}
        for bond in bonds:
            # Fill bond as given
            self._matrix[bond[0]] = {"atoms": bond[1], "bonds": len(bond[1])}
            # Fill reverse bonds
            for atom_b in bond[1]:
                if not atom_b in self._matrix:
                    self._matrix[atom_b] = {"atoms": [], "bonds": 0}
                self._matrix[atom_b]["atoms"].append(bond[0])
                self._matrix[atom_b]["bonds"] += 1


    ###########
    # Editing #
    ###########
    def split(self, atom_a, atom_b):
        """Remove the bond between two atoms from the connectivity matrix.

        Parameters
        ----------
        atom_a : int
            First atom id.
        atom_b : int
            Second atom id.
        """
        if atom_a not in self._matrix or atom_b not in self._matrix:
            return
        if atom_b in self._matrix[atom_a]["atoms"]:
            self._matrix[atom_a]["atoms"].remove(atom_b)
        if atom_a in self._matrix[atom_b]["atoms"]:
            self._matrix[atom_b]["atoms"].remove(atom_a)

    def strip(self, atoms):
        """Remove all bonds of a specified atom from the connection matrix.

        Parameters
        ----------
        atoms : list or int
            Atom id or list of atom ids whose bonds should be removed.
        """
        # Porocess input
        atoms = [atoms] if isinstance(atoms, int) else atoms

        # Split all bonds
        for atom_a in atoms:
            if atom_a not in self._matrix:
                continue
            atoms_b = self._matrix[atom_a]["atoms"][:]
            for atom_b in atoms_b:
                self.split(atom_a, atom_b)

    def remove(self, atoms):
        """Remove one atom or several atoms from the active connectivity matrix.

        Parameters
        ----------
        atoms : list or int
            Atom id or list of atom ids that should be disconnected and removed
            from the active matrix.
        """
        atoms = [atoms] if isinstance(atoms, int) else atoms
        self.strip(atoms)
        for atom in atoms:
            if atom in self._matrix:
                del self._matrix[atom]

    def add(self, atom_a, atom_b):
        """Add a bond between two atoms.

        Parameters
        ----------
        atom_a : int
            First atom id.
        atom_b : int
            Second atom id.
        """
        # Add entry fort first atom
        if atom_a in self._matrix.keys():
            self._matrix[atom_a]["atoms"].append(atom_b)
        else:
            self._matrix[atom_a] = {"atoms": [atom_b], "bonds": -1}

        # Add entry for second atom
        if atom_b in self._matrix.keys():
            self._matrix[atom_b]["atoms"].append(atom_a)
        else:
            self._matrix[atom_b] = {"atoms": [atom_a], "bonds": -1}

    def bound(self, num_bonds, logic="eq"):
        """Return atoms matching a bond-count predicate.

        Supported ``logic`` arguments are

        * **eq** - Equals
        * **lt** - Less than
        * **gt** - Greater than

        Parameters
        ----------
        num_bonds : int
            Number of bonds to compare against.
        logic : str, optional
            Comparison operator.

        Returns
        -------
        atoms : list
            Atom ids that satisfy the requested predicate.

        Raises
        ------
        ValueError
            Raised when ``logic`` is not one of ``"eq"``, ``"lt"``, or
            ``"gt"``.
        """
        if logic=="eq":
            return [atom for atom in self._matrix if len(self._matrix[atom]["atoms"])==num_bonds]
        elif logic=="lt":
            return [atom for atom in self._matrix if len(self._matrix[atom]["atoms"])<num_bonds]
        elif logic=="gt":
            return [atom for atom in self._matrix if len(self._matrix[atom]["atoms"])>num_bonds]

        raise ValueError("Matrix: Wrong logic statement...")


    ##################
    # Getter Methods #
    ##################
    def get_matrix(self):
        """Return the connectivity matrix.

        Returns
        -------
        matrix : dict
            Connectivity mapping from atom ids to bonded neighbors and the
            original bond count.
        """
        return self._matrix
