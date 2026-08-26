"""Antechamber and ``tleap`` helper-file writer."""

from __future__ import annotations

import os
import shutil

from .. import utils
from ..molecule import Molecule


class AntechamberWriter:
    """Write Antechamber helper files for one standalone molecule.

    Parameters
    ----------
    molecule : Molecule
        Molecule for which helper files should be generated.
    output_dir : str or os.PathLike, optional
        Directory receiving generated files.
    """

    def __init__(self, molecule, output_dir="./"):
        """Initialize a helper writer for one molecule.

        Parameters
        ----------
        molecule : Molecule
            Molecule for which helper files should be generated.
        output_dir : str or os.PathLike, optional
            Directory receiving generated files.

        Raises
        ------
        TypeError
            Raised when ``molecule`` is not a :class:`Molecule`.
        """
        if not isinstance(molecule, Molecule):
            raise TypeError("AntechamberWriter requires a Molecule instance.")
        self._molecule = molecule
        self._output_dir = os.fspath(output_dir)
        utils.mkdirp(self._output_dir)

    def write(self, name="", master=""):
        """Write matching shell-job and ``tleap`` input files.

        Parameters
        ----------
        name : str, optional
            Base output name. Defaults to the molecule name.
        master : str, optional
            Existing or new master shell filename to which execution commands
            should be appended.

        Returns
        -------
        paths : tuple[str, str]
            Generated job and ``tleap`` paths.
        """
        molecule_name = self._molecule.get_name() or "molecule"
        short_name = self._molecule.get_short()
        output_name = name or molecule_name
        package_dir = os.path.dirname(os.path.dirname(__file__))
        template_dir = os.path.join(package_dir, "templates")

        job_path = os.path.join(self._output_dir, output_name + ".job")
        tleap_path = os.path.join(self._output_dir, output_name + ".tleap")
        shutil.copy(os.path.join(template_dir, "antechamber.job"), job_path)
        shutil.copy(os.path.join(template_dir, "antechamber.tleap"), tleap_path)

        utils.replace(job_path, "MOLNAME", molecule_name.lower())
        utils.replace(tleap_path, "MOLSHORTLOWER", molecule_name.lower())
        utils.replace(tleap_path, "MOLSHORT", short_name)
        utils.replace(tleap_path, "MOLNAME", molecule_name.lower())

        if master:
            master_path = os.path.join(self._output_dir, master)
            with open(master_path, "a", encoding="utf-8") as file_out:
                file_out.write(f"cd {molecule_name.lower()} #\n")
                file_out.write(f"sh {molecule_name.lower()}.job #\n")
                file_out.write("cd .. #\n")
                file_out.write(f'echo "Finished {molecule_name.lower()}..."\n')
                file_out.write("#\n")
        return job_path, tleap_path
