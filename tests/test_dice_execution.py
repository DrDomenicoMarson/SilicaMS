import json
import os
import subprocess
import sys
import numpy as np
import pytest


@pytest.fixture(scope="class", autouse=True)
def _dice_case_context(request, dice_execution_context):
    """Expose the shared dice context as class attributes."""

    if request.cls is None:
        return

    request.cls.dice = dice_execution_context.dice
    request.cls.search_args = dice_execution_context.search_args
    request.cls.expected = dice_execution_context.expected
    request.cls.repo_root = dice_execution_context.repo_root


class TestDiceExecution:
    """Exercise the serial dice search API."""

    @staticmethod
    def _naive_find(dice, cube_list, atom_type, distance):
        """Reference implementation for validating ``Dice.find``.

        Parameters
        ----------
        dice : Dice
            Dice instance under test.
        cube_list : list or None
            Optional subset of cube ids to search.
        atom_type : list[str]
            Requested atom-type pair.
        distance : list[float]
            Inclusive distance bounds ``[lower, upper]``.

        Returns
        -------
        bond_list : list[list[object]]
            Reference bond list with the same shape as :meth:`Dice.find`.
        """
        pointer = dice.get_pointer()
        cube_list = cube_list if cube_list else list(pointer.keys())
        positions = dice.get_mol().positions_view()
        atom_types = dice.get_mol().atom_types_view()
        box = np.asarray(dice.get_mol().get_box(), dtype=float)
        size = dice.get_size()

        bond_list = []
        for cube_id in cube_list:
            atoms = []
            for neighbor_id in dice.neighbor(cube_id):
                if None not in neighbor_id:
                    atoms.extend(pointer[neighbor_id])

            for atom_id_a in pointer[cube_id]:
                if atom_types[atom_id_a] != atom_type[0]:
                    continue

                partners = []
                for atom_id_b in atoms:
                    if atom_types[atom_id_b] != atom_type[1] or atom_id_a == atom_id_b:
                        continue

                    delta = positions[atom_id_a] - positions[atom_id_b]
                    for dim in range(3):
                        if abs(delta[dim]) > 3 * size:
                            delta[dim] -= box[dim] * round(delta[dim] / box[dim])

                    length = float(np.sqrt(np.dot(delta, delta)))
                    if distance[0] <= length <= distance[1]:
                        partners.append(atom_id_b)

                bond_list.append([atom_id_a, partners])

        return bond_list

    @staticmethod
    def _normalize(bond_list):
        """Normalize bond lists for order-independent comparisons.

        Parameters
        ----------
        bond_list : list
            Bond list returned by :meth:`silicams.dice.Dice.find`.

        Returns
        -------
        normalized : list
            Bond list with sorted entries and partner indices.
        """
        return sorted(
            [[entry[0], sorted(entry[1])] for entry in bond_list],
            key=lambda entry: entry[0],
        )

    def _run_subprocess(self, code):
        """Run a Python subprocess from a ``python -c`` entrypoint.

        Parameters
        ----------
        code : str
            Python source code passed to ``python -c``.

        Returns
        -------
        result : CompletedProcess
            Completed subprocess invocation.
        """
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            self.repo_root
            if not existing_pythonpath
            else self.repo_root + os.pathsep + existing_pythonpath
        )
        return subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            check=False,
            cwd=self.repo_root,
            env=env,
            text=True,
        )

    def test_find_matches_expected(self):
        result = self.dice.find(*self.search_args)
        assert self._normalize(result) == self.expected

    def test_find_matches_naive_reference(self):
        result = self.dice.find(*self.search_args)
        reference = self._naive_find(self.dice, *self.search_args)
        assert self._normalize(result) == self._normalize(reference)

    def test_find_requires_atom_type_and_distance(self):
        with pytest.raises(TypeError):
            self.dice.find()

    def test_find_has_no_entrypoint_warning_for_python_dash_c(self):
        code = """
import json
import warnings
from silicams.dice import Dice
from silicams.pattern import BetaCristobalit

block = BetaCristobalit().generate([2, 2, 2], "z")
dice = Dice(block, 0.2, True)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    result = dice.find(
        None,
        ["Si", "O"],
        [0.155 - 1e-2, 0.155 + 1e-2],
    )
print(json.dumps({
    "length": len(result),
    "warnings": [str(item.message) for item in caught],
}))
"""
        result = self._run_subprocess(code)
        assert result.returncode == 0, result.stderr

        payload = json.loads(result.stdout.strip())
        assert payload["length"] == len(self.expected)
        assert payload["warnings"] == []

    def test_matrix_build_has_no_entrypoint_warning_for_python_dash_c(self):
        code = """
import json
import warnings
from silicams.dice import Dice
from silicams.matrix import Matrix
from silicams.pattern import BetaCristobalit

block = BetaCristobalit().generate([2, 2, 2], "z")
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    matrix = Matrix(
        Dice(block, 0.4, True).find(
            None,
            ["Si", "O"],
            [0.155 - 1e-2, 0.155 + 1e-2],
        )
    )
print(json.dumps({
    "matrix_size": len(matrix.get_matrix()),
    "warnings": [str(item.message) for item in caught],
}))
"""
        result = self._run_subprocess(code)
        assert result.returncode == 0, result.stderr

        payload = json.loads(result.stdout.strip())
        assert payload["matrix_size"] > 0
        assert payload["warnings"] == []
