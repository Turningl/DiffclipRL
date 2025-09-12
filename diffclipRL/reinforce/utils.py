# -*- coding: utf-8 -*-
# @Author : liang
# @File : utils.py


from collections import defaultdict

import numpy as np
import torch
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from typing import Iterable, Protocol, Sequence, Type, TypeVar

from tqdm import tqdm

n_presubsample = 4096
min_2_theta = 0
max_2_theta = 180
wavelength = 'CuKa'
horizontal_noise_range = (1e-2, 1.1e-2)
vertical_noise = 1e-3


def sinc_filt(n_presubsample, min_2_theta, max_2_theta, wavelength, nanomaterial_size):
    min_theta = min_2_theta / 2
    max_theta = max_2_theta / 2
    Q_min = 4 * np.pi * np.sin(np.radians(min_theta)) / wavelength
    Q_max = 4 * np.pi * np.sin(np.radians(max_theta)) / wavelength

    # phase shift for sinc filter = half of the signed Q range
    phase_shift = (Q_max - Q_min) / 2

    # compute Qs
    Qs = np.linspace(Q_min, Q_max, n_presubsample)
    Qs_shifted = Qs - phase_shift

    sinc_filt_ = nanomaterial_size * (np.sinc(nanomaterial_size * Qs_shifted / np.pi) ** 2)

    return sinc_filt_

def structures_to_numpy(structures: Iterable[Structure]):
    """
    Convert a list of Structures to numpy arrays for positions, cell, atomic numbers,
    number of atoms and structure id. Returns a dictionary with the numpy arrays.
    """
    structure_infos: dict[str, list[np.typing.NDArray]] = {
        "pos": [],
        "cell": [],
        "atomic_numbers": [],
        "num_atoms": [],
        "structure_id": [],
        'xrd': [],
    }

    properties = defaultdict(list)
    for structure in tqdm(structures, desc="Converting structures to numpy", miniters=5000):
        # get primitive structure
        # here, structure.properties is not passed to struct if it is not a primitive structure,
        # so we keep the structure object to pass material_id below
        struct = structure.get_primitive_structure()
        # niggli reduction
        struct = struct.get_reduced_structure()

        structure_infos["pos"].append(struct.frac_coords)
        structure_infos["cell"].append(struct.lattice.matrix)
        structure_infos["atomic_numbers"].append(struct.atomic_numbers)
        structure_infos["num_atoms"].append(len(struct))
        structure_infos["structure_id"].append(structure.properties["material_id"])
        structure_infos["xrd"].append(structure.properties["xrd"])

        # for prop, prop_val in structure.properties.items():
        #     if prop in PROPERTY_SOURCE_IDS:
        #         properties[prop].append(prop_val)

    structure_infos["pos"] = np.row_stack(structure_infos["pos"])
    structure_infos["cell"] = np.array(structure_infos["cell"])
    structure_infos["atomic_numbers"] = np.concatenate(structure_infos["atomic_numbers"])
    structure_infos["num_atoms"] = np.array(structure_infos["num_atoms"])
    structure_infos["structure_id"] = np.array(structure_infos["structure_id"])
    structure_infos["xrd"] = np.array(structure_infos["xrd"])

    # for prop in properties:
    #     properties[prop] = np.array(properties[prop])
    #     assert len(properties[prop]) == len(structure_infos["structure_id"])

    return structure_infos, properties


def valid(gen_st_list: list[Structure]):
    valid_gen_st_list = []
    for st in gen_st_list:
        if len(st) == 1:
            valid_gen_st_list.append(st)
            continue
        # check if the lattice length < 60A
        if max(st.lattice.abc) > 60:
            continue
        # check if the lowest distance between atoms > 0.5A
        dist_mat = st.distance_matrix
        lowest_dist = np.min(dist_mat[dist_mat > 0])
        if lowest_dist < 0.5:
            continue
        valid_gen_st_list.append(st)
    return valid_gen_st_list


def unique(st_list: list[Structure]):
    sm = StructureMatcher()
    output_sm = sm.group_structures(st_list)
    return len(output_sm)


def structure_matching(st_list: list[Structure], ref_st: Structure):
    sm = StructureMatcher()
    num_match = 0
    for st in st_list:
        if sm.fit(ref_st, st):
            num_match += 1
    return num_match


def composition_matching(st_list: list[Structure], ref_st: Structure):
    num_match = 0
    for st in st_list:
        if ref_st.composition == st.composition:
            num_match += 1
    return num_match


def crystal_system_matching(
    st_list: list[Structure], ref_st: Structure, symprec=0.1, angle_tolerance=10
):
    num_match = 0
    ref_sga = SpacegroupAnalyzer(
        ref_st, symprec=symprec, angle_tolerance=angle_tolerance
    )
    ref_crystal_system = ref_sga.get_crystal_system()
    for st in st_list:
        try:
            sga = SpacegroupAnalyzer(
                st, symprec=symprec, angle_tolerance=angle_tolerance
            )
            crystal_system = sga.get_crystal_system()
            if crystal_system == ref_crystal_system:
                num_match += 1
        except Exception as e:  # pylint: disable=W0718
            print(e)
    return num_match


def lattice_system_matching(
    st_list: list[Structure], ref_st: Structure, symprec=0.1, angle_tolerance=10
):
    num_match = 0
    ref_sga = SpacegroupAnalyzer(
        ref_st, symprec=symprec, angle_tolerance=angle_tolerance
    )
    ref_lattice_system = ref_sga.get_lattice_type()
    for st in st_list:
        test_st = Structure(
            lattice=st.lattice,
            species=["H"],
            coords=[[0.5, 0.5, 0.5]],
        )
        sga = SpacegroupAnalyzer(
            test_st, symprec=symprec, angle_tolerance=angle_tolerance
        )
        lattice_system = sga.get_lattice_type()
        if lattice_system == ref_lattice_system:
            num_match += 1
    return num_match

def Variable(tensor):
    """Wrapper for torch.autograd.Variable that also accepts
       numpy arrays directly and automatically assigns it to
       the GPU. Be aware in case some operations are better
       left to the CPU."""
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if torch.cuda.is_available():
        return torch.autograd.Variable(tensor).cuda()
    return torch.autograd.Variable(tensor)