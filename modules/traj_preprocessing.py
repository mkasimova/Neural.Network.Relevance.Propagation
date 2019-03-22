import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

import mdtraj as md
import os
import numpy as np
import argparse
from . import filtering

logger = logging.getLogger("trajPreprocessing")


def to_distances(traj,
                 scheme="ca",
                 pairs="all",
                 filter_by_distance_cutoff=False,
                 lower_bound_distance_cutoff=filtering.lower_bound_distance_cutoff_default,
                 upper_bound_distance_cutoff=filtering.upper_bound_distance_cutoff_default,
                 use_inverse_distances=True,
                 ignore_nonprotein=True,
                 periodic=True,
                 ):
    if pairs is None:
        pairs = "all"
    top = traj.topology
    if scheme == 'all-heavy':
        atoms = traj.top.select("{} and element != 'H'".format("protein" if ignore_nonprotein else "all"))
        if pairs is None or pairs == 'all':
            pairs = []
            for idx1, a1 in enumerate(atoms):
                for idx2 in range(idx1 + 1, len(atoms)):
                    a2 = atoms[idx2]
                    pairs.append([a1, a2])
        samples = md.compute_distances(traj, pairs, periodic=periodic, opt=True)
        pairs = np.array([
            [top.atom(a1), top.atom(a2)] for a1, a2 in pairs
        ])
        feature_to_resids = np.array([
            [a1.residue.resSeq, a2.residue.resSeq] for a1, a2 in pairs
        ], dtype=int)
    else:
        chunk_size = 1000  # To use less memory, don't process entire traj at once
        start = 0
        samples = None
        while start < len(traj):
            end = start + chunk_size
            s, pairs = md.compute_contacts(traj[start:end], contacts=pairs, scheme=scheme,
                                           ignore_nonprotein=ignore_nonprotein,
                                           periodic=periodic)
            if samples is None:
                samples = s
            else:
                samples = np.append(samples, s, axis=0)
            start = end
        pairs = np.array([[top.residue(r1), top.residue(r2)] for [r1, r2] in pairs])
        feature_to_resids = np.array([
            [r1.resSeq, r2.resSeq] for r1, r2 in pairs
        ], dtype=int)

    if filter_by_distance_cutoff:
        samples, indices_for_filtering = filtering.filter_by_distance_cutoff(samples,
                                                                             lower_bound_distance_cutoff,
                                                                             upper_bound_distance_cutoff,
                                                                             inverse_distances=False)
    if use_inverse_distances:
        samples = 1 / samples
    return samples, feature_to_resids, pairs


def to_compact_distances(traj, **kwargs):
    """
    compact-dist uses as few distances as possible

    :param traj:
    :param kwargs: see to_distances
    :return:
    """
    samples, feature_to_resids, pairs = to_distances(traj, **kwargs)
    if samples.shape[1] <= 6:
        return samples, feature_to_resids, pairs

    indices_to_include = []
    last_r1 = None
    count_for_current_residue = None
    for idx, (r1, r2) in enumerate(pairs):
        # TODO we actually include a few distances too many here I think, but the order of magnitude is optimal
        if last_r1 != r1:
            count_for_current_residue = 0
        if last_r1 is None or count_for_current_residue < 4:
            indices_to_include.append(idx)
        count_for_current_residue += 1
        last_r1 = r1

    feature_to_resids = feature_to_resids[indices_to_include]
    samples = samples[:, indices_to_include]
    pairs = pairs[indices_to_include]
    return samples, feature_to_resids, pairs


def to_cartesian(traj, query="protein and name 'CA'"):
    """
    :param traj:
    :param query:
    :return: array with every frame along first axis, and xyz coordinates in sequential order for
    every atom returned by the query
    """
    atom_indices = traj.top.select(query)
    atoms = [traj.top.atom(a) for a in atom_indices]
    xyz = traj.atom_slice(atom_indices=atom_indices).xyz
    natoms = xyz.shape[1]
    samples = np.empty((xyz.shape[0], 3 * natoms))
    feature_to_resids = np.empty((3 * natoms, 2), dtype=int)
    for idx in range(natoms):
        start, end = 3 * idx, 3 * idx + 3
        samples[:, start:end] = xyz[:, idx, :]
        feature_to_resids[start:end] = atoms[idx].residue.resSeq
    return samples, feature_to_resids, None


def create_argparser():
    parser = argparse.ArgumentParser(
        epilog='Demystifying stuff since 2018. By delemottelab')
    parser.add_argument('--working_dir', type=str, help='working directory', required=True)
    parser.add_argument('--output_dir', type=str, help='Relative path to output directory from the working dir',
                        required=False, default=None)
    parser.add_argument('--traj', type=str, help='Relative path to trajectory file from the working dir', required=True)
    parser.add_argument('--topology', type=str, help='Relative path to topology file from the working dir',
                        required=False,
                        default=None)
    parser.add_argument('--feature_type', type=str, help='Choice of feature type', required=True)
    parser.add_argument('--dt', type=int, help='Timestep between frames', default=1)
    return parser


if __name__ == "__main__":
    logger.info("----------------Starting trajectory preprocessing------------")
    parser = create_argparser()
    args = parser.parse_args()
    logger.info("Starting with arguments: %s", args)
    dt = args.dt
    traj = md.load(args.working_dir + "/" + args.traj,
                   top=None if args.topology is None else args.working_dir + args.topology,
                   stride=dt)
    logger.info("Loaded trajectory %s", traj)
    if args.feature_type == 'ca_inv':
        samples, feature_to_resids, pairs = to_distances(traj, scheme='ca', use_inverse_distances=True)
    elif args.feature_type == 'closest-heavy_inv':
        samples, feature_to_resids, pairs = to_distances(traj, scheme='closest-heavy', use_inverse_distances=True)
    elif args.feature_type == 'compact_ca_inv':
        samples, feature_to_resids, pairs = to_compact_distances(traj, scheme='ca', use_inverse_distances=True)
    elif args.feature_type == 'cartesian_ca':
        samples, feature_to_resids, pairs = to_cartesian(traj)
    elif args.feature_type == 'cartesian_noh':
        samples, feature_to_resids, pairs = to_cartesian(traj, query="protein and element != 'H'")
    else:
        logger.error("feature_type %s not supported", args.feature_type)
    out_dir = args.working_dir + "/";
    out_dir += args.feature_type if args.output_dir is None else args.working_dir
    out_dir += "/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    np.savez_compressed(out_dir + "samples_dt%s" % dt, array=samples)
    np.save(out_dir + "feature_to_resids", feature_to_resids)
    logger.info("Finished. Saved results in %s", out_dir)
