import numpy as np
from modules import feature_extraction as fe, postprocessing as pp
import mdtraj as md
'''
rest = [0, 550, 0, 350, 0, 450, 0, 1850, 0, 1600, 0, 350]
acti = [2100, 3260, 2125, 3260, 2150, 3920, 3300, 3920, 3300, 3920, 2350, 3920]

rest = np.asarray(rest)
acti = np.asarray(acti)

rest = rest.reshape((len(rest)/2,2))
acti = acti.reshape((len(acti)/2,2))

residues_pairs = []
for i in np.arange(0,61,1):
    for j in np.arange(i+1,61,1):
        residues_pairs.append([i,j])

traj = md.load('/media/mkasimova/Data4/Anton.PRODUCTION_RUN/ready.scripts/Anton.first.run/all.wrapped.CORRECT.xtc', top='/media/mkasimova/Data4/Anton.PRODUCTION_RUN/ready.scripts/Anton.first.run/system.psf')

i_count = 0
for i in [1,2]:

    S5S6_selection = 'chainid '+str(i)+' and (residue 201 to 230 or residue 276 to 306)'
    S5S6_indices = traj.topology.select(S5S6_selection)
    S5S6 = traj.atom_slice(S5S6_indices)

    distances, res_ind = md.compute_contacts(S5S6, contacts=residues_pairs, scheme='closest-heavy')

    if i_count==0:
        data_rest = distances[rest[i_count,0]:rest[i_count,1]]
        data_acti = distances[acti[i_count,0]:acti[i_count,1]]
    else:
        data_rest = np.concatenate((data_rest,distances[rest[i_count,0]:rest[i_count,1]]),axis=0)
        data_acti = np.concatenate((data_acti,distances[acti[i_count,0]:acti[i_count,1]]),axis=0)

    i_count += 1

traj = md.load('/media/mkasimova/Data4/Anton.PRODUCTION_RUN/ready.scripts/Anton.second.run/all.wrapped.xtc', top='/media/mkasimova/Data4/Anton.PRODUCTION_RUN/ready.scripts/Anton.second.run/system.psf')

for i in [0,1,2,3]:

    S5S6_selection = 'chainid '+str(i)+' and (residue 201 to 230 or residue 276 to 306)'
    S5S6_indices = traj.topology.select(S5S6_selection)
    S5S6 = traj.atom_slice(S5S6_indices)

    distances, res_ind = md.compute_contacts(S5S6, contacts=residues_pairs, scheme='closest-heavy')

    data_rest = np.concatenate((data_rest,distances[rest[i_count,0]:rest[i_count,1]]),axis=0)
    data_acti = np.concatenate((data_acti,distances[acti[i_count,0]:acti[i_count,1]]),axis=0)

    i_count += 1

print(data_rest.shape)
print(data_acti.shape)

data = np.concatenate((data_rest,data_acti),axis=0)

np.save('/media/mkasimova/Data4/Anton.PRODUCTION_RUN/ready.scripts/COMMON/S5-S6.contacts/mdtraj/data.npy', data)
'''
data = np.load('/media/mkasimova/Data4/Anton.PRODUCTION_RUN/ready.scripts/COMMON/S5-S6.contacts/mdtraj/data.npy')
data = 1/(data)
cluster_indices = np.loadtxt('/media/mkasimova/Data4/Anton.PRODUCTION_RUN/ready.scripts/COMMON/S5-S6.contacts/mdtraj/cluster_indices.dat')

n_splits = 1
n_iterations = 10

feature_extractors = [
fe.RandomForestFeatureExtractor(data, cluster_indices, n_splits=n_splits, n_iterations=n_iterations, scaling=True, filter_by_distance_cutoff=True),
fe.KLFeatureExtractor(data, cluster_indices, n_splits=n_splits, scaling=True, filter_by_distance_cutoff=True, bin_width=0.01),
fe.MlpFeatureExtractor(data, cluster_indices, n_splits=n_splits, n_iterations=n_iterations, hidden_layer_sizes=(100,), scaling=True, filter_by_distance_cutoff=True)
]

for extractor in feature_extractors:

    extractor.extract_features()
    pp = extractor.postprocessing(working_dir='/media/mkasimova/Data4/Anton.PRODUCTION_RUN/ready.scripts/COMMON/S5-S6.contacts/mdtraj/', rescale_results=True, filter_results=False, pdb_file='/media/mkasimova/Data4/Anton.PRODUCTION_RUN/ready.scripts/COMMON/S5-S6.contacts/mdtraj/S5S6.pdb')
    pp.average().persist()

