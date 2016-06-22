import numpy as np

from copy import deepcopy 

def remove_labels(n_instances, n_to_remove):
    if (n_to_remove > sum(n_instances)):
        raise ValueError('Number of labels to remove greater than number of labeled instances')

    to_remove = np.zeros([len(n_instances)])

    if (n_to_remove == 0):
        return to_remove

    max = np.max(n_instances)

    while (n_to_remove > 0):
        max_idx = np.where(n_instances == max)[0]
        mask = np.array(len(n_instances)*[True])
        mask[max_idx] = False

        if (n_to_remove < len(max_idx)):
            max_idx = max_idx[:n_to_remove]

        # Number of items that would be removed to set all max items equal to second largest smallest number
        mmax = np.max(n_instances[mask]) if (len(n_instances[mask]) > 0) else max-(n_to_remove / len(max_idx))
        remove_it = np.min([(n_to_remove / len(max_idx)), (max - mmax)])

        to_remove[max_idx] += remove_it
        n_instances[max_idx] -= remove_it

        n_to_remove -= len(max_idx) * remove_it
        max -= remove_it

    return to_remove

# Calculate number of labels to remove from each class
#n_instances = np.array([len(b[0]) for b in buckets])
n_instances = np.array([4932, 5678, 4968, 5101, 4859, 4506, 4951, 5175, 4842, 4988])
n_to_remove = 49900

to_remove = remove_labels(deepcopy(n_instances), n_to_remove)

assert (int(np.sum(n_instances - to_remove)) == np.sum(n_instances) - n_to_remove)
