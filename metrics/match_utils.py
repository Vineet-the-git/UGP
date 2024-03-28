"""
Functions for metric calculation
"""
import numpy as np
from collections.abc import Iterable
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, adjusted_rand_score
import torch

import numpy as np

def l2_normalize_batch_pytorch(batch):
    norm = np.linalg.norm(batch, axis=1, keepdims=True)
    normalized_batch = batch / norm
    return normalized_batch


def get_matching_acc(matching, labels1, labels2, order=None):
    """
    Compute the cluster level matching accuracy.

    Parameters
    ----------
    matching: a list of length three.
        The matched pairs are (matching[0][i], matching[1][i]),
        and its score (the higher, the better) is matching[2][i].
    labels1: np.array of shape (n_samples1,)
        The first label vector.
    labels2: np.array of shape (n_samples2,)
        The first label vector.
    order: None or (1, 2) or (2, 1), default=None
        If None, then directly use matching without addressing any redundancy.
        If (1, 2), find one-to-one matching from the first dataset to the second dataset;
        if (2, 1), do the other way around.

    Returns
    -------
    Matching accuracy.
    """
    if order is None:
        return np.mean([labels1[i] == labels2[j] for i, j in zip(matching[0], matching[1])])
    matching = address_matching_redundancy(matching=matching, order=order)
    rows, cols, _ = matching
    return np.mean([labels1[i] == labels2[j] for i, j in zip(rows, cols)])


def get_foscttm(dist, true_matching='identity'):
    """
    Compute the fraction of samples closer than true match.

    Parameters
    ----------
    dist: np.ndarray of shape (n1, n2)
        Distance matrix.
    true_matching: 'identity' or Iterable of length n1, default='identity'
        If is a list, then the ground truth matched pairs are (i, true_matching[i])
        If is 'identity', then true_matching = [0, 1..., n1].

    Returns
    -------
    The fraction of samples closer than true match.
    """
    n1, _ = dist.shape
    if true_matching == 'identity':
        true_matching = np.arange(n1)
    elif isinstance(true_matching, Iterable):
        true_matching = [i for i in true_matching]
    else:
        raise NotImplementedError('true_matching must be \'identity\' or Iterable of length dist.shape[0].')
    # mask[i, j] = True iff dist[i, j] < dist[i, true_matching[i]]
    mask = (dist.T < dist[np.arange(n1), true_matching]).T
    return np.mean(np.mean(mask, axis=1))


def get_matching_alignment_score(estimated_matching, n_samples, true_matching='identity'):
    """
    Compute the alignment between the estimated matching and the true_matching
    according to the metric in https://openproblems.bio/neurips_docs/about_tasks/task2_modality_matching/.

    Parameters
    ----------
    estimated_matching: a list of length three.
        The matched pairs are (matching[0][i], matching[1][i]),
        and its score (the higher, the better) is matching[2][i].
    n_samples: int
        The sample size for the first dataset.
    true_matching: 'identity' or Iterable of length n_samples, default='identity'
        If is a list, then the ground truth matched pairs are (i, true_matching[i])
        If is 'identity', then true_matching = [0, 1..., n_samples].

    Returns
    -------
    The alignment score.
    """
    if true_matching == 'identity':
        true_matching = np.arange(n_samples)
    elif isinstance(true_matching, Iterable):
        true_matching = [i for i in true_matching]
    else:
        raise NotImplementedError('true_matching must be \'identity\' or Iterable of length dist.shape[0].')

    idx1_to_indices2_and_scores = dict()
    for i, j, score in zip(estimated_matching[0], estimated_matching[1], estimated_matching[2]):
        if i not in idx1_to_indices2_and_scores:
            idx1_to_indices2_and_scores[i] = [[j], [score]]
        else:
            idx1_to_indices2_and_scores[i][0].append(j)
            idx1_to_indices2_and_scores[i][1].append(score)

    for idx1, indices2_and_scores in idx1_to_indices2_and_scores.items():
        indices2_and_scores[1] = list(np.array(indices2_and_scores[1]) / np.sum(indices2_and_scores[1]))

    res = 0
    for idx1, idx2 in enumerate(true_matching):
        if idx1 in idx1_to_indices2_and_scores:
            for loc in range(len(idx1_to_indices2_and_scores[idx1][0])):
                candidate_idx2 = idx1_to_indices2_and_scores[idx1][0][loc]
                if idx2 == candidate_idx2:
                    res += idx1_to_indices2_and_scores[idx1][1][loc]
    return res / len(idx1_to_indices2_and_scores)


def get_knn_alignment_score(dist, k_max, true_matching='identity'):
    """
    For each 1 <= k <= k_max, obtain knn matching from dist,
    and compute its matching proximity with the true matching.
    The proximity is calculated by:
    for each cell in arr1, claim it is successfully matched when the true match is in the k-nearest-neighborhood;
    then calculate the average success rate.

    Parameters
    ----------
    dist: np.ndarray of shape (n1, n2)
        Distance matrix.
    k_max: int
        Maximum k for knn matching.
    true_matching: 'identity' or Iterable of length n1, default='identity'
        If is a list, then the ground truth matched pairs are (i, true_matching[i])
        If is 'identity', then true_matching = [0, 1..., n1].

    Returns
    -------
    np.ndarray of shape (k_max,) representing the score for each 1<=k<=k_max.
    """
    n1, n2 = dist.shape
    assert k_max <= n2
    knn_indices = np.argsort(dist, axis=1)[:, :k_max]
    # knn_scores = 1 - dist[np.arange(n1)[:, None], knn_indices]

    if true_matching == 'identity':
        true_matching = np.arange(n1)
    elif isinstance(true_matching, Iterable):
        true_matching = [i for i in true_matching]
    else:
        raise NotImplementedError('true_matching must be \'identity\' or Iterable of length dist.shape[0].')

    res = np.zeros(k_max)
    for idx1, idx2 in enumerate(true_matching):
        candidates = knn_indices[idx1, :]
        idx2_location = np.where(candidates == idx2)[0]
        if len(idx2_location) == 0:
            # even k_max-nn matching does not contain the true match
            continue
        # find the first occurrence of idx2
        # every knn matching with k >= idx_location is able to find the true match
        idx2_location = idx2_location[0]
        # curr_scores = knn_scores[idx1, idx2_location] / np.cumsum(knn_scores[idx1, :])
        # res[idx2_location:] = res[idx2_location:] + curr_scores[idx2_location:]
        res[idx2_location:] = res[idx2_location:] + 1
    return res / n1

"""
Utility functions for matching
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from numpy import linalg as LA
import pynndescent
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def address_matching_redundancy(matching, order=(1, 2)):
    """
    Make a potentially multiple-to-multiple matching to an one-to-one matching according to order.

    Parameters
    ----------
    matching: list of length three
        rows, cols, vals = matching: list
        Each matched pair of rows[i], cols[i], their score (the larger, the better) is vals[i]
    order: None or (1, 2) or (2, 1)
        If None, do nothing;
        If (1, 2), then the redundancy is addressed by making matching
        an injective map from the first dataset to the second;
        if (2, 1), do the other way around.

    Returns
    -------
    rows, cols, vals: list
        Each matched pair of rows[i], cols[i], their score is vals[i].
    """
    if order is None:
        return matching
    res = [[], [], []]
    if order == (1, 2):
        idx1_to_idx2 = dict()
        idx1_to_score = dict()
        for i, j, score in zip(matching[0], matching[1], matching[2]):
            if i not in idx1_to_idx2:
                idx1_to_idx2[i] = j
                idx1_to_score[i] = score
            elif score > idx1_to_score[i]:
                idx1_to_idx2[i] = j
                idx1_to_score[i] = score
        for idx1, idx2 in idx1_to_idx2.items():
            res[0].append(idx1)
            res[1].append(idx2)
            res[2].append(idx1_to_score[idx1])
    elif order == (2, 1):
        idx2_to_idx1 = dict()
        idx2_to_score = dict()
        for i, j, score in zip(matching[0], matching[1], matching[2]):
            if j not in idx2_to_idx1:
                idx2_to_idx1[j] = i
                idx2_to_score[j] = score
            elif score > idx2_to_score[j]:
                idx2_to_idx1[j] = i
                idx2_to_score[j] = score
        for idx2, idx1 in idx2_to_idx1.items():
            res[0].append(idx1)
            res[1].append(idx2)
            res[2].append(idx2_to_score[idx2])
    else:
        raise NotImplementedError('order must be in {None, (1, 2), (2, 1)}.')

    return res

def distance_pred(latent1 , latent2, metric):
    distances = np.zeros((latent1.shape[0], latent2.shape[0]))
    if metric=='euclidean':
        for i in range(latent1.shape[0]):
            distances[i,:] = LA.norm(latent2 - latent1[i], axis=1)  # Calculate Euclidean distance
    elif metric=='cosine':
        latent1 = l2_normalize_batch_pytorch(latent1)
        latent2 = l2_normalize_batch_pytorch(latent2)
        distances = 1 - np.dot(latent1,latent2.T)
    return distances
    
def match_cells(arr1, arr2, metric, assignment_type='linear', verbose=True):
    """
    Get matching between arr1 and arr2 using linear assignment, the distance is 1 - Pearson correlation.

    Parameters
    ----------
    arr1: np.array of shape (n_samples1, n_features)
        The first data matrix
    arr2: np.array of shape (n_samples2, n_features)
        The second data matrix
    base_dist: None or np.ndarray of shape (n_samples1, n_samples2)
        Baseline distance matrix
    wt_on_base_dist: float between 0 and 1
        The final distance matrix to use is (1-wt_on_base_dist) * dist[arr1, arr2] + wt_on_base_dist * base_dist
    verbose: bool, default=True
        Whether to print the progress

    Returns
    -------
    rows, cols, vals: list
        Each matched pair of rows[i], cols[i], their distance is vals[i]
    """
    if verbose:
        print('Start the matching process...', flush=True)
        print('Computing the distance matrix...', flush=True)
    dist = distance_pred(arr1, arr2, metric)
    if verbose and assignment_type=='linear':
        print('Getting matchings by Linear Assignment.....', flush=True)
        rows,cols = linear_sum_assignment(dist)
    if verbose and assignment_type=='mindist':
        print('Getting matchings by Minimum Distance...', flush=True)
        rows = np.arange(arr1.shape[0])
        cols = np.argmin(dist, axis=1)
    if verbose:
        print('Linear assignment completed!', flush=True)
    return rows, cols, np.array([dist[i, j] for i, j in zip(rows, cols)])


import numpy as np

def filter_matchings(matchings, filter_prop=0.3, verbose=True):
    """
    Filter matchings based on the score percentile.

    Parameters:
    - matchings: a list of length three.
      The matched pairs are (matching[0][i], matching[1][i]),
      and its score (the higher, the better) is matching[2][i].
    - filter_prop: The percentile threshold for filtering matchings.

    Returns:
    A filtered list of matchings.
    """
    if len(matchings) != 3:
        raise ValueError("Input must be a list of length three.")

    scores = np.array(matchings[2])
    threshold = np.percentile(scores, (1.0-filter_prop) * 100)
    if verbose:
        print("Matchings before filtering: ", len(matchings[0]))
        print("Threshold distance for filtering the matchings: ", threshold)
    rows = []
    cols = []
    matching_scores = []
    for i in range(len(scores)):
        if matchings[2][i] <= threshold:
            rows.append(matchings[0][i])
            cols.append(matchings[1][i])
            matching_scores.append(matchings[2][i])
    if verbose:
        print("Matchings are reduced to {} after filtering...".format(len(rows)))
    return [rows, cols, matching_scores]


def propagate(curr_arr1, curr_arr2, matchings, metric="euclidean", verbose=True):
    # curr_arr1 are rna latents because we are using rna as modality 1
    # curr_arr2 are protein latents because we are using protein as modality 2
    curr_propagated_matching = [[], [], []]
    good_indices1 = np.array(matchings[0])
    good_indices2 = np.array(matchings[1])
    # get remaining indices
    # propagation will only be done for those indices
    good_indices1_set = set(good_indices1)
    remaining_indices1 = [i for i in range(curr_arr1.shape[0]) if i not in good_indices1_set]
    good_indices2_set = set(good_indices2)
    remaining_indices2 = [i for i in range(curr_arr2.shape[0]) if i not in good_indices2_set]

    # propagate for remaining indices in arr1
    if len(remaining_indices1) > 0:
        # get 1-nearest-neighbors and the corresponding distances
        remaining_indices1_nns, remaining_indices1_nn_dists = get_nearest_neighbors(
            query_arr=curr_arr1[remaining_indices1, :],
            target_arr=curr_arr1[good_indices1, :],
            metric=metric
        )
        matched_indices2 = good_indices2[list(remaining_indices1_nns.astype(int))]
        curr_propagated_matching[0].extend(remaining_indices1)
        curr_propagated_matching[1].extend(matched_indices2)
        curr_propagated_matching[2].extend(remaining_indices1_nn_dists)

    # propagate for remaining indices in arr2
    if len(remaining_indices2) > 0:
        # get 1-nearest-neighbors and the corresponding distances
        remaining_indices2_nns, remaining_indices2_nn_dists = get_nearest_neighbors(
            query_arr=curr_arr2[remaining_indices2, :],
            target_arr=curr_arr2[good_indices2, :],
            metric=metric
        )
        matched_indices1 = good_indices1[remaining_indices2_nns.astype(int)]
        curr_propagated_matching[0].extend(matched_indices1)
        curr_propagated_matching[1].extend(remaining_indices2)
        curr_propagated_matching[2].extend(remaining_indices2_nn_dists)

    if verbose:
        print('Done!', flush=True)
    return curr_propagated_matching


def get_nearest_neighbors(query_arr, target_arr,
                          metric='euclidean'):
    """
    For each row in query_arr, compute its nearest neighbor in target_arr.

    Parameters
    ----------
    query_arr: np.array of shape (n_samples1, n_features)
        The query data matrix.
    target_arr: np.array of shape (n_samples2, n_features)
        The target data matrix.
    metric: string, default='correlation'
        The metric to use in nearest neighbor search.

    Returns
    -------
    neighbors: np.array of shape (n_samples1)
        The i-th element is the index in target_arr to whom the i-th row of query_arr is closest to.
    dists: np.array of shape (n_samples1)
        The i-th element is the distance corresponding to neighbors[i].
    """
    arr = np.vstack([query_arr, target_arr])
    query_arr = arr[:query_arr.shape[0], :]
    pivot_arr = arr[query_arr.shape[0]:, :]
    # approximate nearest neighbor search
    index = pynndescent.NNDescent(pivot_arr, n_neighbors=100, metric=metric)
    neighbors, dists = index.query(query_arr, k=50)
    neighbors, dists = neighbors[:, 0], dists[:, 0]
    return neighbors, dists

def get_all_metrics(rna_latents, protein_latents,labels_l1, labels_l2, dim_use = 20, k_max=25, metric='euclidean', assignment_type='linear', filter_prop=0.3, verbose=True):
    print("{} metric will be used for calculating the distance!".format(metric))
    metrices_dic = {}
    matchings = match_cells(rna_latents, protein_latents, metric, assignment_type,verbose)
    distance_matrix_for_all_cells = distance_pred(np.concatenate([rna_latents, protein_latents], axis=0), np.concatenate([rna_latents, protein_latents], axis=0), metric)
    dist_matrix = distance_pred(rna_latents, protein_latents, metric)

    lv1_acc = get_matching_acc(matching=matchings, 
    labels1=labels_l1, 
    labels2=labels_l1 
    )
    lv2_acc = get_matching_acc(matching=matchings, 
        labels1=labels_l2, 
        labels2=labels_l2 
    )
    if verbose:
        print(f'For All matchings: lv1 matching acc: {lv1_acc:.3f},\nlv2 matching acc: {lv2_acc:.3f}.')
    foscttm_score = get_foscttm(
        dist=dist_matrix,
        true_matching='identity'
    )
    knn_score = get_knn_alignment_score(dist=dist_matrix, k_max=k_max, true_matching='identity')
    silhouette_f1_score_lv1, silhouette_f1_score_lv2 = get_sillohette_f1_score(rna_latents, protein_latents,labels_l1, labels_l2, distance_matrix_for_all_cells,verbose=True)
    if verbose:
        print(f'For All matchings: FOSCTTM score: {foscttm_score}, \nKnn Alignment score: {knn_score}')
        print(f'For All matchings: Silhouette F1 score for lv1: {silhouette_f1_score_lv1}, \nSilhouette F1 score for lv2: {silhouette_f1_score_lv2}')

    cm = confusion_matrix(labels_l1[matchings[0]], labels_l1[matchings[1]])
    ConfusionMatrixDisplay(
        confusion_matrix=np.round((cm.T/np.sum(cm, axis=1)).T*100), 
        display_labels=np.unique(labels_l1)
    ).plot()
    metrices_dic['original'] = {
        'lv1_acc': lv1_acc,
        'lv2_acc': lv2_acc,
        'foscttm_score': foscttm_score,
        'silhouette_f1_score_lv1': silhouette_f1_score_lv1,
        'silhouette_f1_score_lv2': silhouette_f1_score_lv2
    }


    if verbose:
        print("\n\n")
        print("Matches are being filtered...")
    filtered_matchings = filter_matchings(matchings, filter_prop, verbose)
    lv1_acc = get_matching_acc(matching=filtered_matchings, 
    labels1=labels_l1, 
    labels2=labels_l1 
    )
    lv2_acc = get_matching_acc(matching=filtered_matchings, 
        labels1=labels_l2, 
        labels2=labels_l2 
    )
    if verbose:
        print(f'For only filtered matchings: lv1 matching acc: {lv1_acc:.3f},\nlv2 matching acc: {lv2_acc:.3f}.')
    # foscttm_score = get_foscttm(
    #     dist=dist_matrix,
    #     true_matching='identity'
    # )
    # knn_score = get_knn_alignment_score(dist=dist_matrix, k_max=k_max, true_matching='identity')
    # silhouette_f1_score_lv1, silhouette_f1_score_lv2 = get_sillohette_f1_score(rna_latents, protein_latents,labels_l1, labels_l2, distance_matrix_for_all_cells,verbose=True)
    # if verbose:
        # print(f'For only filtered matchings: FOSCTTM score: {foscttm_score}, \nKnn Alignment score: {knn_score}')
        # print(f'For only filtered matchings: Silhouette F1 score for lv1: {silhouette_f1_score_lv1}, \nSilhouette F1 score for lv2: {silhouette_f1_score_lv2}')
    cm = confusion_matrix(labels_l1[filtered_matchings[0]], labels_l1[filtered_matchings[1]])
    ConfusionMatrixDisplay(
        confusion_matrix=np.round((cm.T/np.sum(cm, axis=1)).T*100), 
        display_labels=np.unique(labels_l1)
    ).plot()
    metrices_dic['filtered'] = {
        'lv1_acc': lv1_acc,
        'lv2_acc': lv2_acc,
        'foscttm_score': foscttm_score,
        'silhouette_f1_score_lv1': silhouette_f1_score_lv1,
        'silhouette_f1_score_lv2': silhouette_f1_score_lv2
    }
    if verbose:
        print("\n\n")
        print("Matches are being propagated....")
    propagated_matchings = propagate(rna_latents, protein_latents, filtered_matchings, metric, verbose)
    lv1_acc = get_matching_acc(matching=propagated_matchings, 
    labels1=labels_l1, 
    labels2=labels_l1 
    )
    lv2_acc = get_matching_acc(matching=propagated_matchings, 
        labels1=labels_l2, 
        labels2=labels_l2 
    )
    if verbose:
        print(f'For all matchings after propagation: lv1 matching acc: {lv1_acc:.3f},\nlv2 matching acc: {lv2_acc:.3f}.')
    # foscttm_score = get_foscttm(
    #     dist=dist_matrix,
    #     true_matching='identity'
    # )
    # knn_score = get_knn_alignment_score(dist=dist_matrix, k_max=k_max, true_matching='identity')
    # silhouette_f1_score_lv1, silhouette_f1_score_lv2 = get_sillohette_f1_score(rna_latents, protein_latents,labels_l1, labels_l2, distance_matrix_for_all_cells,verbose=True)
    # if verbose:
    #     print(f'For All matchings after propagation: FOSCTTM score: {foscttm_score}, \nKnn Alignment score: {knn_score}')
    #     print(f'For All matchings after propagation: Silhouette F1 score for lv1: {silhouette_f1_score_lv1}, \nSilhouette F1 score for lv2: {silhouette_f1_score_lv2}')
    cm = confusion_matrix(labels_l1[propagated_matchings[0]], labels_l1[propagated_matchings[1]])
    ConfusionMatrixDisplay(
        confusion_matrix=np.round((cm.T/np.sum(cm, axis=1)).T*100), 
        display_labels=np.unique(labels_l1)
    ).plot()
    plt.show()
    metrices_dic['propagated'] = {
        'lv1_acc': lv1_acc,
        'lv2_acc': lv2_acc,
        'foscttm_score': foscttm_score,
        'silhouette_f1_score_lv1': silhouette_f1_score_lv1,
        'silhouette_f1_score_lv2': silhouette_f1_score_lv2
    }
    return metrices_dic

def get_sillohette_f1_score(rna_latents, protein_latents,labels_l1, labels_l2, distance_matrix_for_all_cells, metric='precomputed', verbose=True):
    batch_labels = np.concatenate([np.zeros(rna_latents.shape[0]), np.ones(protein_latents.shape[0])])
    # Calculate Silhouette widths for modality mixing
    normalized_modality_silhouette_widths = (1+silhouette_samples(distance_matrix_for_all_cells, batch_labels, metric=metric))/2
    slt_mix = 1 - normalized_modality_silhouette_widths

    # Calculate Silhouette widths for cell type clustering
    slt_clust_lv1 = (1 + silhouette_samples(distance_matrix_for_all_cells, np.concatenate([labels_l1,labels_l1], axis=0), metric=metric))/ 2
    slt_clust_lv2 = (1 + silhouette_samples(distance_matrix_for_all_cells, np.concatenate([labels_l2,labels_l2], axis=0), metric=metric))/ 2

    # Calculate F1 score
    f1_score_lv1 = (2 * slt_clust_lv1.mean() * slt_mix.mean()) / (slt_clust_lv1.mean() + slt_mix.mean())
    f1_score_lv2 = (2 * slt_clust_lv2.mean() * slt_mix.mean()) / (slt_clust_lv2.mean() + slt_mix.mean())
    return f1_score_lv1, f1_score_lv2

