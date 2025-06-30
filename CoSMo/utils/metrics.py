import numpy as np

def get_breaking_points(sequence):
    """
    Identify segment boundaries in a sequence of labels.
    
    Args:
        sequence: Array of class labels
        
    Returns:
        Array where 1 indicates the start of a new segment, 0 elsewhere
    """
    sequence = np.asarray(sequence)
    breaking_points = np.zeros_like(sequence)
    
    breaking_points[0] = 1  

    breaking_points[1:] = (sequence[1:] != sequence[:-1]).astype(int)
    
    return breaking_points


def boundaries_to_segmentation(boundaries):
    """
    Convert boundary indicators to a list of segment sets.
    
    """
    segments = []
    current_segment = set()
    num_items = len(boundaries)

    for i in range(num_items):
        is_start_of_new_segment = (boundaries[i] == 1)

        if is_start_of_new_segment:
            if i > 0 and current_segment:
                segments.append(current_segment)
            current_segment = set()
            
        current_segment.add(i)

    if current_segment:
        segments.append(current_segment)

    return segments


def calculate_mndd(predictions, labels):
    """
    Calculate Minimum Normalized Document Distance.
    
    MnDD = N - sum(max_overlap(gt_segment_i, pred_segment_j) for j) for i
    
    """
    num_items = len(predictions)
    if num_items != len(labels):
        raise ValueError('Predictions and Labels mismatch!')
    
    if num_items == 0:
        return 0
    
    pred_boundaries = get_breaking_points(predictions)
    labels_boundaries = get_breaking_points(labels)
    
    pred_segmentation = boundaries_to_segmentation(pred_boundaries)
    labels_segmentation = boundaries_to_segmentation(labels_boundaries)
    
    sum_max_overlaps = 0

    for g_seg in labels_segmentation:
        max_overlap_for_this_pred_seg = 0
        
        for p_seg in pred_segmentation:
            overlap = len(p_seg.intersection(g_seg))
            if overlap > max_overlap_for_this_pred_seg:
                max_overlap_for_this_pred_seg = overlap

        sum_max_overlaps += max_overlap_for_this_pred_seg

    mndd = num_items - sum_max_overlaps
    return mndd


def calculate_iou(seg1, seg2):
    """
    Calculate Intersection over Union between two segments.
    
    """
    intersection = len(seg1.intersection(seg2))
    union = len(seg1.union(seg2))
    return intersection / union if union > 0 else 0


def calculate_segment_matching(pred_segments, true_segments):
    """
    Calculate maximum bipartite matching between predicted and ground truth segments
    using Intersection over Union (IoU) with threshold 0.5.

    """
    iou_matrix = []
    for p_idx, p_seg in enumerate(pred_segments):
        for t_idx, t_seg in enumerate(true_segments):
            iou = calculate_iou(p_seg, t_seg)
            if iou > 0.5: 
                iou_matrix.append((p_idx, t_idx, iou))

    iou_matrix.sort(key=lambda x: -x[2])
    
    matched_p = set()
    matched_t = set()
    true_positives = []
    
    for p_idx, t_idx, iou in iou_matrix:
        if p_idx not in matched_p and t_idx not in matched_t:
            matched_p.add(p_idx)
            matched_t.add(t_idx)
            true_positives.append((p_idx, t_idx, iou))
    
    return true_positives, matched_p, matched_t


def panoptic_quality_metrics(predictions, labels):
    """
    Calculate document-level precision, recall, F1, and Segmentation Quality
    following the Panoptic Quality approach.
    
    Args:
        predictions: Predicted class labels
        labels: Ground truth class labels
        
    Returns:
        Dictionary containing precision, recall, F1, weighted_f1, and sq metrics
    """
    pred_boundaries = get_breaking_points(predictions)
    true_boundaries = get_breaking_points(labels)
    
    pred_segmentation = boundaries_to_segmentation(pred_boundaries)
    true_segmentation = boundaries_to_segmentation(true_boundaries)
    
    true_positives, matched_p, matched_t = calculate_segment_matching(
        pred_segmentation, true_segmentation
    )
    
    tp = len(true_positives)
    fp = len(pred_segmentation) - len(matched_p)  
    fn = len(true_segmentation) - len(matched_t)  
    
    avg_iou = sum(iou for _, _, iou in true_positives) / tp if tp > 0 else 0
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    pq = f1 * avg_iou
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pq": pq,
        "sq": avg_iou
    }