import numpy as np
import matplotlib.pyplot as plt
from tools import read_predicted_boxes, read_ground_truth_boxes


def calculate_iou(prediction_box, gt_box):
    """Calculate intersection over union of single predicted and ground truth box.

    Args:
        prediction_box (np.array of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (np.array of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]

        returns:
            float: value of the intersection of union for the two boxes.
    """

    # Calculate intersection (w * h of the inner square)
    xmin_max = max(prediction_box[0], gt_box[0])
    ymin_max = max(prediction_box[1], gt_box[1])
    xmax_min = min(prediction_box[2], gt_box[2])
    ymax_min = min(prediction_box[3], gt_box[3])

    intersection = max(xmax_min - xmin_max, 0.) * max(ymax_min - ymin_max, 0.)

    # Calculate union (area_box1 + area_box1 - intersection)
    prediction_box_area = (prediction_box[2] - prediction_box[0]) * (prediction_box[3] - prediction_box[1])
    gt_box_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])

    union = prediction_box_area + gt_box_area - intersection

    # Compute union
    iou = intersection / union

    assert iou >= 0 and iou <= 1
    return iou


def calculate_precision(num_tp, num_fp, num_fn):
    """ Calculates the precision for the given parameters.
        Returns 1 if num_tp + num_fp = 0

    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of precision
    """

    if num_tp == 0:
        return 1
    else:
        precision = num_tp / (num_tp + num_fp)
        return precision


def calculate_recall(num_tp, num_fp, num_fn):
    """ Calculates the recall for the given parameters.
        Returns 0 if num_tp + num_fn = 0
    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of recall
    """
    if num_tp == 0:
        return 0
    else:
        recall = num_tp / (num_tp + num_fn)
        return recall


def get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold):
    """Finds all possible matches for the predicted boxes to the ground truth boxes.
        No bounding box can have more than one match.

        Remember: Matching of bounding boxes should be done with decreasing IoU order!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns the matched boxes (in corresponding order):
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of box matches, 4].
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of box matches, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    """

    matches = list()
    # Iterate over all prediction_boxes
    for n, pred_box in enumerate(prediction_boxes):
        # Find all possible matches with a IoU
        ious = np.array([calculate_iou(pred_box, gt_box) for gt_box in gt_boxes])
        # Find match with the highest IoU over threshold and add match to list
        if ious.max() >= iou_threshold:
            matches.append((n, ious.argmax(), ious.max()))

    # Sort all matches on IoU in descending order
    matches = np.array(matches, dtype=[('pred', int), ('gt', int), ('iou', float)])
    matches = np.sort(matches, order=['iou', 'pred'])[::-1]

    return prediction_boxes[matches['pred']], gt_boxes[matches['gt']]


def calculate_individual_image_result(prediction_boxes, gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes,
       calculates true positives, false positives and false negatives
       for a single image.
       NB: prediction_boxes and gt_boxes are not matched!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        dict: containing true positives, false positives, true negatives, false negatives
            {"true_pos": int, "false_pos": int, false_neg": int}
    """
    # Get matched boxes
    matched_prediction_boxes, matched_gt_boxes = get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold)

    # Count number of results
    results = dict(
        true_pos=len(matched_prediction_boxes),
        false_pos=len(prediction_boxes) - len(matched_prediction_boxes),
        false_neg=len(gt_boxes) - len(np.unique(matched_gt_boxes, axis=0)),
    )

    return results


def calculate_precision_recall_all_images(all_prediction_boxes, all_gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates recall and precision over all images
       for a single image.
       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        tuple: (precision, recall). Both float.
    """
    # Get number of true positives, false positives, false negatives over all images
    num_tp, num_fp, num_fn = 0, 0, 0
    for prediction_boxes, gt_boxes in zip(all_prediction_boxes, all_gt_boxes):
        tp, fp, fn = calculate_individual_image_result(prediction_boxes, gt_boxes, iou_threshold).values()
        num_tp += tp
        num_fp += fp
        num_fn += fn

    # Calculate precision and recall over all images
    precision = calculate_precision(num_tp, num_fp, num_fn)
    recall = calculate_recall(num_tp, num_fp, num_fn)

    return precision, recall


def get_precision_recall_curve(all_prediction_boxes, all_gt_boxes, confidence_scores, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates the recall-precision curve over all images.
       for a single image.

       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        scores: (list of np.array of floats): each element in the list
            is a np.array containting the confidence score for each of the
            predicted bounding box. Shape: [number of predicted boxes]

            E.g: score[0][1] is the confidence score for a predicted bounding box 1 in image 0.
    Returns:
        precisions, recalls: two np.ndarray with same shape.
    """
    # Instead of going over every possible confidence score threshold to compute the PR
    # curve, we will use an approximation
    confidence_thresholds = np.linspace(0, 1, 500)
    precisions = []
    recalls = []

    # iterate over all thresholds
    for threshold in confidence_thresholds:
        # get prediction boxes over confidence threshold
        pred_boxes = []
        for img_pred_boxes, img_scores in zip(all_prediction_boxes, confidence_scores):
            img_idx = img_scores >= threshold
            pred_boxes.append(img_pred_boxes[img_idx])

        # calculate precision and recall for given confidence threshold
        precision, recall = calculate_precision_recall_all_images(pred_boxes, all_gt_boxes, iou_threshold)
        precisions.append(precision)
        recalls.append(recall)

    return np.array(precisions), np.array(recalls)


def plot_precision_recall_curve(precisions, recalls):
    """Plots the precision recall curve.
        Save the figure to precision_recall_curve.png:
        'plt.savefig("precision_recall_curve.png")'

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        None
    """
    plt.figure(figsize=(20, 20))
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.8, 1.0])
    plt.ylim([0.8, 1.0])
    plt.savefig("precision_recall_curve.png")


def calculate_mean_average_precision(precisions, recalls):
    """ Given a precision recall curve, calculates the mean average
        precision.

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        float: mean average precision
    """
    # Calculate the mean average precision given these recall levels.
    recall_levels = np.linspace(0, 1.0, 11)

    p_interp = []
    for recall in recall_levels:
        # find maximum precision value over the current recall level (add 0, if no precision values left)
        r_idx = recalls >= recall
        p_interp.append(max(precisions[r_idx]) if len(precisions[r_idx]) else 0)

    average_precision = sum(p_interp) / len(p_interp)
    return average_precision


def mean_average_precision(ground_truth_boxes, predicted_boxes):
    """ Calculates the mean average precision over the given dataset
        with IoU threshold of 0.5

    Args:
        ground_truth_boxes: (dict)
        {
            "img_id1": (np.array of float). Shape [number of GT boxes, 4]
        }
        predicted_boxes: (dict)
        {
            "img_id1": {
                "boxes": (np.array of float). Shape: [number of pred boxes, 4],
                "scores": (np.array of float). Shape: [number of pred boxes]
            }
        }
    """
    # DO NOT EDIT THIS CODE
    all_gt_boxes = []
    all_prediction_boxes = []
    confidence_scores = []

    for image_id in ground_truth_boxes.keys():
        pred_boxes = predicted_boxes[image_id]["boxes"]
        scores = predicted_boxes[image_id]["scores"]

        all_gt_boxes.append(ground_truth_boxes[image_id])
        all_prediction_boxes.append(pred_boxes)
        confidence_scores.append(scores)

    precisions, recalls = get_precision_recall_curve(
        all_prediction_boxes, all_gt_boxes, confidence_scores, 0.5)
    plot_precision_recall_curve(precisions, recalls)
    mean_average_precision = calculate_mean_average_precision(precisions, recalls)
    print("Mean average precision: {:.4f}".format(mean_average_precision))


if __name__ == "__main__":
    ground_truth_boxes = read_ground_truth_boxes()
    predicted_boxes = read_predicted_boxes()
    mean_average_precision(ground_truth_boxes, predicted_boxes)
