import numpy as np


def convertTabToPitchVector(tab):
    """
    Converts a guitar tablature representation to a pitch vector representation.

    Args:
        tab (numpy.ndarray): A 6x21 array representing the guitar tablature, where each row
                             corresponds to a string, and each column corresponds to a fret position.

    Returns:
        numpy.ndarray: A 44-dimensional vector representing the pitch classes, with 1 indicating
                       the presence of a pitch class and 0 otherwise.
    """
    pitch_vector = np.zeros(44)
    string_pitches = [40, 45, 50, 55, 59, 64]  # MIDI pitches for open strings

    for string_num, fret_vector in enumerate(tab):
        # Index of the highest activated fret
        fret_class = np.argmax(fret_vector)

        if fret_class > 0:  # If the string is not open
            # Calculate pitch class index
            pitch_num = fret_class + string_pitches[string_num] - 41
            pitch_vector[pitch_num] = 1

    return pitch_vector


def convertTabToBinary(tab):
    """
    Converts a guitar tablature representation to a binary representation.

    Args:
        tab (numpy.ndarray): A 6x21 array representing the guitar tablature, where each row
                             corresponds to a string, and each column corresponds to a fret position.

    Returns:
        numpy.ndarray: A 6x20 array representing the binary tablature, where each row corresponds
                       to a string, and each column corresponds to a fret position (excluding the
                       open string position).
    """
    tab_arr = np.zeros((6, 20))

    for string_num, fret_vector in enumerate(tab):
        # Index of the highest activated fret
        fret_class = np.argmax(fret_vector)

        if fret_class > 0:  # If the string is not open
            fret_num = fret_class - 1  # Convert fret class to fret number
            tab_arr[string_num, fret_num] = 1

    return tab_arr


def pitch_precision(pred, gt):
    """
    Calculates the precision of the pitch class predictions.

    Args:
        pred (numpy.ndarray): A list of 6x21 arrays representing the predicted guitar tablature.
        gt (numpy.ndarray): A list of 6x21 arrays representing the ground truth guitar tablature.

    Returns:
        float: The precision of the pitch class predictions.
    """
    pitch_pred = np.array([convertTabToPitchVector(p) for p in pred])
    pitch_gt = np.array([convertTabToPitchVector(gt) for gt in gt])

    numerator = np.sum(np.multiply(pitch_pred, pitch_gt).flatten())
    denominator = np.sum(pitch_pred.flatten())

    return numerator / denominator


def pitch_recall(pred, gt):
    """
    Calculates the recall of the pitch class predictions.

    Args:
        pred (numpy.ndarray): A list of 6x21 arrays representing the predicted guitar tablature.
        gt (numpy.ndarray): A list of 6x21 arrays representing the ground truth guitar tablature.

    Returns:
        float: The recall of the pitch class predictions.
    """
    pitch_pred = np.array([convertTabToPitchVector(p) for p in pred])
    pitch_gt = np.array([convertTabToPitchVector(gt) for gt in gt])

    numerator = np.sum(np.multiply(pitch_pred, pitch_gt).flatten())
    denominator = np.sum(pitch_gt.flatten())

    return numerator / denominator


def pitch_f_measure(pred, gt):
    """
    Calculates the F-measure of the pitch class predictions.

    Args:
        pred (numpy.ndarray): A list of 6x21 arrays representing the predicted guitar tablature.
        gt (numpy.ndarray): A list of 6x21 arrays representing the ground truth guitar tablature.

    Returns:
        float: The F-measure of the pitch class predictions.
    """
    p = pitch_precision(pred, gt)
    r = pitch_recall(pred, gt)

    return (2 * p * r) / (p + r)


def tab_precision(pred, gt):
    """
    Calculates the precision of the binary tablature predictions.

    Args:
        pred (numpy.ndarray): A list of 6x21 arrays representing the predicted guitar tablature.
        gt (numpy.ndarray): A list of 6x21 arrays representing the ground truth guitar tablature.

    Returns:
        float: The precision of the binary tablature predictions.
    """
    tab_pred = np.array([convertTabToBinary(p) for p in pred])
    tab_gt = np.array([convertTabToBinary(gt) for gt in gt])

    numerator = np.sum(np.multiply(tab_pred, tab_gt).flatten())
    denominator = np.sum(tab_pred.flatten())

    return numerator / denominator


def tab_recall(pred, gt):
    """
    Calculates the recall of the binary tablature predictions.

    Args:
        pred (numpy.ndarray): A list of 6x21 arrays representing the predicted guitar tablature.
        gt (numpy.ndarray): A list of 6x21 arrays representing the ground truth guitar tablature.

    Returns:
        float: The recall of the binary tablature predictions.
    """
    tab_pred = np.array([convertTabToBinary(p) for p in pred])
    tab_gt = np.array([convertTabToBinary(gt) for gt in gt])

    numerator = np.sum(np.multiply(tab_pred, tab_gt).flatten())
    denominator = np.sum(tab_gt.flatten())

    return numerator / denominator


def tab_f_measure(pred, gt):
    """
    Calculates the F-measure of the binary tablature predictions.

    Args:
        pred (numpy.ndarray): A list of 6x21 arrays representing the predicted guitar tablature.
        gt (numpy.ndarray): A list of 6x21 arrays representing the ground truth guitar tablature.

    Returns:
        float: The F-measure of the binary tablature predictions.
    """
    p = tab_precision(pred, gt)
    r = tab_recall(pred, gt)

    return (2 * p * r) / (p + r)


def tab_disamb(pred, gt):
    """
    Calculates the disambiguation ratio between the binary tablature precision and
    the pitch class precision.

    Args:
        pred (numpy.ndarray): A list of 6x21 arrays representing the predicted guitar tablature.
        gt (numpy.ndarray): A list of 6x21 arrays representing the ground truth guitar tablature.

    Returns:
        float: The ratio of binary tablature precision to pitch class precision.
    """
    tp = tab_precision(pred, gt)
    pp = pitch_precision(pred, gt)

    return tp / pp
