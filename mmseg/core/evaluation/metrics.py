import mmcv
import numpy as np
from mmcv.utils import print_log
from sklearn.metrics import roc_auc_score


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def beforeval(self):
        isval = np.sum(self.confusion_matrix,axis=1)>0
        self.confusion_matrix = self.confusion_matrix*isval

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()

        return Acc

    def Pixel_Accuracy_Class(self):

        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc


    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        isval = np.sum(self.confusion_matrix,axis=1)>0
        MIoU = np.nansum(MIoU*isval)/isval.sum()
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)

        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix
    
    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


def intersect_and_union(pred_label,
                        label,
                        num_classes,
                        ignore_index,
                        label_map=dict(),
                        reduce_zero_label=False):
    """Calculate intersection and Union.

    Args:
        pred_label (ndarray): Prediction segmentation map.
        label (ndarray): Ground truth segmentation map.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. The parameter will
            work only when label is str. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. The parameter will
            work only when label is str. Default: False.

     Returns:
         ndarray: The intersection of prediction and ground truth histogram
             on all classes.
         ndarray: The union of prediction and ground truth histogram on all
             classes.
         ndarray: The prediction histogram on all classes.
         ndarray: The ground truth histogram on all classes.
    """

    if isinstance(pred_label, str):
        pred_label = np.load(pred_label)

    if isinstance(label, str):
        label = mmcv.imread(label, flag='unchanged', backend='pillow')
    # modify if custom classes
    if label_map is not None:
        for old_id, new_id in label_map.items():
            label[label == old_id] = new_id
    if reduce_zero_label:
        # avoid using underflow conversion
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255

    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]

    intersect = pred_label[pred_label == label]
    area_intersect, _ = np.histogram(
        intersect, bins=np.arange(num_classes + 1))
    area_pred_label, _ = np.histogram(
        pred_label, bins=np.arange(num_classes + 1))
    area_label, _ = np.histogram(label, bins=np.arange(num_classes + 1))
    area_union = area_pred_label + area_label - area_intersect

    return area_intersect, area_union, area_pred_label, area_label


def intersect_and_union2(pred_label,
                        label,
                        num_classes,
                        ignore_index,
                        label_map=dict(),
                        reduce_zero_label=False,
                        evaluator=None):
    """Calculate intersection and Union.

    Args:
        pred_label (ndarray): Prediction segmentation map.
        label (ndarray): Ground truth segmentation map.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. The parameter will
            work only when label is str. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. The parameter will
            work only when label is str. Default: False.

     Returns:
         ndarray: The intersection of prediction and ground truth histogram
             on all classes.
         ndarray: The union of prediction and ground truth histogram on all
             classes.
         ndarray: The prediction histogram on all classes.
         ndarray: The ground truth histogram on all classes.
    """

    if isinstance(pred_label, str):
        pred_label = np.load(pred_label)

    if isinstance(label, str):
        label = mmcv.imread(label, flag='unchanged', backend='pillow')
    # modify if custom classes
    if label_map is not None:
        for old_id, new_id in label_map.items():
            label[label == old_id] = new_id
    if reduce_zero_label:
        # avoid using underflow conversion
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255

    evaluator.add_batch(label, pred_label)
    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]

    intersect = pred_label[pred_label == label]
    area_intersect, _ = np.histogram(
        intersect, bins=np.arange(num_classes + 1))
    area_pred_label, _ = np.histogram(
        pred_label, bins=np.arange(num_classes + 1))
    area_label, _ = np.histogram(label, bins=np.arange(num_classes + 1))
    area_union = area_pred_label + area_label - area_intersect

    return area_intersect, area_union, area_pred_label, area_label

def total_intersect_and_union(results,
                              gt_seg_maps,
                              num_classes,
                              ignore_index,
                              label_map=dict(),
                              reduce_zero_label=False):
    """Calculate Total Intersection and Union.

    Args:
        results (list[ndarray]): List of prediction segmentation maps.
        gt_seg_maps (list[ndarray]): list of ground truth segmentation maps.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. Default: False.

     Returns:
         ndarray: The intersection of prediction and ground truth histogram
             on all classes.
         ndarray: The union of prediction and ground truth histogram on all
             classes.
         ndarray: The prediction histogram on all classes.
         ndarray: The ground truth histogram on all classes.
    """

    num_imgs = len(results)
    assert len(gt_seg_maps) == num_imgs
    total_area_intersect = np.zeros((num_classes, ), dtype=np.float)
    total_area_union = np.zeros((num_classes, ), dtype=np.float)
    total_area_pred_label = np.zeros((num_classes, ), dtype=np.float)
    total_area_label = np.zeros((num_classes, ), dtype=np.float)
    evaluator = Evaluator(num_classes)
    evaluator.reset()

   
    return total_area_intersect, total_area_union, \
        total_area_pred_label, total_area_label


def mean_iou(results,
             gt_seg_maps,
             num_classes,
             ignore_index,
             nan_to_num=None,
             label_map=dict(),
             reduce_zero_label=False):
    """Calculate Mean Intersection and Union (mIoU)

    Args:
        results (list[ndarray]): List of prediction segmentation maps.
        gt_seg_maps (list[ndarray]): list of ground truth segmentation maps.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. Default: False.

     Returns:
         float: Overall accuracy on all images.
         ndarray: Per category accuracy, shape (num_classes, ).
         ndarray: Per category IoU, shape (num_classes, ).
    """

    all_acc, acc, iou = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mIoU'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label)
    return all_acc, acc, iou



def mean_dice(results,
              gt_seg_maps,
              num_classes,
              ignore_index,
              nan_to_num=None,
              label_map=dict(),
              reduce_zero_label=False):
    """Calculate Mean Dice (mDice)

    Args:
        results (list[ndarray]): List of prediction segmentation maps.
        gt_seg_maps (list[ndarray]): list of ground truth segmentation maps.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. Default: False.

     Returns:
         float: Overall accuracy on all images.
         ndarray: Per category accuracy, shape (num_classes, ).
         ndarray: Per category dice, shape (num_classes, ).
    """

    all_acc, acc, dice = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mDice'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label)
    return all_acc, acc, dice




def eval_metrics(results,
                 gt_seg_maps,
                 num_classes,
                 ignore_index,
                 metrics=['mIoU'],
                 nan_to_num=None,
                 label_map=dict(),
                 reduce_zero_label=False):
    """Calculate evaluation metrics
    Args:
        results (list[ndarray]): List of prediction segmentation maps.
        gt_seg_maps (list[ndarray]): list of ground truth segmentation maps.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. Default: False.
     Returns:
         float: Overall accuracy on all images.
         ndarray: Per category accuracy, shape (num_classes, ).
         ndarray: Per category evalution metrics, shape (num_classes, ).
    """
    
    if isinstance(metrics, str):
        metrics = [metrics]
    allowed_metrics = ['mIoU', 'mDice']
    if not set(metrics).issubset(set(allowed_metrics)):
        raise KeyError('metrics {} is not supported'.format(metrics))
    


    preds = []
    logits = []


    for tensor in results:
        preds.append(int(tensor[0].item()))
        logits.append(tensor[1].item())




    binary_labels = [1 if 2 in seg_map else 0 for seg_map in gt_seg_maps]
    print("preds: ", preds)
    print("label: ", binary_labels)
    print("len of preds: ", len(preds))

    

    TP, FP, FN, TN = 0, 0, 0, 0


    for i, _ in enumerate(results):
        if preds[i] and binary_labels[i]:
            TP += 1
        elif preds[i] and not binary_labels[i]:
            FP += 1
        elif not preds[i] and binary_labels[i]:
            FN += 1
        elif not preds[i] and not binary_labels[i]:
            TN += 1


    
    mvo_accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    mvo_recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    mvo_precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    mvo_specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

    f1_score = 2 * mvo_precision * mvo_recall / (mvo_precision + mvo_recall) if (mvo_precision + mvo_recall) > 0 else 0
    auc = roc_auc_score(binary_labels, logits)
    
    total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label = total_intersect_and_union(results, gt_seg_maps,
                                                     num_classes, ignore_index,
                                                     label_map,
                                                     reduce_zero_label)

    ret_metrics = []
    for metric in metrics:
        if metric == 'mIoU':
            iou = total_area_intersect / total_area_union
            precision = total_area_intersect / total_area_pred_label
            dice = 2 * total_area_intersect / (
                total_area_pred_label + total_area_label)
            ret_metrics.append(TP)
            ret_metrics.append(FP)
            ret_metrics.append(FN)
            ret_metrics.append(TN)
            ret_metrics.append(mvo_specificity)
            ret_metrics.append(mvo_recall)
            ret_metrics.append(mvo_precision)
            ret_metrics.append(mvo_accuracy)
            ret_metrics.append(f1_score)
            ret_metrics.append(auc)
                
    
        elif metric == 'mDice':
            dice = 2 * total_area_intersect / (
                total_area_pred_label + total_area_label)
            ret_metrics.append(dice)


            
    if nan_to_num is not None:
        ret_metrics = [
            np.nan_to_num(metric, nan=nan_to_num) for metric in ret_metrics
        ]
    return ret_metrics