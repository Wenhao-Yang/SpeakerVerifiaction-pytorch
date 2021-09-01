"""
@Overview: There are errors for computing eer and acc, when it comes to l2 and cosine distance.
For l2 distacne: when the distance is less than the theshold, it should be true;
For cosine distance: when the distance is greater than the theshold, it's true.
"""
import os
from operator import itemgetter
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.stats import norm

from Process_Data.constants import cValue_1


def evaluate(distances, labels):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 30, 0.01)
    tpr, fpr, accuracy = calculate_roc(thresholds, distances,
        labels)

    thresholds = np.arange(0, 30, 0.001)
    val,  far = calculate_val(thresholds, distances,
        labels, 1e-3)

    return tpr, fpr, accuracy, val,  far

def calculate_roc(thresholds, distances, labels):

    nrof_pairs = min(len(labels), len(distances))
    nrof_thresholds = len(thresholds)

    tprs = np.zeros((nrof_thresholds))
    fprs = np.zeros((nrof_thresholds))
    acc_train = np.zeros((nrof_thresholds))
    accuracy = 0.0

    indices = np.arange(nrof_pairs)

    # Find the best threshold for the fold

    for threshold_idx, threshold in enumerate(thresholds):
        tprs[threshold_idx], fprs[threshold_idx], acc_train[threshold_idx] = calculate_accuracy(threshold, distances, labels)
    best_threshold_index = np.argmax(acc_train)

    return tprs[best_threshold_index], fprs[best_threshold_index], acc_train[best_threshold_index]

def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp+fn==0) else float(tp) / float(tp+fn)
    fpr = 0 if (fp+tn==0) else float(fp) / float(fp+tn)
    #fnr = 0 if (tp+fn==0) else float(fn) / float(tp+fn)

    acc = float(tp+tn)/dist.size
    return tpr, fpr, acc

def calculate_eer(thresholds, distances, labels):

    nrof_pairs = min(len(labels), len(distances))
    nrof_thresholds = len(thresholds)

    tprs = np.zeros((nrof_thresholds))
    fprs = np.zeros((nrof_thresholds))
    fnrs = np.zeros((nrof_thresholds))
    acc_train = np.zeros((nrof_thresholds))
    accuracy = 0.0

    indices = np.arange(nrof_pairs)

    # Find the threshold where fnr=fpr for the fold
    # Todo: And the highest accuracy??
    eer_index = 0
    fpr_fnr = 1.0
    for threshold_idx, threshold in enumerate(thresholds):
        tprs[threshold_idx], fprs[threshold_idx], fnrs[threshold_idx], acc_train[threshold_idx] = calculate_eer_accuracy(threshold, distances, labels)
        if np.abs(fprs[threshold_idx]-fnrs[threshold_idx])<fpr_fnr:
            eer_index = threshold_idx
            fpr_fnr = np.abs(fprs[threshold_idx]-fnrs[threshold_idx])

    #print("Threshold for the eer is {}.".format(thresholds[eer_index]))
    return  fnrs[eer_index], acc_train[eer_index]

def calculate_eer_accuracy(threshold, dist, actual_issame):
    predict_issame = np.greater(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)

    fnr = 0 if (tp + fn == 0) else float(fn) / float(tp + fn)

    acc = float(tp + tn) / dist.size
    return tpr, fpr, fnr, acc

def calculate_val(thresholds, distances, labels, far_target=0.1):
    nrof_pairs = min(len(labels), len(distances))
    nrof_thresholds = len(thresholds)

    indices = np.arange(nrof_pairs)

    # Find the threshold that gives FAR = far_target
    far_train = np.zeros(nrof_thresholds)

    for threshold_idx, threshold in enumerate(thresholds):
        _, far_train[threshold_idx] = calculate_val_far(threshold, distances, labels)
    if np.max(far_train)>=far_target:
        f = interpolate.interp1d(far_train, thresholds, kind='slinear')
        threshold = f(far_target)
    else:
        threshold = 0.0

    val, far = calculate_val_far(threshold, distances, labels)

    return val, far

def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    if n_diff == 0:
        n_diff = 1
    if n_same == 0:
        return 0,0
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far


def evaluate_kaldi_eer(distances, labels, cos=True, re_thre=False):
    """
    The distance score should be larger when two samples are more similar.
    :param distances:
    :param labels:
    :param cos:
    :return:
    """
    # split the target and non-target distance array
    target = []
    non_target = []
    new_distances = []

    for (distance, label) in zip(distances, labels):
        if not cos:
            distance = -distance

        new_distances.append(-distance)
        if label:
            target.append(distance)
        else:
            non_target.append(distance)

    new_distances = np.array(new_distances)
    target = np.sort(target)
    non_target = np.sort(non_target)

    target_size = target.size
    nontarget_size = non_target.size
    # pdb.set_trace()
    target_position = 0
    while target_position + 1 < target_size:
        # for target_position in range(target_size):
        nontarget_n = nontarget_size * target_position * 1.0 / target_size
        nontarget_position = int(nontarget_size - 1 - nontarget_n)

        if (nontarget_position < 0):
            nontarget_position = 0
        # The exceptions from non targets are samples where cosine score is > the target score
        # if (non_target[nontarget_position] <= target[target_position]):
        #     break
        if (non_target[nontarget_position] < target[target_position]):
            # print('target[{}]={} is < non_target[{}]={}.'.format(target_position, target[target_position], nontarget_position, non_target[nontarget_position]))
            break
        target_position += 1

    eer_threshold = target[target_position]
    eer = target_position * 1.0 / target_size

    # max_threshold = np.max(distances)
    # thresholds = np.arange(0, max_threshold, 0.001)
    thresholds = np.sort(np.unique(target))
    tpr, fpr, best_accuracy = calculate_roc(thresholds, new_distances, labels)

    # return eer threshold.
    if re_thre:
        return eer, eer_threshold, best_accuracy
    return eer, best_accuracy


# Creates a list of false-negative rates, a list of false-positive rates
# and a list of decision thresholds that give those error-rates.
def ComputeErrorRates(scores, labels):
    # Sort the scores from smallest to largest, and also get the corresponding
    # indexes of the sorted scores.  We will treat the sorted scores as the
    # thresholds at which the the error-rates are evaluated.
    sorted_indexes, thresholds = zip(*sorted([(index, threshold) for index, threshold in enumerate(scores)],
                                             key=itemgetter(1)))
    sorted_labels = []
    labels = [int(labels[i]) for i in sorted_indexes]
    fnrs = []  # 小于阈值的正例数目
    fprs = []  # 小于阈值的反例数目

    # At the end of this loop, fnrs[i] is the number of errors made by
    # incorrectly rejecting scores less than thresholds[i]. And, fprs[i]
    # is the total number of times that we have correctly accepted scores
    # greater than thresholds[i].
    for i in range(0, len(labels)):
        if i == 0:
            fnrs.append(labels[i])
            fprs.append(1 - labels[i])
        else:
            fnrs.append(fnrs[i - 1] + labels[i])
            fprs.append(fprs[i - 1] + 1 - labels[i])

    fnrs_norm = sum(labels)  # 样本中的正例数目
    fprs_norm = len(labels) - fnrs_norm  # 样本中的反例数目

    # Now divide by the total number of false negative errors to obtain the false positive rates across all thresholds.
    # 小于阈值而被认为是反例的正例在所有正例的样本比重
    fnrs = [x / float(fnrs_norm) for x in fnrs]

    # Divide by the total number of corret positives to get the true positive rate.
    # Subtract these quantities from 1 to get the false positive rates.
    # 大于阈值而被认为是正例的反例在所有反例中的样本比重

    fprs = [1 - x / float(fprs_norm) for x in fprs]

    return fnrs, fprs, thresholds


# Computes the minimum of the detection cost function.  The comments refer to
# equations in Section 3 of the NIST 2016 Speaker Recognition Evaluation Plan.
def ComputeMinDcf(fnrs, fprs, thresholds, p_target, c_miss, c_fa):
    """
    :param fnrs: 正例的错误拒绝率
    :param fprs: 反例的错误接受率
    :param thresholds: 判断的阈值
    :param p_target: a priori probability of the specified target speaker
    :param c_miss: cost of a missed detection 遗漏正例的损失值
    :param c_fa: cost of a spurious detection 错误接受的损失值
    :return:
    """
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]
    for i in range(0, len(fnrs)):
        # See Equation (2).  it is a weighted sum of false negative
        # and false positive errors.
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            # 找到最小的det值
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    # See Equations (3) and (4).  Now we normalize the cost.
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf, min_c_det_threshold


def evaluate_kaldi_mindcf(scores, labels, return_threshold=False):
    c_miss = 1
    c_fa = 1
    labels = [int(x) for x in labels]
    fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)

    p_target = 0.01
    mindcf_01, threshold_01 = ComputeMinDcf(fnrs, fprs, thresholds, p_target, c_miss, c_fa)

    p_target = 0.001
    mindcf_001, threshold_001 = ComputeMinDcf(fnrs, fprs, thresholds, p_target, c_miss, c_fa)

    if return_threshold:
        return (mindcf_01, threshold_01, mindcf_001, threshold_001)

    return mindcf_01, mindcf_001

# from https://blog.csdn.net/qq_28228605/article/details/103728793
def plot_DET_curve(pf_max=0.3):
    # 设置刻度范围
    pmiss_min = 0.005
    pmiss_max = pf_max
    pfa_min = 0.005
    pfa_max = pf_max

    # 刻度设置
    pticks = [0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002,
              0.005, 0.01, 0.02, 0.03, 0.05, 0.07,
              0.1, 0.15, 0.25, 0.4, 0.8, 0.9,
              0.95, 0.98, 0.99, 0.995, 0.998, 0.999,
              0.9995, 0.9998, 0.9999, 0.99995, 0.99998, 0.99999]

    # 刻度*100
    xlabels = [' 0.005', ' 0.01 ', ' 0.02 ', ' 0.05 ', '  0.1 ', '  0.2 ',
               ' 0.5  ', '  1   ', '  2   ', '  3   ', '  5   ', '  7   ',
               '  10  ', '  15  ', '  25  ', '  40  ', '  80  ', '  90  ',
               '  95  ', '  98  ', '  99  ', ' 99.5 ', ' 99.8 ', ' 99.9 ',
               ' 99.95', ' 99.98', ' 99.99', '99.995', '99.998', '99.999']

    ylabels = xlabels

    # 确定刻度范围
    n = len(pticks)
    # 倒叙
    for k, v in enumerate(pticks[::-1]):
        if pmiss_min <= v:
            tmin_miss = n - k - 1  # 移动最小值索引位置
        if pfa_min <= v:
            tmin_fa = n - k - 1  # 移动最小值索引位置
    # 正序
    for k, v in enumerate(pticks):
        if pmiss_max >= v:
            tmax_miss = k + 1  # 移动最大值索引位置
        if pfa_max >= v:
            tmax_fa = k + 1  # 移动最大值索引位置

    # FRR
    plt.figure(figsize=(12, 12))
    # plt.rc('font', family='Times New Roman')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
    plt.rcParams['axes.unicode_minus'] = False
    plt.title('DET', fontsize=22)
    plt.xlim(norm.ppf(pfa_min), norm.ppf(pfa_max))

    plt.xticks(norm.ppf(pticks[tmin_fa:tmax_fa]), xlabels[tmin_fa:tmax_fa])
    plt.xlabel('False Alarm probability (in %)', fontsize=18)
    plt.xticks(fontsize=16)

    # FAR
    plt.ylim(norm.ppf(pmiss_min), norm.ppf(pmiss_max))
    plt.yticks(norm.ppf(pticks[tmin_miss:tmax_miss]), ylabels[tmin_miss:tmax_miss])
    plt.ylabel('Miss probability (in %)', fontsize=18)
    plt.yticks(fontsize=16)
    plt.grid()
    plt.plot([-40, 1], [-40, 1], alpha=0.5, color='gray', linestyle='--', linewidth=1)

    return plt


def save_det(save_path, score_files=[], names=[], pf_max=0.3):
    if len(score_files) > 0 and len(score_files) == len(names):
        det_plt = plot_DET_curve(pf_max=pf_max)

        for i, scf in enumerate(score_files):
            if os.path.exists(scf):
                print(scf)
                scores, labels = [], []
                with open(scf, 'r') as f:
                    for line in f.readlines():
                        score, label = line.split()
                        try:
                            s = float(score)
                            l = int(label)
                        except:
                            l = int(score)
                            s = float(label)
                        scores.append(s)
                        labels.append(l)

                fnrs, fprs, _ = ComputeErrorRates(scores, labels)
                x, y = norm.ppf(fnrs), norm.ppf(fprs)
                det_plt.plot(x, y, label=names[i], color=cValue_1[i])

        det_plt.legend(loc='upper right', fontsize=18)
        det_plt.savefig(save_path + "/det.png")

# save_det(save_path="Data/xvector/LoResNet8/timit/spect_egs_None",
#          score_files=["Data/xvector/LoResNet8/timit/spect_egs_None/soft_dp05/scores",
#                       "Data/xvector/LoResNet8/timit/spect_egs_None/center_dp05/scores",
#                       "Data/xvector/LoResNet8/timit/spect_egs_None/coscenter_dp05/scores",
#                       "Data/xvector/LoResNet8/timit/spect_egs_None/gaussian_cov_dp05/scores",
#                       "Data/xvector/LoResNet8/timit/spect_egs_None/gaussian_dp05/scores"],
#         names=["soft", "center", "coscenter", "gasscov", "gass"])
# # fnrs, fprs, = ComputeErrorRates(scores, labels)
# x, y = norm.ppf(fnrs), norm.ppf(fprs)
# plt.plot(x, y)
# plt.plot([-40, 1], [-40, 1])
# plt.plot(np.arange(0,40,1),np.arange(0,40,1))
# plt.show()

# from asv-tools
def load_score(score_path, names, sep=" "):
    df = pd.read_csv(score_path, sep=sep, names=names, converters={0: str, 1: str, 2: float})
    return df


def save_score(score, score_path, sep=" "):
    df = pd.DataFrame(score)
    df.to_csv(score_path, header=None, sep=sep, index=False)


def snorm(input_score, enroll_cohort_score, test_cohort_score, output_score,
          second_cohort=True):
    """ Symmetrical Normalization.
    Reference: Kenny, P. (2010). Bayesian speaker verification with heavy-tailed priors. Paper presented at the Odyssey.
    """
    enroll_test_names = ["enroll", "test", "score"]

    if second_cohort:
        enroll_cohort_names = ["enroll", "cohort", "score"]
        test_cohort_names = ["test", "cohort", "score"]
    else:
        enroll_cohort_names = ["cohort", "enroll", "score"]
        test_cohort_names = ["cohort", "test", "score"]

    input_score = load_score(input_score, enroll_test_names)
    enroll_cohort_score = load_score(enroll_cohort_score, enroll_cohort_names)
    test_cohort_score = load_score(test_cohort_score, test_cohort_names)

    output_score = []

    # This .groupby function is really an efficient method than 'for' grammar.
    enroll_group = enroll_cohort_score.groupby("enroll")
    test_group = test_cohort_score.groupby("test")

    enroll_mean = enroll_group["score"].mean()
    enroll_std = enroll_group["score"].std()
    test_mean = test_group["score"].mean()
    test_std = test_group["score"].std()

    for _, row in input_score.iterrows():
        enroll_key, test_key, score = row
        normed_score = 0.5 * ((score - enroll_mean[enroll_key]) / enroll_std[enroll_key] + \
                              (score - test_mean[test_key]) / test_std[test_key])
        output_score.append([enroll_key, test_key, normed_score])

    # logger.info("Normalize scores done.")
    save_score(output_score, output_score)


def asnorm(input_score, enroll_cohort_score, test_cohort_score, output_score,
           top_n=300,
           second_cohort=True, cross_select=False):
    """ Adaptive Symmetrical Normalization.
    Reference: Cumani, S., Batzu, P. D., Colibro, D., Vair, C., Laface, P., & Vasilakakis, V. (2011). Comparison of
               speaker recognition approaches for real applications. Paper presented at the Twelfth Annual Conference
               of the International Speech Communication Association.

               Cai, Danwei, et al. “The DKU-SMIIP System for NIST 2018 Speaker Recognition Evaluation.” Interspeech 2019,
               2019, pp. 4370–4374.

    Recommend: Matejka, P., Novotný, O., Plchot, O., Burget, L., Sánchez, M. D., & Cernocký, J. (2017). Analysis of
               Score Normalization in Multilingual Speaker Recognition. Paper presented at the Interspeech.

    """
    enroll_test_names = ["enroll", "test", "score"]

    if second_cohort:
        enroll_cohort_names = ["enroll", "cohort", "score"]
        test_cohort_names = ["test", "cohort", "score"]
    else:
        enroll_cohort_names = ["cohort", "enroll", "score"]
        test_cohort_names = ["cohort", "test", "score"]

    input_score = load_score(input_score, enroll_test_names)
    enroll_cohort_score = load_score(enroll_cohort_score, enroll_cohort_names)
    test_cohort_score = load_score(test_cohort_score, test_cohort_names)

    output_score = []

    # Note that, .sort_values function will return NoneType with inplace=True and .head function will return a DataFrame object.
    # The order sort->groupby is equal to groupby->sort, so there is no problem about independence of trials.
    enroll_cohort_score.sort_values(by="score", ascending=False, inplace=True)
    test_cohort_score.sort_values(by="score", ascending=False, inplace=True)

    if cross_select == "true":
        # The SQL grammar is used to implement the cross selection based on pandas.
        # Let A is enroll_test table, B is enroll_cohort table and C is test_cohort table.
        # To get a test_group (select "test:cohort" pairs) where the cohort utterances' scores is selected by enroll_top_n,
        # we should get the D table by concatenating AxC with "enroll" key firstly and then
        # we could get the target E table by concatenating BxD wiht "test"&"cohort" key.
        # Finally, the E table should be grouped by "enroll"&"test" key to make sure the group key is unique.
        enroll_top_n = enroll_cohort_score.groupby("enroll").head(top_n)[["enroll", "cohort"]]
        test_group = pd.merge(pd.merge(input_score[["enroll", "test"]], enroll_top_n, on="enroll"),
                              test_cohort_score, on=["test", "cohort"]).groupby(["enroll", "test"])

        test_top_n = test_cohort_score.groupby("test").head(top_n)[["test", "cohort"]]
        enroll_group = pd.merge(pd.merge(input_score[["enroll", "test"]], test_top_n, on="test"),
                                enroll_cohort_score, on=["enroll", "cohort"]).groupby(["enroll", "test"])
    else:
        enroll_group = enroll_cohort_score.groupby("enroll").head(top_n).groupby("enroll")
        test_group = test_cohort_score.groupby("test").head(top_n).groupby("test")

    enroll_mean = enroll_group["score"].mean()
    enroll_std = enroll_group["score"].std()
    test_mean = test_group["score"].mean()
    test_std = test_group["score"].std()

    for _, row in input_score.iterrows():
        enroll_key, test_key, score = row
        if cross_select == "true":
            normed_score = 0.5 * ((score - enroll_mean[enroll_key, test_key]) / enroll_std[enroll_key, test_key] + \
                                  (score - test_mean[enroll_key, test_key]) / test_std[enroll_key, test_key])
        else:
            normed_score = 0.5 * ((score - enroll_mean[enroll_key]) / enroll_std[enroll_key] + \
                                  (score - test_mean[test_key]) / test_std[test_key])
        output_score.append([enroll_key, test_key, normed_score])

    save_score(output_score, output_score)
