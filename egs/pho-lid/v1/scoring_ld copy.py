import os
from sklearn.metrics import balanced_accuracy_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np

def get_trials(utt2lan, num_lang, output):
    with open(utt2lan, 'r') as f:
        lines = f.readlines()
    utt_list = [os.path.split(x.split(sep='\t')[0])[-1].strip('.npy') for x in lines]

    lang_list = [x.split(sep='\t')[1].strip().replace('[', '').replace(']', '').split(sep=', ') for x in lines]
    
    float_lang_list = []
    for row in lang_list:
      float_row = []
      for label in row:
         if label.replace('.', '').isnumeric():
            float_row.append(int(float(label)))
      float_lang_list.append(float_row)
    lang_list = float_lang_list


    targets = [i for i in range(num_lang)]
    with open(output, 'w') as f:
        for i in range(len(utt_list)):
          for idx, j in enumerate(range(len(lang_list[i]))):
            target_utt = lang_list[i][j]
            utt = utt_list[i] + '-{}'.format(idx)
            for target in targets:
                if target == target_utt:
                    f.write("{} {} target\n".format(utt, target))
                else:
                    f.write("{} {} nontarget\n".format(utt, target))

def get_score(utt2lan, scores, num_lang, output):
    with open(utt2lan, 'r') as f:
        lines = f.readlines()
    utt_list = [os.path.split(x.split(sep='\t')[0])[-1].strip('.npy') for x in lines]
    lang_list = [x.split(sep='\t')[1].strip().replace('[', '').replace(']', '').split(sep=', ') for x in lines]
    
    float_lang_list = []
    for row in lang_list:
      float_row = []
      for label in row:
         if label.replace('.', '').isnumeric():
            float_row.append(int(float(label)))
      float_lang_list.append(float_row)
    lang_list = float_lang_list

    print(scores)

    targets = [i for i in range(num_lang)]
    with open(output, 'w') as f:
        for i in range(len(utt_list)):
          print(f'i={i}')
          for idx, j in enumerate(range(len(scores[i]))):
            for lang_id in targets:
              score_utt = scores[i]
              str_ = "{} {} {}\n".format(utt_list[i]+"-{}".format(idx), lang_id, score_utt[lang_id])
              f.write(str_)

def get_langid_dict(trials):
  ''' Get lang2lang_id, utt2lang_id dicts and lang nums, lang_id starts from 0.
      Also return trial list.
  '''
  langs = []
  lines = open(trials, 'r').readlines()
  for line in lines:
    utt, lang, target = line.strip().split()
    langs.append(lang)

  langs = list(set(langs))
  langs.sort()
  lang2lang_id = {}
  for i in range(len(langs)):
    lang2lang_id["{}".format(i)] = i

  utt2lang_id = {}
  trial_list = {}
  for line in lines:
    utt, lang, target = line.strip().split()
    if target == 'target':
      utt2lang_id[utt] = lang2lang_id[lang]
    trial_list[lang + utt] = target

  return lang2lang_id, utt2lang_id, len(langs), trial_list


def process_pair_scores(scores, lang2lang_id, utt2lang_id, lang_num, trial_list):
  ''' Replace both lang names and utt ids with their lang ids,
      for unknown utt, just with -1. Also return the min and max scores.
  '''
  pairs = []
  stats = []
  lines = open(scores, 'r').readlines()

  # print(trial_list)
  for line in lines:
    # print(line.strip().split())
    utt, lang, score = line.strip().split()
    if lang + utt in trial_list:
      if utt in utt2lang_id:
        pairs.append([lang2lang_id[lang], utt2lang_id[utt], float(score)])
      else:
        pairs.append([lang2lang_id[lang], -1, float(score)])
      stats.append(float(score))
  return pairs, min(stats), max(stats)


def process_matrix_scores(scores, lang2lang_id, utt2lang_id, lang_num, trial_list):
  ''' Convert matrix scores to pairs as returned by process_pair_scores.
  '''
  lines = open(scores, 'r').readlines()
  langs_order = {} # langs order in the first line of scores
  langs = lines[1].strip().split()
  for i in range(len(langs)):
    langs_order[i] = langs[i]

  pairs = []
  stats = []
  for line in lines[0:]:
    items = line.strip().split()
    utt = items[0]
    sco = items[2:]
    for i in range(len(sco)):
      if langs_order[i] + utt in trial_list:
        if utt in utt2lang_id:
          pairs.append([lang2lang_id[langs_order[i]], utt2lang_id[utt], float(sco[i])])
        else:
          pairs.append([lang2lang_id[langs_order[i]], -1, float(sco[i])])
        stats.append(float(sco[i]))
  return pairs, min(stats), max(stats)


def get_cavg(pairs, lang_num, min_score, max_score, bins = 20, p_target = 0.5):
  ''' Compute Cavg, using several threshhold bins in [min_score, max_score].
  '''
  cavgs = [0.0] * (bins + 1)
  precision = (max_score - min_score) / bins
  for section in range(bins + 1):
    threshold = min_score + section * precision
    # Cavg for each lang: p_target * p_miss + sum(p_nontarget*p_fa)
    target_cavg = [0.0] * lang_num
    for lang in range(lang_num):
      p_miss = 0.0 # prob of missing target pairs
      LTa = 0.0 # num of all target pairs
      LTm = 0.0 # num of missing pairs
      p_fa = [0.0] * lang_num # prob of false alarm, respect to all other langs
      LNa = [0.0] * lang_num # num of all nontarget pairs, respect to all other langs
      LNf = [0.0] * lang_num # num of false alarm pairs, respect to all other langs
      for line in pairs:
        if line[0] == lang:
          if line[1] == lang:
            LTa += 1
            if line[2] < threshold:
              LTm += 1
          else:
            LNa[line[1]] += 1
            if line[2] >= threshold:
              LNf[line[1]] += 1
      if LTa != 0.0:
        p_miss = LTm / LTa
      for i in range(lang_num):
        if LNa[i] != 0.0:
          p_fa[i] = LNf[i] / LNa[i]
      p_nontarget = (1 - p_target) / (lang_num - 1)
      target_cavg[lang] = p_target * p_miss + p_nontarget*sum(p_fa)
    cavgs[section] = sum(target_cavg) / lang_num
  return cavgs, min(cavgs)


def compute_cavg(trial_txt, score_txt, p_target=0.5):
    '''
    :param trail: trial file
    :param score: score file
    :param p_target: default 0.5
    :return: Cavg (average cost)
    '''
    lang2lang_id, utt2lang_id, lang_num, trial_list = get_langid_dict(trial_txt)
    pairs, min_score, max_score = process_pair_scores(score_txt, lang2lang_id, utt2lang_id, lang_num, trial_list)
    threshhold_bins = 20
    p_target = p_target
    cavgs, min_cavg = get_cavg(pairs, lang_num, min_score, max_score, threshhold_bins, p_target)
    return round(min_cavg, 4)

def compute_cprimary(trial_txt, score_txt, p_target_1=0.5, p_target_2=0.1):
    cprimary = compute_cavg(trial_txt, score_txt, p_target_1) + compute_cavg(trial_txt, score_txt, p_target_2)
    cprimary = cprimary/2
    return cprimary

def get_weighted_acc(outputs, truths, lang_num, ignore_idx=100):
  ''' Compute balanced accuracy.
  '''
  # outputs = outputs.flatten()
  # truths = truths.flatten()
  acc = list()
  weights = list()
  total = list()
  LTa = list()

  for lang in range(lang_num):
    acc.append(0)
    weights.append(0)
    total.append(0)
    LTa.append(0)

  for output, truth in zip(outputs, truths):
      t = int(truth)
      o = int(output)
      if t != ignore_idx:
        total[t] += 1
        if t == o:
          LTa[t] += 1    

  total_sum = sum(total)
  for lang in range(lang_num):
    if total[lang] > 0.0:
      acc[lang] = LTa[lang] / total[lang]
    else:
       acc[lang] = 0
  if total_sum >0:
    weights[lang] = total[lang] / total_sum 
  else:
    weights[lang] = 0
    
  b_acc = 0.0
  for lang in range(lang_num):
    b_acc += weights[lang] * acc[lang]

  # print(f'Class counts: {total}, weights: {weights}')
  return b_acc, acc, weights

def compute_wacc(outputs, truths, num_langs):
    '''
    :param trail: trial file
    :param score: score file
    :return: balanced accuracy
    '''
    bacc, acc, weights = get_weighted_acc(outputs, truths, num_langs)
    acc = [round(a, 4) for a in acc]
    weights = [round(w, 4) for w in weights]
    return round(bacc, 4), acc, weights


def get_bacc(y_true, y_pred):
  return round(balanced_accuracy_score(y_true, y_pred), 4)

def draw_roc(y, scores, fname):
  fpr, tpr, thresholds = roc_curve(y, scores)
  print(f'fpr: {fpr}\ntpr: {tpr}\nthresholds:{thresholds}')

  plt.figure()
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.0])
  plt.step(fpr, tpr, 'bo-')

  x = np.arange(0.0, 1.1, 0.1)
  y = -x + 1
  plt.plot(x, y, '--')

  plt.xlabel('FPR = 1 - speciality')
  plt.ylabel('TPR = sensitivity')
  plt.title('EER - '+fname)
  plt.savefig(fname)

if __name__ == "__main__":
    import subprocess
    eer_txt = '/home/hexin/Desktop/hexin/datasets/eer_3s.txt'
    score_txt = '/home/hexin/Desktop/hexin/datasets/score_3s.txt'
    trial_txt = '/home/hexin/Desktop/hexin/datasets/trial_3s.txt'
    subprocess.call(f"/home/hexin/Desktop/hexin/kaldi/egs/subtools/computeEER.sh "
                    f"--write-file {eer_txt} {trial_txt} {score_txt}", shell=True)
