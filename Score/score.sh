#!/usr/bin/env bash

stage=0
if [ $stage -le 0 ]; then
  trials=$voxceleb1_trials
  scores=exp/scores_voxceleb1_test
  eer=$(compute-eer <(Vector_Score/prepare_for_eer.py $trials $scores) 2>/dev/null)
  mindcf1=$(Vector_Score/compute_min_dcf.py --p-target 0.01 $scores $trials 2>/dev/null)
  mindcf2=$(Vector_Score/compute_min_dcf.py --p-target 0.001 $scores $trials 2>/dev/null)
  echo "EER: $eer%"
  echo "minDCF(p-target=0.01): $mindcf1"
  echo "minDCF(p-target=0.001): $mindcf2"
fi
