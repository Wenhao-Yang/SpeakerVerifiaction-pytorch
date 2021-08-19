#!/usr/bin/env bash


for dir in $*; do
  if [ ! -f $dir/feats.scp ]; then
    echo "$0: no such file $dir/feats.scp"
    exit 1;
  fi

  mv $dir/feats.scp $dir/feats.scp.back
  sort -k 2 $dir/feats.scp.back > $dir/feats.scp
done
