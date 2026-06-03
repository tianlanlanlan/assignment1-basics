#!/bin/bash
set -ex

readonly hf_url_prefix=https://hf-mirror.com

wget $hf_url_prefix/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget $hf_url_prefix/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
wget $hf_url_prefix/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
wget $hf_url_prefix/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz

gunzip owt_train.txt.gz
gunzip owt_valid.txt.gz
