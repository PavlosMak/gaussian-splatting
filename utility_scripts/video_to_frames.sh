#!/bin/sh
INPUT_PATH=$1
OUTPUT_PATH=$2
ffmpeg -i $INPUT_PATH -qscale:v 1 -qmin 1 -vf fps=2 "${OUTPUT_PATH}/%04d.jpg"
