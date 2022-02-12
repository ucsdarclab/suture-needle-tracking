#!/bin/sh

python -m tracking.run -m=read_files -Kf=data/trial_1/K.csv -nr=11.46e-3 -rpf=data/trial_1/reference_points.csv -pv=100 -pn=5000 -si=1 -if=data/trial_1/images/ -af=data/trial_1/actions.csv -Drf=data/trial_1/DLC_results.csv
