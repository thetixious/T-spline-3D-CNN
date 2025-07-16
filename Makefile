all: preprocess train evaluate
T-spline-3D-CNN
preprocess:
	python T-spline-3D-CNN/archive/convert_to_obj.py
	# rhino step (ручной)
	python T-spline-3D-CNN/preprocessing/point_sampler.py

train:
	python T-spline-3D-CNN/train.py --config configs/config.yaml

evaluate:
	python T-spline-3D-CNN/evaluate.py --checkpoint models/best.pth
