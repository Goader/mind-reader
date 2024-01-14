python main.py \
	accelerator=gpu \
	datamodule.timeseries.train=data/train/timeseries \
	datamodule.timeseries.val=data/val/timeseries \
	datamodule.timeseries.test=data/test/timeseries \
	datamodule.design.train=data/train/design \
	datamodule.design.val=data/val/design \
	datamodule.design.test=data/test/design \
	datamodule.batch_size=32 \
	model.num_layers=4 \
	task.max_epochs=30
