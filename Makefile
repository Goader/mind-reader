S3_BUCKET = "s3://natural-scenes-dataset"

DATA_DIR = "data"
TIMESERIES_DIR = "$(DATA_DIR)/timeseries"
ROI_DIR = "$(DATA_DIR)/roi"
DESIGN_DIR = "$(DATA_DIR)/design"

ROI_TYPE = "HCP_MMP1"
SUBJECTS = 8


# ========== downloading data ==========

timeseries:
	for i in $(shell seq 1 $(SUBJECTS)); do \
		mkdir -p "$(ROI_DIR)/subj0$(i)/"; \
		aws s3 cp "$(S3_BUCKET)/nsddata_timeseries/ppdata/subj0$(i)/func1mm/timeseries/" "$(TIMESERIES_DIR)/subj0$(i)/" \
			--recursive --exclude "*" --include "timeseries_session*_run*.nii.gz"; \
	done

roi:
	for i in $(shell seq 1 $(SUBJECTS)); do \
		mkdir -p $(ROI_DIR)/subj0$(i)/; \
		aws s3 cp $(S3_BUCKET)/nsddata/ppdata/subj0$(i)/func1mm/roi/$(ROI_TYPE).nii.gz $(ROI_DIR)/subj0$(i)/; \
	done

design:
	for i in $(shell seq 1 $(SUBJECTS)); do \
		mkdir -p $(DESIGN_DIR)/subj0$(i)/; \
		aws s3 cp $(S3_BUCKET)/nsddata_timeseries/ppdata/subj0$(i)/func1mm/design/ $(DESIGN_DIR)/subj0$(i)/ \
			--recursive --exclude "*" --include "design_session*_run*.txt"; \
	done

stimuli:
	aws s3 cp $(S3_BUCKET)/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5 $(DATA_DIR)/

downloaded_data: timeseries roi design stimuli

# ========== preparing data ==========

parcelated_timeseries:
	python3 scripts/generate_parcelated_timeseries.py ...

clip_embeddings:
	python3 scripts/generate_clip_embeddings.py \
		--images $(DATA_DIR)/nsd_stimuli.hdf5 \
		--output $(DATA_DIR)/clip_embeddings.pt \
		--batch_size 64 \
		--device cuda

prepared_data: parcelated_timeseries clip_embeddings

# ========== data ==========

data: downloaded_data prepared_data

# ========== cleaning ==========

clean_downloaded_data:
	rm -rf $TIMESERIES_DIR; \
	rm -rf $ROI_DIR; \
	rm -rf $DATA_DIR/nsd_stimuli.hdf5

clean_prepared_data:
	rm -rf $DESIGN_DIR;
	# todo add clip embeddings and parcelated timeseries

clean: clean_downloaded_data clean_prepared_data
