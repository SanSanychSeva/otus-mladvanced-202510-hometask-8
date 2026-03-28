train-model:
	python pipelines_scripts/model_training_pipeline.py

run-local-api:
	flask run

test-local-api:
	python pipelines_scripts/trigger_rest_api.py