train-model:
	python pipelines_scripts/model_training_pipeline.py

run-mlaas:
	flask run

test-mlaas:
	python pipelines_scripts/trigger_rest_api.py

build-docker-image:
	docker build -t cancer-prediction-mlaas-prod -f Dockerfile .

run-docker-container:
	docker run --rm -p 5000:5000 --name docker-with-cancer-prediction-mlaas cancer-prediction-mlaas-prod

push-image-to-dockerhub:
	docker tag cancer-prediction-mlaas-prod sansanychseva/ml-advanced-202510-ht8-docker-image
	docker push sansanychseva/ml-advanced-202510-ht8-docker-image