This repo containes e2e DEVops in VScode from jupyter notebook to container run REST API service
making predictions of cancer diagnosis based on blood analysis and other patient's data taken here: 
https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset/ 

# How to run and test MLaaS service:
1) first you have to pull this repo and create folders ./model/preproc in the project folder
2) then you need to recreate virtual environment as per files pyproject.toml and uv.lock
3) the Makefile containes all other necessary steps:
4) `make train-model` will train and save ML-model as well as fitted preprocessing objects in the ./model/preproc folder structure as '.bin'-files and '.npy'-files (NB!: not saved to GitHub! Thus the folders for them must be created manually, they are not pulled from GitHub!)
5) `make build-docker-image` shall build docker image based on Dockerfile and .py and .bin files in folders src, pipelines_scripts and models/preproc
6) `make run-docker-container` shall run the built image as a container instance ready to serve ML-predict-requests
7) `make test-mlaas` is to be used for testing the service run in the container, it calls the testing script, that randomly selects two cases from training dataset: one for positive case, another for negative case and sends one after another to REST API, printing the predictions returned by the service.  Each new run you will see predicted different cases selected from training dataset.

# Status of work: FINISHED:
## Migration from model training jupyter to production code run in a docker-container is complete:
1) notebook was split to functions in src-folder
2) functions tests for pytest are added to tests-folder
3) e2e training pipeline is placed in pipelines-folder 
4) stand-alone offline inference using saved model implemented in src-folder, covered by pytest
5) REST API server script is ready and tested by self-written trigger script too
6) Docker image for MLaaS service created and tested 

# Project folder structure (checked by tests too):
* .\data - initial datasets as well as intermediate data matricies - like validation date extracted before training
* .\models (all in .gitignore) - the trained model save
* .\models/preproc (all in .gitignore) - preproc objects fitted on trained data saves: scalers for numerical and encoders for categorical, as well as raw dataset fields split to numerical, categorical and binary groups for preprocessing
* .\config (empty - not in git) - not used but checked by tests for existence
* .\src - code for all functions used by pipelines
* .\tests - pytest code for checking src-code
* .\notebooks - jupyter written DS ideas implemented into production project
* .\pipelines_scripts - model training pipeline and Flask application with REST API server for pre-trained model, as well as self-written simulator of REST API client requesting ML-model predict (requests trigger)