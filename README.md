# Status:
## Migration from model training jupyter to a python pipeline is complete:
1) notebook was split to functions in src-folder
2) functions tests for pytest are added to tests-folder
3) e2e training pipeline is placed in pipelines-folder 
4) stand-alone offline inference using saved model implemented in src-folder, covered by pytest
5) REST API script is ready too

## Now working on the dockering REST API publishing of the model inference

# Project folder structure (checked by tests too):
* .\data - initial datasets as well as intermediate data matricies - like validation date extracted before training
* .\models - the trained model save
* .\models/preproc - preproc objects fitted on trained data saves: scalers for numerical and encoders for categorical, as well as raw dataset fields split to numerical, categorical and binary groups for preprocessing
* .\config - not used but checked by tests for existence
* .\src - code for all functions used by pipelines
* .\tests - pytest code for checking src-code
* .\notebooks - jupyter written DS ideas implemented into production project