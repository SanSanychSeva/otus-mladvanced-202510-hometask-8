# Status:
## Migration from model training jupyter to a python pipeline is complete:
1) notebook was split to functions in src-folder
2) functions tests for pytest are added to tests-folder
3) e2e training pipeline is placed in pipelines-folder 

## Now working on the REST API publishing of the model inference

# Project folder structure (checked by tests too):
* .\data - initial datasets as well as intermediate data matricies - like validation date extracted before training
* .\models - the trained model save
* .\models/preproc - preproc objects fitted on trained data saves: scalers for numerical and encoders for categorical
* .\config - not used but checked by tests for existance
* .\src - code for all functions used by pipelines
* .\tests - pytest code for checking src-code
* .\notebooks - jupiter written DS ideas implemented into production project