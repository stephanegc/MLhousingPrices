# MLhousingPrices

This repository contains a package to analyse the California Housing Prices data set.  
It contains the code (**MLhousingPrices**) and notebooks (**MLhousingPrices/resources**) to run the code interactively.   
   
To run the code in the notebooks, please clone this repository and install the packages listed in the **requirements.txt** file, via :   
**pip install -r ./requirements.txt** (within a terminal with this gitrepo a the working directory, containing a virtual environment)
   
The package itself can be installed via :    
**pip install ./dist/mlhousingprices-3.0.0.tar.gz**  (within a terminal with this gitrepo a the working directory, containing a virtual environment)

## Code
* **MLhousingPrices/preprocessor.py** contains the code to preprocess the data.
* **MLhousingPrices/model.py** contains the code to train models.
* **MLhousingPrices/app.py** contains the code to locally run the FastAPI app.

## Notebooks
* **MLhousingPrices/resources/exploratory_analysis.ipynb** is the notebook containing the data exploration and the explanation of the feature engineering/selection and model training. 
The model that was eventually selected is used in the FastAPI (predict endpoint), and is stored under **MLhousingPrices/resources/xgb_tuned.pkl**.
* **MLhousingPrices/resources/demo.ipynb** is the notebook containing the explanation of how to query the FastAPI endpoints, and how to use the classes in the package to predict outputs and train models (what is used in the FastAPI endpoints).
