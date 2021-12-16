# AML Final Project

> Jace Yang, Kevia Qu, Sarosh Sopariwalla, Yu-Chieh Chen, Yunzhe Zhang

Explanations of the folders:

- the `code` contains the entire workflow of this project ordered by the prefix of the file names.

    - The `source.py` contains the commonly used functions / packages imported across all notebooks

- the `input` folder includes the data downloaded from https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/data, which are too big to upload to GitHub.

- the `processing` folder contains the temporary files like the cleaned data that will be accessed to each model files, as well as the trained models themselves.

    - Some of the `.feather` files genreated from `0_Input Data.ipynb` and `1_Feature_Engineering.ipynb` are also over 1 GB and not uploaded to GitHub.

- the `output` folder saves the graphs and the result of the [project EDA slides](https://docs.google.com/presentation/d/1EzBHjcaslv-fPCEv-8x-c_Oe6x9LoPT1apVnat1MatY/edit?usp=sharing) and th report.