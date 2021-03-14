# Neural_Network_Charity_Analysis
### Overview
Using machine learning models and neural networks we will evaluate 34,000 organizations that have received funding from Alphabet Soup.  We will perform analysis to determine which organizations are more likely to  use fund effectively.

For this project we used data analysis tools language Python and Pandas, data preprocessing tools from SKlearn (StandardScaler and OneHotEncoder) and Neural Network models from TensorFlow.

### Results
When preparing our categorical data for encoding we visualized the density of the application type values. Here we can identify that '400' is roughly the cut-off value below which we can consolidate values as 'other': 

![image1.png](/Resources/image1.png)

We selected a deep neural net for our model.

We selected the field "Is_Successful" to be the target or dependent variable (which means it as to be dropped from the X dataset). Our feature variables were "STATUS", "ASK_AMT" and "APPLICATION_TYPE..." encoded fields. Some columns were not used in the analysis such as EIN and NAME.

We trained the model storing checkpoint weights in the Resources/AlphabetSoupCharity.hdf5 file.

The initial loss and accuracy scores are below:

![image3.png](/Resources/image3.png)

### Summary
We made three attempts to optimze the model:

### First Optimization
First we changed the threshold for categorizing ATTRIBUTE_TYPE from 400 to 600:

![image2.png](/Resources/image2.png)

Results were are small change in accuracy of four thousanths of a point to 0.729:

![image4.PNG](/Resources/image4.PNG)

### Second Optimization
The second attempt was to change the number of hidden neurons:

![image5.PNG](/Resources/image5.PNG)

Results from the second optimization have increased accuracy to 0.730!

![image6.PNG](/Resources/image6.PNG)

### Third Optimization
The third optimization is removing the STATUS column from the feature set.
This had no change to the accuracy of the model.

![image7.PNG](/Resources/image7.PNG)

### Next Steps
Further analysis could be performed with the continued optimization of the dataframe columns. We could perform a correlation analysis to identify other columns that could be removed from the feature set to improve loss and accuracy metrics.
