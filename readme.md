
<font size="4">
    
# A practical guide to machine learning materials (with python)

This guide is supposed to act as a general outline for simple machine learning. I have included a 
supplimentary python file (**SPF**), found [here](https://github.com/kaaiian/Houston_ML_in_MSE_workshop/blob/master/machine_learn_elastic_properties.py), to get you started with the coding aspects. 

This workflow should be sufficient to build simple models to predict materials properties based on the composition. 

If you are completely new to coding, you can follow [this link](https://www.youtube.com/watch?v=5mDYijMfSzs) (a good video I found on youtube) to get a brief walkthrough on how to download python using the anaconda distribution on a windows computer. [This link](https://www.youtube.com/watch?v=WhBg-mC0ChQ&t) goes through the supplimentary python file (**SPF**) line-by-line and explains in detail how the code is being used and why. 

-----

## Find some materials data

In order to do some machine learning, we need materials data. 

Some easy places to get data include: 

- [Citrination](https://citrination.com). You will need to make an account. Probably want to use API's to get a useful amount of data.

- [Materials Project](https://materialsproject.org/). Same deal. You will need to make an account. Again, you will probably want to use API's to get a useful amount of data. <br>

- [Aflow](http://aflowlib.org/). Just need to search for the property you want. Data is retruned in html table. This can easily be copied into excel.

- [Matminer](https://hackingmaterials.github.io/matminer/). "Matminer is a Python library for data mining the properties of materials". A small bit of python will give you quick access to Citrination and Materials Project data as well as select data  from individual publications.

- [Literature](https://scholar.google.com/).

For this guide, we can use data obtained from the [Aflow](http://aflowlib.org/). This [link](https://youtu.be/ptLzE0o6sGs) showns me using aflow to get elastic data and saving it to a CSV file.

(*If you follow the video, don't just grab the first thousand. You will likely want to return 3000 compounds to make sure you get everything! There are a total of 2382 ternary-compounds with elastic data*)

## Clean the data

Once we have data, we want to make sure that the data is clean and usable. This means we need to know how we want to use the data. 

In our case, we will be using the composition to try to predict some materials properties. 

For this reason, we will want to make sure our data is formated to clearly have a composition and its associated properties. The following table gives an example of what clean data could look like. Using elastic properties as our target for learning, we might get something like this.

| formula | Bulk Modulus (GPa) |
|:---|:---|
| Ag10S8Sb2 | 35.2 |
| Ag1Al1S2 | 71.6 |
|... | ... |
| Sn6V6Y1 | 112.3 |

(*If you grabbed the aflow data following the youtube link. You will notice the 'ENTRY ' column has the formula as well as the unique aflow id. in square-brackets. I remove this using lines 41-77 in the **SPF**.*)

## Evaluate the data for issues.

---
- Are there duplicates? 
- What is the distribution? 
- Are there distinct chemical groups? 
- Are there useful auxilary properties?
---

### Duplicates?

One we have the data, we will want to make sure there are no problems with the data. A first thing to check is the existance of duplicate instances of data. For us, that means looking for duplicate compositions in our dataset, we will want to devise a plan to address or remove duplicates. Note, sometimes duplicate formulae can have very different properties if they are in different environments (E.g. high vs. low temp/pressure, Different crystal/micro structure, thin-film vs. bulk, etc.).

Common ways to handle duplicate instances of data:
- Drop all but 1 at random
    - a simple approach that is useful if values are not too dissimilar
- Take the mean value
    - the same idea, but feels a bit better
- Take the median value
    - proably a better approach, but can be ill-defined for even numbers
- Distringuish between duplicates
    - add additional features that encodes extra information such as structure, pressure, temp, etc.

### Distribution?

After taking care of duplicate values, we can look at the distribution of the data to make sure that there is no funny business going on. However, before we look at the distribution it is often useful to prepare our "engineering senses". 

Some questions to ask before looking at the data:
- What do I expect this distribution to look like?
- Do I expect there to be negative values?
- What materials do I expect to be at the extremes?
- What would unreasonable values look like?
- Can I expect to see extreme outliers?
- Does it make more sense to predict the log(property) (think electrical conductivity)

Here are my (non-expert) oppinions on bulk modulus
- What do I expect this distribution to look like?
    - Log normal-ish is pretty common
- Do I expect there to be negative values?
    - That doesn't make sense to me given the definition of bulk modulus
- What materials do I expect to be at the extremes?
    - Maybe formulae with boron, nitorgen or carbon
- What would unreasonable values look like?
    - Diamond is really hard and has a bulk modulus of ~550 (googled that. lol). Values in that range are probably worth investigating.
- Can I expect to see extreme outliers
    - My gut feeling: "DFT is less common for mundane compositions. So we should probably see couple." (might be very wrong)

### Distinct Chemical Groups?

If we are being hardcore, we can try to partition our data into distinct chemical groups that exists (for a good paper on why and how to do this, click [here](https://pubs.rsc.org/en/content/articlelanding/2018/me/c8me00012c)). We will skip this, but say it might be something to try.

### Auxilary Properties?

This is related to working with duplicates. Often times there is more to our data than just "composition". We should stop here and consider: 
1. Do we have properties besides just the composition? and
2. What tool we want to make and how do we plan to use it? 

If we want to make a model that can predict our property given nothing but composition.... than our machine learning algorithm needs to only be trained on composition.

If we want to identify possible [superhard materials](https://pubs.acs.org/doi/abs/10.1021/jacs.8b02717) (shout out to the Brgoch group) from the pearson crystal database, than we can include materials properties associated with the structure.

(*If you want to add auxilary properties, these should be added to the feature vector befor any scaling steps.*)

## Featurize the data (machine readable representation of formula)

Once we really know what we are dealing with, we need to get our data into a machine learnable format. That is to say, machine learning algorithms require a vector representation of the data (making this vector is the 'featurization' step). But how do we vectorize materials data? There are a lot of ways, [matminer](https://hackingmaterials.github.io/matminer/featurizer_summary.html) has a table of techinques and is a place to start

For now, lets look at our data and consier what we are working with.

| formula | Bulk Modulus (GPa) |
|:---|:---|
| Ag10S8Sb2 | 35.2 |
| Ag1Al1S2 | 71.6 |
|... | ... |
| Sn6V6Y1 | 112.3 |

It looks like we want to correlate a chemical formula to bulk modulus. We want to create a vector that can represent any composition. Perhaps the simplest way to do this is to make a vector where each component represents an element ([cool paper here](https://www.nature.com/articles/s41598-018-35934-y)).

That vector would look something like this:

$v_{formula} = $ (Ag, Al, B, ..., O, ..., Zr)

Then, if we wanted to conver Al$_2$O$_3$, it would be as simple as putting a "2" in the aluminum dimension, a "3" in the oxygen dimension, with "0" everywhere else.

$v_{Al_2O_3} = $ (0, 2, 0, ..., 3, ..., 0)

This is a simple approach that illustrates the idea. More elegantly, we can do featurization using a composition-based feature vector ([read more here](https://www.nature.com/articles/npjcompumats201628)).
 

## Split the data into training set (for building the model) and test set (for ensuring out-of-sample performance)

If are going to model the data, it is best practice to split our data into a training and test set. We will use the training set to optimize and fit the model, and the test set as a final indicator of model performance. Because most machine learning models have a number of tunable parameters, we will want to split the training set into train & validation parts. This can be done in a way similar to the train test split. Or it can be done using a cross-validation scheme. The following illustrates the a typical **train-test split followed by 3-fold cross-validation**:

![train-test-validation](https://upload.wikimedia.org/wikipedia/commons/thumb/1/12/Train-Test-Validation.png/640px-Train-Test-Validation.png)

## Consider scaling the features

Once we have our features. We will probably want to scale them. 

Most machine learning algorithms rely on gradient descent to find the optimal model fit. 

If features have drastically different values, issues can arrise with during the search for our optimum. 

**Inser image of scaled vs non-scaled for gradient boosting

Algorithms that will most likely require gradient descent:
- Linear Regression, 
- Ridge Regression, 
- Lasso,
- Support Vector Machines, 
- Neural Networks, 
- Logistic regression, etc.

Algorithms that do not require gradient descent (usually decision tree based):
- Decision Tree, 
- Random Forest,
- Gradient Tree Boosting, etc.

## Select desired algorithms

At this point, we can decide on the algorithm we want to use.

Here are some general characteristics for a few popular algorithms
- random forest (rf): 
    - usually works
    - doesn't need any feature scaling
    - little effort to get good results
    - scales pretty good with data
    - can learn complex non-linear relationships

- support vector machine (svm): (regression=(SVR), classification=(SVC))
    - requires detailed parameter tuning
    - models are slow to train with large amounts of data ( $\approx 10000$)
    - resulting models are generally very good
    - can learn complex non-linear relationships
    
- linear regression (lr):
    - a simple approach, has some interpretability
    - regularization can be added if overfitting
        - L1 regularization = Lasso
        - L2 regularization = Ridge Regression
    - extermely fast. Works with huge amounts of data
    - very limited model complexity (often underfits)

- neural network (nn):
    - more difficult to implement (reasonably sized models require additional software)
    - requires detailed parameter tuning
    - scales well with large amounts of data
    - networks can be designed to learn arbitrarily complex relationships
        - particularly useful for image data
    - usually considered the most accurate

A safe bet is to try many or all of them! Lets go simple and use a support vector regression (SVR).

## Optimize algorithm parameters

Regardless of the algorithm we choose, we will want to optimize the model parameters to get the best performance possible. Trying a bunch of different parameter combinations is a simple way to do this. We can do just that using a 'grid search'. Once a gride search is complete, we can judge whether our best parameters seem optimal. If so, we can make a final model and check our performance test set. If not, we can repeat the grid search changing the range of parameters we look at. 

(*When using ridge, lasso, svm, or nn models, it is a general rule of thumb to search logarithmic space for most parameters.*)

## Check performance on the test set

Once we are satisfied with our performance on the training set, we can move to the test set. The test set should have similar, or slightly worse, performance to the training data. This is the final step and should be done after all optimization has taken place and we are sure about our model and parameter choices.

To check our performance, we will take the optimal parameters from the grid search and train a new model with all of the training data. Once a model is trained with all the data, we can use it to predict on the test data (Do not forget, if you scaled the training data, you need to also scale the test data in the same way)! Once we have predictions on the test data we can calculate an R$^2$ score, and RMSE and we can also plot the results. If your model is succesful, we can take it into the "real world" and get started predicting!

## Make predictions on future compounds

Once we have a fully trained and validated model, we of course want to use it. Here are some steps to make sure a new formula is ready to be put through our model to generate a prediction.

New prediction checklist:
1. Convert the composition/formula into a feature vector
2. If you added auxilary information, add that to the feature vector
    - the meaning of each vector component must match the vectors from the training set)
3. Use scaling (calulated from the training) set on the new feature vector
4. Input prepared vector into the model to generate prediction!


</font>