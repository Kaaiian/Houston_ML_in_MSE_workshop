# =============================================================================
#                               Import Libraries
# =============================================================================
# Python has a bunch of open source libraries we will want to use. 
# We will need to specify which ones here. Don't worry too much about 
# understanding each import right now. They will also be explained when they 
# are called in the code.

#---------------general python libraries-------------------------
# Get the in pandas libarary. Allows us to do 'excel' like actions.
# (the "as pd" stores the library on the variable "pd")
import pandas as pd
# numpy allows us to do vector math and has a lot of common math functions.
import numpy as np
# matplotlib.pyplot allows us to quickly and easily plot data to visualize it.
import matplotlib.pyplot as plt

#---------------machine learning python libraries-----------------
# The train_test_split function allows us to easily split our data
from sklearn.model_selection import train_test_split
# We use these classes to setup a grid search to find optimal model parameters
from sklearn.model_selection import KFold, GridSearchCV
# StandardScalar and Normalizer are used to scale the data.
from sklearn.preprocessing import StandardScaler, Normalizer
# We will use a support vector regression (SVR) to predict our properties
from sklearn.svm import SVR
# We can get simple metrics to monitor performance
from sklearn.metrics import r2_score, mean_squared_error

#---------------custom python code-------------------------------------------
# We need to import custom python code that is to complicated for this guide.
import composition
import utils

# =============================================================================
#                               Clean the data
# =============================================================================
# We want to clean the aflow data that we downloaded from aflowlib.org.
# To do this, we will first read in the .csv file using pandas (pd)

# Read in the aflow data as a .csv file. (df stands for 'DataFrame')
df_aflow_elastic = pd.read_csv('aflow_data/aflow_elastic_data.csv')

# We can look individual columns of data using the format df['column_name']
# Here, I will save a variable named 'uncleaned_formulae'
uncleaned_formulae = df_aflow_elastic['ENTRY ']

# -----------------------------------------------------------------------------
# We use a "for loop" to iterate through each formula (value from excel cells).
# While looping, I will appemd 'clean' formulae into the "cleaned" list
cleaned_formulae = []

# start the "for loop"
for cell_value in uncleaned_formulae:
    # cell_value format: "Ag10S8Sb2 [e6cass8dks00as]"
    # desired format: "Ag10S8Sb2"
    # need to remove: " [e6cass8dks00as]"
    # -----------------------------------------------------------
    # We want to read the cell value and only extract the formula.
    # We can do this by removing all characters that come after the " ["
    # To accomplish this, I will use the code: cell_value.split("[")
    # This will split the string at the "[". After I can simple take the first
    # part of the split. (str.split() returns a list, we want the index "0")

    # split string into list
    split_list = cell_value.split(" [")
    # get the first item of the list (indexed at location "0")
    clean_formula = split_list[0]
    # append the 'clean formula' to the list of "cleaned_formulae"
    cleaned_formulae.append(clean_formula)

# Lets now make a new dataframe to hold our clean data.
df_cleaned = pd.DataFrame()
# We can now add a column with the cleaned formulae, df['column name'] = values
df_cleaned['formula'] = cleaned_formulae
# We can also add a column with the target property we want to predict
df_cleaned['bulk_modulus'] = df_aflow_elastic['AEL VRH bulk modulus ']


# =============================================================================
#                              Evaluate the data
# =============================================================================
# We now want to make sure that everthing looks okay with our data.

# ----------------------------------
# Handle duplicate formula instances
# ----------------------------------
# As a first step, we can check for duplicates. Because this data is ordered in
# a well though manner, we can get away looking for matching strings.
# For chemical formula, it is probably better to check for duplicates on a 
# chemical basis (ie. do the fractional compositions match, Al2O3 = Al4O6)

# We can use "Series.value_counts()" on the formula column to see if a string
# value appears more than once.
check_for_duplicates = df_cleaned['formula'].value_counts()

# if we open this up, we see that there are duplicate values. 
# We can remove duplicates by taking a mean. Because we are working fast here,
# we can just drop duplicates after the first instance. (Notice, the "inplace"
# argument allows use to make this change to the dataframe directly.)
df_cleaned.drop_duplicates('formula', keep='first', inplace=True)

# -------------------------------------
# Check the property for anything "odd"
# -------------------------------------
# We can now look at the property to see if there is anything unexpected.
# Some things we might want to look out for: "should there be negative values?"
# "How high does this property get?" "What do I expect the average to be?" etc.

# We can quickly look at the values with a histrogram plot.
plt.figure(1, figsize=(10, 10))
df_cleaned['bulk_modulus'].hist(bins=20, grid=False, edgecolor='black')
plt.plot()

# -----------------------------------------
# Consider tagging distinct chemical groups
# -----------------------------------------
# A lot of materials data is more 'structured' than this. We can often expect
# to find materials of the same type. A more detailed evaluation would probably
# tag those materials, and be cleaver when making the training and test data.
# We are going to ignore this step for simlicity.

# ---------------------------------------------------------------------
# Consider whether you want to use auxilary properties in your learning
# ---------------------------------------------------------------------
# If we are interested in prediction the bulk modulus for any arbitrary formula
# we will want to make sure that our features can be completely derived from
# the chemical composition. If we are interested in a more focused task, such
# as screening the PCD for superhard materials* than we can considered
# using additional info. Descriptions of the strucutre are a good addition.
#      *Brgoch group (https://pubs.acs.org/doi/abs/10.1021/jacs.8b02717)


# =============================================================================
#                             Featurize the data
# =============================================================================
# We can now look at making the chemical formula "machine readable".
# We want to be able to give the compute a vector that describes the formula
# in a meaninful way. 

# The simplest version of this is a vector were each component represents a
# different formula, ie. (Ag, Al, ..., O, ..., Zr). Each formula (take alumina)
# can now be easily encoded.  Al2O3 ==> (0, 2, ..., 3, ..., 0)
# We can usually do better however. Instead of using just the elements, we can
# make a feature from a combination of atomic & elemental properties
# to make a composition-based feature vector (CBFV).

# I have supplied code (composition.py) that does this for us automatically. 
# If we a pandas.DataFrame with a 'formula' and 'target' column, the function 
# will return the features for each instance of data (X), the target values (y)
# and the formula associated with those (formulae).

# Lets rename our columns to match the required input
df_cleaned.columns = ['formula', 'target']

# Lets convert our chemical formula into features here
X, y, formulae = composition.generate_features(df_cleaned)


# =============================================================================
#                           Make a train-test split
# =============================================================================
# Now that we have a "machine_readable" input, we want to partition our data
# into a train and test split. The training set will be used to train and 
# optimize our model. The test set will be reserved till the end to ensure our
# model is capable of accurate predictions beyond the data used for training.

# lets perform this step using sklearn function "train_test_split". 
# We simple give it the fraction of data we want in the test set. We can also
# give it a 'random seed' so we can recreate the same split everytime.
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.33,
                                                    random_state=42)


# =============================================================================
#                          Consider Scaling the data
# =============================================================================
# We can use StandardScalar to scale each feature to a mean=0 and variance=1.
# We then normalize the vector for each instance of data. This is done for
# numerical purposes (allows us to use gradient-descent-based algorithms).

# FOR STANDARD SCALAR:
# "Fit the training data" (calculate the mean & variance of the training data)
# "Transform" (subtract mean & divide by variance from each instance of data)

# FOR NORMALIZER:
# "Fit" (does nothing, but is kept for workflow purposes)
# "Transform" (scales the vector, for each instance of data, to unit norm)

scalar = StandardScaler()
normalizer = Normalizer()

# Do the scaling step
X_train_scaled = scalar.fit_transform(X_train)  # get statistics & transform
X_test_scaled = scalar.transform(X_test)  # trandform using 'training' stats.
# Do the normalizing step
X_train_scaled = normalizer.fit_transform(X_train_scaled)  # normalize vectors
X_test_scaled = normalizer.transform(X_test_scaled)  # normalize vectors

# Algorithms that generally need scaling: Linear, Ridge, & Lasso regressions,
# Support Vector Machines, Neural Networks, Logistic regression

# Algorithms that DO NOT need scaling: Decision Tree, Random Forest,
# Gradient Tree Boosting, etc. (usually decision tree based algorithms)


# =============================================================================
#                          Select desired algorithm
# =============================================================================
# We can consult the literature and see that most non-linear algorithm do a
# good job of predicting materials properties from the composition. Let's
# follow the Brgoch group and use a support vector regression. 

# here we can define the algorithm we want to use to model the data
model = SVR()


# =============================================================================
#                          Optimize Parameters
# =============================================================================
# We are using a SVR to model the data. We can change the parameters we use in
# the modeling to get better or worse models. These parameters generally
# dictate the amount of 'regularization' we apply to the model. Regularization 
# is our metaphorical dial for adjusting model complexity. Using a small amount
# of regularization may lead to a model that is too complex, grossly overfiting
# on the training data. Using too much regularization makes our model "simple"
# incapable of learning anything useful. We want to balance this by searching
# over a large range of possible parameter values. For simplicity we will use 
# a grid search.

# We first start by defining a cross-validation scheme. In this case we can use
# 5-fold cross-validation on shuffled data.
cv = KFold(n_splits=5, shuffle=True, random_state=1)

# We now define the parameter space we want to search over. In the past I have
# found C=10 and gamma=1 to be generally effective values. Center our parameter
# search around these values. Note, it is good practice to have your search
# span several orders of magnitude. We will set our parameters accordingly.
c_parameters = np.logspace(-1, 3, 5)
gamma_parameters = np.logspace(-2, 2, 5)

# save the search space as a dictionary
parameter_candidates = {'C': c_parameters,
                        'gamma': gamma_parameters}

# with this line, we will define the full grid search.
grid = GridSearchCV(estimator=model,
                    param_grid=parameter_candidates,
                    cv=cv)

# Here we start running the grid search. 
# The grid object will contain the optimal parameters when our search is done.
grid.fit(X_train_scaled, y_train)

# Here we save the best parameters
best_parameters = grid.best_params_
print(best_parameters)

# Here we visualize model performance for different model parameters
utils.plot_2d_grid_search(grid, midpoint=0.7, vmin=-0, vmax=1)
plt.plot()

# =============================================================================
#                     Check performance on the test set
# =============================================================================
# We can now use these optimal parameters to fit a model to all the training
# data. This model can then be applied to the test data to see how well we did.

# make our final model using the "best_parameters" dictionary as our arguments.
final_model = SVR(**best_parameters)

# fit the model to the training data
final_model.fit(X_train_scaled, y_train)

# predict on the test data
y_test_predicted = final_model.predict(X_test_scaled)

# plot the results to see how we did
utils.plot_act_vs_pred(y_test, y_test_predicted)
score = r2_score(y_test, y_test_predicted)
rmse = np.sqrt(mean_squared_error(y_test, y_test_predicted))

print('r2 score: {:0.3f}, rmse: {:0.2f}'.format(score, rmse))


# =============================================================================
#                     Make predictions on new compounds
# =============================================================================
# If we want to make predictions for a new compounds, we will first need to
# vectorize the composition. We will then need to scale it by the training data
# and finally, we will need to make predictions from the trained model. 

# Lets define a simple class to do that here (dont worry about following along)

class MaterialsModel():
    def __init__(self, trained_model, scalar, normalizer):
        self.model = trained_model
        self.scalar = scalar
        self.normalizer = normalizer

    def predict(self, formula):
        '''
        Parameters
        ----------
        formula: str or list of strings
            input chemical formula or list of formulae you want predictions for
    
        Return
        ----------
        prediction: pd.DataFrame()
            predicted values generated from the given data
        '''
        # Store our formula in a dataframe. Give dummy 'taget value'.
        # (we will use composition.generate_features() to get the features)
        if type(formula) is str:
            df_formula = pd.DataFrame()
            df_formula['formula'] = [formula]
            df_formula['target'] = [0]
        if type(formula) is list:
            df_formula = pd.DataFrame()
            df_formula['formula'] = formula
            df_formula['target'] = np.zeros(len(formula))
        # here we get the features associated with the formula
        X, y, formula = composition.generate_features(df_formula)
        # here we scale the data (acording to the training set statistics)
        X_scaled = self.scalar.transform(X)
        X_scaled = self.normalizer.transform(X_scaled)
        y_predicted = self.model.predict(X_scaled)
        # save our predictions to a dataframe
        prediction = pd.DataFrame(formula)
        prediction['predicted value'] = y_predicted
        return prediction

# initialize an object to hold our bulk modulus model
bulk_modulus_model = MaterialsModel(final_model, scalar, normalizer)

# lets define some formulae we are interested in
formulae_to_predict = ['NaCl', 'Pu2O4', 'NaNO3']
formula = 'NaCl'

# use the bulk modulus object to generate predictions for our formulae!
bulk_modulus_prediction = bulk_modulus_model.predict(formula)

# Email me with additional questions!
# kaaikauwe@gmail.com