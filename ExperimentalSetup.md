# Terminology

*_DDT (Down-down time)_*: This is similar to what PILOT described as the inter-keystroke time. It is the time interval between the key-presses of the two keys in a digraph 

*_Penalty Score_*: The penalty score indicates the possibility of the password given the output of the classifier, i.e, the higher the penalty score the less probable the password.  

# Experimental Setup
Our experimental setup was similar to the PILOT in many aspects. 

## Training the classifier
Similar to the PILOT, we used a Random Forest classifier to train our model. For training the model, we used [MSU Dataset] (“http://www.cse.msu.edu/~liuxm/typing”), GREYC, and GREYC Web Dataset. To get excellent results for our attack, we chose to use datasets that were primarily utilized for keystroke dynamics. 

Similar to the preprocessing elements introduced in the PILOT paper, we under-sampled each digraph in the dataset that appeared more than 1,000 times to 1,000 randomly selected instances. Also, we eliminated digraphs that had less than 100 instances. Further, we carried out several other operations to ensure that the data we were working with was as balanced as possible.

## Password Prediction
The trained RF generated a list of N digraphs based on a list of DDT. Using the generated list, we ranked the passwords in the RockYou dataset in accordance to the penalty score metric. 

## Ranking Metrics and Threshold

To rank the passwords, we primarily relied on the RockYou dataset. We restricted our domain to 8-character alphanumeric passwords. Given a password in the RockYou dataset, we used the trained model to generate a penalty score. We repeated this process for each password in the RockYou dataset. The number of attempts required for the model to predict a given password is the position of the password in the list of passwords ranked in the non-decreasing order of penalty scores.

The idea of this attack was to confirm that by adding a threshold we were able to distort the passwords that were predicted by the RF. We repeated the same procedure as described above to compare the number of guesses required to predict a given password with/without the threshold. 

# Differences between PILOT and X
In the PILOT, the researchers used three public datasets [MSU, Stony Brook, Clarkson] to train two different classifiers — Random Forest (RF),  Neural Network(NN) — that predicted digraphs based on a given inter-key time. After analyzing the results in the paper we learned that the NN performed poorly as compared to the RF classifier which is why we decided to limit the scope of our attack to just use the RF model.

We also used different datasets as mentioned in the _Training the Classifier_ section. In addition, we used only the password _lamondre_ from the PILOT for the replication of this experiment. 