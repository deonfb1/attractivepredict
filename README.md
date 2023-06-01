# Facial Attractiveness Predictor
What makes a face attractive? Whatever it is, the answer(s) are extremely nuanced. So after finding a celebrity dataset on facial features, I set out to find an "answer" through an h2o deep learning neural network. https://www.kaggle.com/datasets/jessicali9530/celeba-dataset

## Machine Learning Approach
This project uses h2o, a fast and scalable open-source machine learning platform, to build a facial attractiveness predictor based on a neural network. The neural network is a multi-layer feedforward artificial neural network that learns by adjusting the weights and biases of the connections between the neurons using a gradient descent algorithm. The network is trained on a dataset of celebrity faces with binary labels for different facial features. The network is evaluated using 10-fold cross-validation and various performance metrics such as R-squared and confusion matrix. After training multiple models, it was determined that the model with 3 hidden layers of 5 nodes each was best as the model had the highest R-squared. The network also outputs the variable importance of the features used for prediction.

## Neural Network
### Confusion Matrix
The testing set is composed of:

* Total data points: 40,520
* Attractive faces: 20,767
* Unattractive faces: 19,753

|                    | Predicted Attractive | Predicted Unttractive |
|--------------------|-----------------------|---------------------|
| **Actual Attractive** | 16,313                | 4,454               |
| **Actual Unattractive**   | 4,173                 | 15,580              |

* Sensitivity/Recall: % of (actual) attractive faces correctly predicted.
TP / (TP + FN) = 78.6%
* Specificity: % of (actual) unattractive faces correctly predicted.
TN / (FP + TN) = 78.9%
* Precision: % of (predicted) attractive faces that are attractive.
TP / (TP + FP) = 79.6%

These results show that the neural network achieved a reasonable level of accuracy in predicting facial attractiveness. However, there is still room for improvement, as the model incorrectly identifies faces with false positives and negatives approximately 20% of the time. 

### Variable Importance
In addition to evaluating the model's performance, the neural network also provides information about the importance of the features used for prediction. The variable importance analysis helps understand which facial features have the most significant impact on determining attractiveness. 

<img width="461" alt="Variable Black 1" src="https://github.com/deonfb1/attractivepredict/assets/90005922/9979cbd4-7170-4415-af12-f1f66c9090b0">
<img width="459" alt="Variable Black 2" src="https://github.com/deonfb1/attractivepredict/assets/90005922/40e7a544-02f3-4741-9fad-e8dbb2eda2d4">

H2O neural networks, like other deep learning models, don't have easily interpretable coefficients like traditional linear models. This lack of coefficients in neural networks makes it challenging to directly determine the contribution of individual variables to the prediction of attractiveness or unattractiveness.

__So I made a visualization of prominent features seen in both sets of faces, conducting my own analysis.__

<img width="697" alt="Importance Tableau" src="https://github.com/deonfb1/attractivepredict/assets/90005922/0642ea10-b1da-47c6-810c-92521b33dd34">

By presenting the data in this way, you can reasonably make assumptions on which of the most important features contribute more to which outcome.

**Attractive**: Young, Wearing Lipstick, Heavy Makeup, Smiling, Mouth Slightly Open, Oval Face

**Unattractive**: Male, Big Nose, Blurry, Eyeglasses, Chubby, Double Chin, Gray Hair, Bald

<img width="698" alt="Full Top" src="https://github.com/deonfb1/attractivepredict/assets/90005922/4ce9ab52-9558-46db-ae61-c802de218a80" title="All variables excluding 'No Beard' & 'Male'">
<img width="692" alt="Full Bottom" src="https://github.com/deonfb1/attractivepredict/assets/90005922/a9bb8d10-6ad9-44d1-8a3c-43d05baa8ccd" title="All variables excluding 'No Beard' & 'Male'">

__Key Takeaways__
* **Youthfulness is highly attractive.** 93% of the faces rated as attractive were young, reflecting the human preference for youthful features.
* **Makeup boosts attractiveness ratings significantly.** This is not surprising, as makeup enhances confidence and appeal.  
* **Males are less likely to be rated as attractive.** Is it due to innate differences in beauty? Lower expectations for men’s appearance? Or are there other sociocultural factors at play?
* **The old stereotypes persist.** This data reinforces the notion that negative stereotypes remain, associating unattractiveness with characteristics like age, baldness, weight, or nerdiness.
* **Aesthetic changes don't matter (much).** Factors such as hair color, jewelry, mustache, sideburns, or goatee were the lowest in variable importance.
* **Mouth Slightly Open has high importance yet low disparity.** Only 2% more attractive faces have this feature than unattractive ones. This implies that a slightly open mouth signals either sexiness or disgust to the human brain depending on various factors. Still, this data may suggest that when attracting someone it's best to "get the London look".

I could optimize the neural network, tweak the training model, and experiment with even more nodes and regularization techniques but that's not the goal of this project. **The goal of my projects are to explore and innovate with interesting topics in the field of machine learning.** Attractiveness is subjective. And a project like this introduces the potential for many biases. Such as researcher bias, as the determination of attractiveness relies on subjective judgments. It raises questions about who decided which celebrities were considered attractive or not, and why they've disgraced Danny Devito. Additionally, there may be training set bias, where the collected young faces may be significantly skewed towards attractiveness, or male faces skewed towards unattractiveness.

Another form of bias is variable selection bias. The model indicated that wearing a necklace was the second least important variable. If that variable were to be replaced with something like race, I suspect the importance goes up. And things start to get interesting in both an intriguing and possibly problematic way. Non-physical variables such as fame or height can also have significant effects on ones perceived attractiveness (looking at you, Pete Davidson and Jay Z). One of the more important features left out of this dataset is facial symmetry. And what better way to experiment with what's possible in machine learning than by applying some computer vision techniques?
