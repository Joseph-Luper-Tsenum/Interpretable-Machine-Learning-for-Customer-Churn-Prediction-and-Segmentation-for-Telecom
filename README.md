### Interpretable Machine Learning for Customer Churn Prediction and Segmentation for Telecom


#### Introduction/Background 
 
Customer churn is the loss of customers or clients. It refers to the phenomenon where customers cease their relationship with a business or service.  Telephone service companies, Internet service providers and many other companies use customer attrition analysis and rates as their key business metrics. Cost of retaining an existing customer is less expensive than acquiring a new one. Companies have customer service branches as an attempt to stop customers from churning. With an emphasis on effectively categorizing customers in the telecom sector, our research is dedicated to accurately forecasting customer attrition by leveraging interpretable machine learning techniques. Proactive Decision-Making: Churn prediction allows businesses to take proactive measures, such as targeted marketing or personalized incentives to retain customers at risk of churning. 

Customer segmentation enables businesses to tailor marketing messages and promotions to specific customer groups, improving the effectiveness of campaigns. Both churn prediction and segmentation contribute to optimizing resource allocation by focusing efforts and resources on high-value customers and potential churners. Our project is committed to precisely predicting customer attrition through the utilization of interpretable ML approaches, with a focused effort on efficiently classifying customers in the telecommunications industry. 

Churn prediction involves using machine learning models to forecast which customers are likely to churn in the future. Customer segmentation is the process of categorizing a customer base into distinct groups based on shared characteristics or behavior. This allows businesses to tailor their strategies to different customer segments more effectively.


#### Problem Definition

It is difficult for telecom company to retain customers. Customers can easily switch providers, driven by rapid technological change and intense market competition. No robust and accurate model exists that assist companies in identifying factors that contributes to customer churning. Customer churn prediction operates as a supervised learning task. Customer segmentation functions as an unsupervised clustering task. Past research has explored a variety of feature selection techniques, including Principal Component Analysis (PCA), information gain, and Linear Discriminant Analysis (LDA) [1]. Numerous studies have engaged machine learning algorithms, such as Decision Trees, Random Forests [2], and Support Vector Machines (SVM), with the objective of effectively predicting customer churn [3].

Traditional machine learning models, such as black-box models like deep learning or ensemble methods, may provide accurate predictions but lack interpretability. This means that it is difficult to understand and explain why a particular prediction or segmentation decision was made. SHAP (Shapley Additive Explanations) machine learning, on the other hand, provides explanations for the predictions or segmentation outcomes. It assigns importance shapley values to each feature, indicating their contribution towards the final prediction or segmentation decision. 
![1problem](https://github.gatech.edu/storage/user/56953/files/ca69afc9-8890-48ab-923c-a92c43604be1)


#### Data Collection and Cleaning

The project employs the Cell2Cell dataset [4] from Kaggle, which underwent preprocessing to ensure a balanced analytical approach. Originally comprising 71,047 entries with 58 attributes each, the dataset received further refinement and cleaning before use. The initial phase of analysis and processing involved purging entries with incomplete data and excluding the 'service area' attribute. The remaining data was then methodically categorized，labeled and normalized in preparation for the subsequent stages of analysis. For the 58 features, 31 of them are real numbers and others are composed of characters and booleans. Real numbers have been converted through the following equation:

<img src="https://latex.codecogs.com/svg.image?\hat{X}_{ij}=\frac{X_{ij}-Mean(X_j)}{Var(X_j)}"/>

where <img src="https://latex.codecogs.com/svg.image?X_{ij}"/> is raw data from <img src="https://latex.codecogs.com/svg.image?i"/> row and <img src="https://latex.codecogs.com/svg.image?j"/> column,  <img src="https://latex.codecogs.com/svg.image?Mean(X_j)"/> is the mean value of the <img src="https://latex.codecogs.com/svg.image?j"/> column, <img src="https://latex.codecogs.com/svg.image?Var(X_j)"/> is the variance of j column and <img src="https://latex.codecogs.com/svg.image?\hat{X}_{ij}"/> is the normalized data from <img src="https://latex.codecogs.com/svg.image?i"/> row and <img src="https://latex.codecogs.com/svg.image?j"/> column. Regarding the categorical and boolean data types, we utilized one-hot encoding to transform them into the binary array format, effectively enabling their employment in our analysis as categorical variables.

#### Data Visualization

We performed exploratory data analysis on this dataset in order to observe some of its characteristics and trends. 
Below is a histogram plot of the monthly revenue:
![Untitled](https://github.gatech.edu/storage/user/76361/files/5f9a641e-aac9-4875-8401-76cc11e78759)

We also created a correlation matrix of all the features of our dataset:
![Untitled](https://github.gatech.edu/storage/user/76361/files/502da8b5-d304-46dd-b72f-5b264f2283f9)

To simplify the process of understanding, below is a table of the top 10 most highly correlated features along with their correlation coefficients:

![image](https://github.gatech.edu/storage/user/76361/files/965d4854-5394-42ff-a53e-ebd355065b49)

We can also see how monthly minutes vs monthly revenue affects churn by creating a scatterplot:
![Untitled](https://github.gatech.edu/storage/user/76361/files/eb4d5819-8661-4e7d-a1b2-dd95cbf12266)


#### Methods 

##### Classification 

For the 58 features, we applied PCA algorithm to eliminate some unnecessary dimensions and to select features. With the principle features and their weights, the topmost principal features that counts for more than 85% in sum were selected. With fewer and more important features, Random Forest model was applied in the classification problem, which is a popular and robust supervised learning approach. Alternatively, we also tried logistic regression and SVM models to compare the accuracy of our models.

##### SHAP (SHapley Additive exPlanations)

SHAP values are based on cooperative game theory, specifically Shapley values. SHAP values provide a way to fairly distribute the contribution of each feature to a model's output. Positive SHAP values contribute positively to the prediction, while negative values contribute negatively. SHAP values offer both global and local explanations to enhance our understanding of model predictions. While global explanations provide an overall understanding of how different features contribute to the model's predictions across the entire dataset, local explanations offer insights into why a specific prediction was made for an individual instance or a small subset of data.

![shap](https://github.gatech.edu/storage/user/56953/files/8d3622cd-48fa-4566-832d-02e6b66ca3ad)



#### Results and Discussion

#### Dimentionality Reduction using Principal Component Analysis

Before the encoding of categorical features, our dataset comprised 58 features. Following the encoding process, the total number of features increased to 73. For the purpose of supervised learning modeling, a subset of features accounting for more than 85% of the total sum was selected, resulting in a final set of 20 features. We fed the reduced principal components (20 features) into our Random Forest classifier, logistic regression and SVM models. We divided our task into PCA form data and normal data. We used 30% samples for testing and 70% for training.

#### Random Forest Classification

In our analysis, considering the binary nature of the output (churn or not), we employed the Random Forest classifier on both the original set of features (73 features, excluding PCA) and the reduced set of features obtained through Principal Component Analysis (PCA), consisting of 20 features. Notably, the model utilizing PCA features demonstrated superior performance compared to its counterpart without PCA feature reduction. This outcome emphasizes the efficacy of feature reduction through PCA and underscores the advantageous performance of Random Forest when applied to a reduced number of features. Additionally, it is essential to highlight that this observation has meaningful implications for optimizing the model's computational efficiency and interpretability.

In a further exploration of the intersection between Principal Component Analysis (PCA) and the Random Forest classifier, we delved into the nuanced details of our analysis. The combined utilization of PCA, which condensed the feature space to 20 principal components, and the Random Forest algorithm yielded noteworthy insights. Our findings indicate that the Random Forest model, when applied to the reduced set of PCA features, not only exhibited enhanced predictive performance but also showcased a heightened efficiency in computational processing. The synergy between PCA and Random Forest, specifically in the context of feature reduction, contributed to a more streamlined and interpretable model.

Moreover, the model's capacity to discern meaningful patterns and relationships within the data was evidently augmented by leveraging the reduced feature set derived from PCA. This not only reinforces the importance of employing dimensionality reduction techniques but also emphasizes their potential impact on the overall efficacy of machine learning models. This intersection underscores the practical significance of feature reduction strategies in machine learning, offering a compelling avenue for further research and refinement in predictive modeling.

Random Forest Classification includes the following steps:
1. Ensemble of Decision Trees: This model has sets of decision trees. They are trained independently.
2. Random Subsampling of Features: During the construction of each tree, a random subset of features is considered at each split. This decorrelates the trees and makes the ensemble more robust.
3. Voting or Averaging: The predictions of individual trees are combined through a voting mechanism, where the class that receives the most votes becomes the final prediction. For regression tasks, the predictions are averaged.

As in any classification task, evaluating model performance goes beyond mere accuracy metrics. A thorough investigation into precision, recall, and the Receiver Operating Characteristic (ROC) curve offers a nuanced comprehension of the Random Forest model's capabilities, highlighting both strengths and potential areas for improvement. 


##### Random Forest Confusion Matrix Without PCA Features (original features)
![image](https://github.gatech.edu/storage/user/56953/files/00cd3cb0-5e77-4e33-acd7-0d43f01910a3)

##### Random Forest ROC and Precision-Recall Without PCA Features (original features)
![Untitled](https://github.gatech.edu/storage/user/76361/files/0e5d1174-fa87-42cb-9614-cf978381dd33)

##### Random Forest Confusion Matrix With PCA Features (feature reduction)
![image](https://github.gatech.edu/storage/user/56953/files/e7de75ea-0f45-475e-ac4f-6615f94fe384)

##### Random Forest  ROC and Precision-Recall With PCA Features (feature reduction)
![Untitled](https://github.gatech.edu/storage/user/76361/files/cf816d83-4037-4cc2-9aee-2453820e4b5b)


#### Logistic Regression

In the context of binary classification for our labeled data (churn or not), we applied the Logistic Regression classifier to assess its performance using our original features (73 features without PCA) and PCA features (20 features). Upon comparison, our Logistic Regression model leveraging PCA features demonstrated competitive performance compared to the model without PCA feature reduction. This outcome emphasizes the utility of feature reduction through PCA and its positive impact on Logistic Regression's ability to work effectively with a reduced set of features.

The Logistic Regression model, being a linear classifier, benefits from reduced dimensionality achieved through PCA. It is notable that Logistic Regression is inherently sensitive to the input feature space, and the application of PCA aids in capturing the most significant features for predictive accuracy.

Logistic Regression includes the following steps:

1. Use linear combination model of the features: <img src="https://latex.codecogs.com/svg.image?z=b_0 + b_1x_1 + b_2x_2 \cdot\cdot\cdot + b_nx_n"/>
2. Put them into Sigmoid function <img src="https://latex.codecogs.com/svg.image?f(z) = \frac{1}{1 + e^{-z}}"/>
3. Maximum Likelihood Estimation Optimization. The goal is to find the set of weights and bias that maximizes the likelihood of the observed outcomes given the input features.

As with any classification task, assessing the model's performance extends beyond accuracy metrics. Further examination of precision, recall, and the Receiver Operating Characteristic (ROC) curve can provide a more nuanced understanding of the Logistic Regression model's strengths and areas for improvement.

##### Logistic Regression Confusion Matrix Without PCA Features (original features)
![image](https://github.gatech.edu/storage/user/56953/files/e2a1f300-b15a-4af5-be31-29b95f16aeea) 

##### Logistic Regression ROC and Precision-Recall Without PCA Features (original features)
![Untitled](https://github.gatech.edu/storage/user/76361/files/f63de7eb-9c59-4a86-8326-e4a0f1584852)

##### Logistic Regression Confusion Matrix With PCA Features (feature reduction)
![image](https://github.gatech.edu/storage/user/56953/files/f2d93635-fd60-4aae-ab23-001464e18320)

##### Logistic Regression ROC and Precision-Recall With PCA Features (feature reduction)
![Untitled](https://github.gatech.edu/storage/user/76361/files/92f42a97-f678-46f8-97f8-6df244a78018)

#### Support Vector Machines (SVM)

In the context of binary classification for our labeled data, where the objective is to distinguish between churn and non-churn instances, we employed the Support Vector Machine (SVM) classifier. This evaluation involved a comprehensive assessment of its performance using both our original features (comprising 73 features without PCA) and PCA features (reduced to 20). Upon a rigorous comparative analysis, our SVM model leveraging PCA features demonstrated robust performance, showcasing its competitiveness when compared to the model without PCA feature reduction. This underscores the pivotal role of feature reduction through Principal Component Analysis (PCA) and its positive impact on enhancing SVM's adaptability with a more concise set of features.

SVM, as a powerful classification algorithm, benefits from reduced dimensionality facilitated by PCA. It is essential to recognize that SVM's efficacy is intricately linked to the characteristics of the input feature space. The application of PCA proves to be advantageous by distilling the most influential features, contributing to heightened predictive accuracy in the context of SVM. This underscores the importance of thoughtful feature selection and reduction strategies in optimizing the performance of SVM models for binary classification tasks.

SVM is to optimize the following 
## Objective function:
<img src="https://latex.codecogs.com/svg.image?\text{Maximize } \frac{1}{2} \|w\|^2 - C \sum_{i=1}^{N} \xi_i"/>

## Constraints:

<img src="https://latex.codecogs.com/svg.image?[y_i (w^T x_i + b) \geq 1 - \xi_i, \quad \text{for } i = 1, 2, \ldots, N"/>

<img src="https://latex.codecogs.com/svg.image?\xi_i \geq 0"/>
 
## Variables:
- <img src="https://latex.codecogs.com/svg.image?w"/>: Weights
- <img src="https://latex.codecogs.com/svg.image?C"/>: Regularization Parameter
- <img src="https://latex.codecogs.com/svg.image?xi_i"/>: Slack Variables

In the evaluation of our Support Vector Machine (SVM) classifier for the binary churn prediction task, we delved beyond basic accuracy metrics to gain a more comprehensive understanding of its performance. Precision, recall, and the Receiver Operating Characteristic (ROC) curve were analyzed to unearth the strengths and potential areas for improvement in the SVM model.

##### SVM Confusion Matrix Without PCA Features (original features)
![image](https://github.gatech.edu/storage/user/56953/files/a51b223d-9069-4794-bb09-5700976d4d0f) 

##### SVM ROC and Precision-Recall Without PCA Features (original features)
![Untitled](https://github.gatech.edu/storage/user/76361/files/d8adaca0-dc4a-4960-b382-02263338306d)

##### SVM Confusion Matrix With PCA Features (feature reduction)
![image](https://github.gatech.edu/storage/user/56953/files/42ff5f81-7708-4328-8e75-04a46f1636c4)

##### SVM ROC and Precision-Recall With PCA Features (feature reduction)
![Untitled](https://github.gatech.edu/storage/user/76361/files/836a9eff-96db-4f3b-93ac-d23c33e04657)

#### Comparing Performance Metrics Across Different Models

Our results show that Random Forest, Logistic Regression and SVM, in the context of churn prediction, provides valuable insights when considering precision and recall. Precision assesses the accuracy of positive predictions, capturing the proportion of correctly identified churn cases among all instances predicted as churn. Recall, on the other hand, measures the model's ability to correctly identify all actual churn cases, highlighting its sensitivity to capturing true positives.

Furthermore, the ROC curve illustrates the trade-off between sensitivity and specificity across different threshold values, offering a visual representation of the model's discriminatory power. The area under the ROC curve (AUC-ROC) provides a summarized performance metric, where a higher AUC-ROC value signifies improved overall model performance.

                                                 
| Performance Metric  | Accuracy | Precision | Recall | F score
| ------------- |:-------------:|:-----:|:-----:|-----:|
| Random Forest with PCA| **0.72**| 0.69| **0.72**| **0.63**
| Random Forest without PCA | 0.71| 0.64| 0.71| 0.61  
| Logistic Regression with PCA| 0.72| 0.64| 0.72| 0.61
| Logistic Regression without PCA | 0.71| 0.59| 0.71| 0.60
| SVM with PCA|  0.72| **0.75**|  0.72| 0.60
| SVM without PCA |  0.71| 0.63|  0.71| 0.59

Upon evaluating the comparison, it is evident that our Random Forest model outperformed others across all the performance metrics illustrated in the table above.

Since our best performing model is Random Forest with PCA but we cannot visualise feature importances after PCA, we have below the feature importance plot for Random Forest Classifier without PCA:
![Untitled](https://github.gatech.edu/storage/user/76361/files/1664603f-741c-4515-89bd-f25e65dfea64)


##### SHAP Explainability

Before using SHAP for model interpretability, we used ROC curve (a receiver operating characteristic curve) and confusion matrix to test our black-box model performance for our classification task. The SHAP results in telco customer churn prediction and customer segmentation provided insights into the importance and impact of different features on the prediction or segmentation outcome. 

This information can help telcos understand the key drivers behind churn and prioritize their retention efforts accordingly. The SHAP values can highlight the specific features that contribute positively or negatively to the churn prediction, providing a clear understanding of the factors influencing customer churn.

We leveraged the SHAP (SHapley Additive exPlanations) method for interpreting our black-box models, including Random Forest, Logistic Regression, and SVM. This analysis aimed to compare the interpretability achieved through SHAP with the conventional feature importance techniques, such as Gini importance, PCA. The objective was to assess whether applying the SHAP method provides a superior means of visualizing model results and interpreting their significance. This comparative evaluation contributed valuable insights into the effectiveness of SHAP in enhancing our understanding of the underlying factors influencing model predictions. The SHAP values highlighted the specific features that contributed positively or negatively to the churn prediction, providing a clear understanding of the factors influencing customer churn either on a global scale or local explanation.


##### SHAP Global Explanation for Random Forest Model

![shap2](https://github.gatech.edu/storage/user/56953/files/3f92abb7-2e91-4138-ae00-50a6bb15a822)

We used our 'without PCA' data from our Random Forest model to generate our shapley values. From our SHAP global feature importance plot above, our top five features for customer churn include CurrentEquipmentDays, MonthsInService, PercChangeMinutes, MonthlyMinutes, TotalRecurringChange. This means that these features consistently contributed the most to the likelihood of churn across the entire customer base. This insight is valuable for strategic decision-making, allowing businesses to focus on key features influencing churn at a broad level. 


##### SHAP Local Explanation for Random Forest Model

![shap3](https://github.gatech.edu/storage/user/56953/files/8ee2c6fa-0f3c-4903-a12b-207713800513)

Local explanations help understand why certain customers are grouped together. Local SHAP values can reveal which specific features led to the classification for each customer being high risk for churn or not. Local explanations empower personalized interventions and by knowing which features drive a specific prediction allows for targeted actions. From the figure above, it can be deduced that the first customer has lower risk of churning since most of the important features are positively related with only Age and PeakCallsOut that are negatively related (fx) = 0.14 < E[f(x)] = 0.286


##### SHAP Dependence Plot for Random Forest Model

![shap4](https://github.gatech.edu/storage/user/56953/files/ae3e6ddd-f93a-41e4-87b9-09a139fab0ff)

The SHAP Dependence Plot above illustrates how changes in the CurrentEquipmentDays and MonthsInService features influence the model's predictions. In the figure above, there was a positive correlation since the SHAP values increase as CurrentEquipmentDays increases. The higher values of CurrentEquipmentDays are associated with higher predicted outcomes. The dynamics between different features play a crucial role in shaping customer behavior, influencing predictive outcomes, and determining the effectiveness of segmentation strategies.


##### SHAP Heatmap Visualization for Random Forest Model

![shap5](https://github.gatech.edu/storage/user/56953/files/c8e52c60-1c27-4135-b176-59f1008d2dc3)

We visualized the SHAP values for the most influential features across various instances in the dataset. Positive SHAP values are represented in shades of blue and negative values in shades of red. Darker colors indicate a higher magnitude of contribution, providing insights into the features driving the predictions. The color intensity in each cell represents the magnitude and direction (positive or negative) of the SHAP values for that feature for that specific prediction. As shown in the figure above, variations in the top features collectively influence predictions across different instances.


##### SHAP Global Explanation for Logistic Regression Model

![shap6](https://github.gatech.edu/storage/user/56953/files/de75176c-82e5-4aa2-8cf0-2ad2186ca00c)

Utilizing the 'with PCA' dataset derived from our logistic regression model, we applied SHAP values to ascertain feature importance. The SHAP global feature importance plot highlights the top five contributors to customer churn in our analysis: PC1, PC7, PC6, PC17 and PC4. These segments consistently demonstrated the highest impact on the likelihood of churn across the entirety of our customer base.

This explanation offers strategic significance, enabling businesses to concentrate their efforts on the pivotal features that consistently influence churn at a broad level. By recognizing and prioritizing these influential segments, organizations can make informed decisions, directing resources and interventions towards areas crucial for mitigating customer churn.


##### SHAP Local Explanation for Logistic Regression Model

![shap7](https://github.gatech.edu/storage/user/56953/files/f8f01008-f162-4a1f-88b7-5dded47df3cd)

Localized interpretations provide insights into the clustering of specific customers. Through local SHAP values, we can pinpoint the precise features that contributed to categorizing individual customers as either high risk or not for churn. These explanations empower the implementation of personalized interventions. The ability to know which features drive a particular prediction facilitates the formulation of targeted and informed actions tailored to the unique characteristics of individual customers. 

As can be seen above, the SHAP value for a particular feature is - 0.635 and the expected value of f(x) (E[f(x)]) is - 0.919, it means that the feature has a negative impact on the prediction of customer churn. The negative SHAP value indicates that the presence or value of this feature decreases the likelihood of a customer churning. The expected value of f(x) (-0.919) represents the average prediction for customer churn. This means that, on average, the model predicts a lower probability of churn when this feature is present or has a higher value. Therefore, the feature with a SHAP value of -0.635 has a negative impact on the prediction of customer churn, and on average, when this feature is present or has a higher value, the predicted probability of churn is lower.


##### SHAP Dependence Plot for Logistic Regression Model

![shap8](https://github.gatech.edu/storage/user/56953/files/51911771-36e7-4f9b-a9cc-3e0763a07253)

The SHAP Dependence Plot depicted above shows the impact of variations in the PC1 and PC17 segments on the model's predictions. As observed in the figure above, a negative correlation is evident. Since the line slopes downward from left to right, it indicates a negative correlation. This means that an increase in the feature value is associated with a decrease in the predicted outcome.


##### SHAP Heatmap Visualization for Logistic Regression Model

![image](https://github.gatech.edu/storage/user/56953/files/bae25b3c-3e52-4dc5-9608-56b254f01a07)


We employed visualization techniques to depict the SHAP values corresponding to the most influential features across diverse instances within the dataset. Positive SHAP values are portrayed through a spectrum of blue hues, while negative values are depicted in various shades of red. The depth of color intensity serves as a visual indicator of the magnitude of contribution, with darker hues indicating a more substantial impact. The visualization encapsulates the intricate dynamics of features, showcasing their individual contributions to predictions. The varying color intensities within each cell offer a clear representation of the magnitude and direction (positive or negative) of the SHAP values associated with specific features for each distinct prediction. As evidenced in the figure above, discernible patterns emerge, illustrating how alterations in the top segments collectively influence predictions across a spectrum of instances within the dataset.


##### SHAP Global Explanation for SVM Model

![shap9](https://github.gatech.edu/storage/user/56953/files/457f4d84-673b-4b84-8f59-6ebeb3528107)

Leveraging the 'with PCA' dataset derived from our SVM model, we applied SHAP values to discern the significance of features. The SHAP global feature importance plot explains the primary contributors to customer churn within our analysis: PC12, PC1, PC5, PC15, and PC13. These segments consistently exhibited the most substantial impact on the likelihood of churn across our entire customer base.

This explanation holds strategic importance, providing businesses with the insight to focus efforts on the key features consistently influencing churn at a broad level. Recognizing and prioritizing these influential segments empowers organizations to make informed decisions, strategically allocating resources and interventions to areas critical for mitigating customer churn.


##### SHAP Local Explanation for SVM Model

![image](https://github.gatech.edu/storage/user/56953/files/7e6a740f-c78d-47de-b690-000156e085f0)

Localized interpretations offer valuable insights into the grouping of distinct customer profiles. Utilizing local SHAP values, we can precisely identify the features responsible for classifying individual customers as high risk or not for churn. These detailed explanations empower the implementation of personalized interventions, allowing for targeted actions based on a clear understanding of each customer's characteristics. The capability to discern which features drive specific predictions facilitates the formulation of informed and tailored strategies, ensuring a more effective and customized approach to addressing customer dynamics.

As shown above, the SHAP value for a particular feature is 0.3 and the expected value of f(x) (E[f(x)]) is 0.282, it means that this feature has a positive impact on the prediction of customer churn. The positive SHAP value indicates that the presence or value of this feature increases the likelihood of a customer churning. The expected value of f(x) (0.282) represents the average prediction for customer churn. This means that, on average, the model predicts a higher probability of churn when this feature is present or has a higher value. Hence, the SHAP value of 0.3 has a positive impact on the prediction of customer churn, and on average, when this feature is present or has a higher value, the predicted probability of churn is higher.


##### SHAP Dependence Plot for SVM Model

![image](https://github.gatech.edu/storage/user/56953/files/e0d1180a-12c0-4a3c-bc7c-6eb6a0f4b101)

The SHAP Dependence Plot presented above illustrates the influence of fluctuations in the PC1 and PC12 segments on the model's predictions. As discerned in the figure, a positive correlation is apparent. The downward slope of the line from right to left confirms this positive correlation. In practical terms, this signifies that as the values of the PC1 and PC12 segments increase, there is a corresponding increase in the model's predicted outcomes.

##### SHAP Heatmap Visualization for SVM Model

![image](https://github.gatech.edu/storage/user/56953/files/56b5f834-b6ee-46a3-9af0-3c4c1d40fa81)

We applied visualization techniques to represent SHAP values corresponding to the most influential features across a diverse set of instances in the dataset. Positive SHAP values are depicted using a spectrum of blue hues, while negative values are presented in varying shades of red. The depth of color intensity serves as a visual gauge of the contribution magnitude, where darker hues signify a more substantial impact. This visualization encapsulates the intricate dynamics of features, providing a comprehensive display of their individual contributions to predictions. The varied color intensities within each cell distinctly convey the magnitude and direction (positive or negative) of SHAP values associated with specific features for each unique prediction. As depicted in the figure above, discernible patterns emerge, highlighting how variations in the top segments collectively influence predictions across a spectrum of instances in the dataset.


#### Conclusion

Tree-based models inherently capture complex interactions and non-linearities in data, making SHAP particularly effective in revealing the intricate feature contributions and dependencies within these models for customer churn prediction and segmentation. In contrast, logistic regression, being a linear model, may struggle to capture and express the nuanced relationships present in intricate datasets, limiting the depth of interpretability achievable through SHAP.

Leveraging SHAP as an interpretable machine learning approach provides businesses with valuable insights for strategic decision-making in customer churn prediction and segmentation. The ability to comprehend the impact of key features both globally and locally enhances the interpretability of complex models like Random Forest. 

On a broad level, SHAP enables businesses to identify and prioritize features influencing churn across the entire customer base. This global understanding aids in resource allocation, strategic planning, and the formulation of proactive retention strategies. Recognizing which features play a pivotal role in the model's predictions allows for the development of targeted interventions to address overarching patterns in customer behavior.

At the local level, understanding whether features have a positive or negative influence on predictions provides a better view. This level of interpretability allows businesses to tailor their approach to individual customers or specific segments. Personalized retention efforts can be designed based on the specific characteristics driving a customer's likelihood of churning. 

The interpretability offered by SHAP not only facilitates effective communication of model behavior but also instills confidence in the decision-making process. It empowers stakeholders to make informed choices, adapt strategies based on real-time insights, and ultimately enhance customer satisfaction and loyalty. In an era where transparency and accountability in machine learning are paramount, SHAP stands as a powerful tool for aligning predictive models with business objectives while ensuring a customer-centric and strategic approach to addressing churn.


#### References

[1] Zadoo A, Jagtap T, Khule N, Kedari A, Khedkar S. A review on Churn Prediction and Customer Segmentation using Machine Learning. In: 2022 International Conference on Machine Learning, Big Data, Cloud and Parallel Computing (COM-IT-CON); Faridabad, India; 2022:174-178. doi:10.1109/COM-IT-CON54601.2022.9850924.

[2] Breiman L. Random Forests. Machine Learning. 2001;45:5-32. doi:10.1023/A:1010933404324.

[3] Mnassri B. Customer Churn Prediction: Telecom Churn Dataset. Kaggle.com. Published 2019. Accessed October 6, 2023. Available from: https://www.kaggle.com/code/mnassrib/customer-churn-prediction-telecom-churn-dataset/notebook#4.-Model-Building.

[4] Telecom churn (cell2cell) https://www.kaggle.com/datasets/jpacse/datasets-for-churn-telecom

[5] Lundberg, S. M., & Lee, S-I. (2017). A Unified Approach to Interpreting Model Predictions. In Proceedings of the 31st Conference on Neural Information Processing Systems (NeurIPS), Long Beach, CA, USA.

[6] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.


Link to our dataset: https://drive.google.com/drive/folders/1Tb1ntfGCRgwVLek2e2IKTTYiZKsSoPSW?usp=sharing

Link to YouTube for final video presentation: https://youtu.be/TgT8dDXehqQ
