# Medical_Speciality_Classification_from_Transcription

**Introduction:** Medical transcription involves converting medical dictation into a written or electronic format by healthcare professionals. It is crucial in maintaining accurate medical records for patient care and billing. The medical specialty of the professional can be identified based on the content of the transcription. However, classifying these transcriptions manually can be time-consuming. Automatic classification based on a medical specialty can save time for transcriptionists. It can also help to ensure patients receive appropriate follow-up care. In a hospital system, there is a vast amount of medical transcription generated daily. Preparing an automated classification is even more valuable. Overall, automated medical transcription classification can improve efficiency and accuracy in the healthcare industry. [1,2]

**Dataset and Methodology:** Due to HIPPA privacy regulations, it is hard to find medical datasets. In this project, an open-source dataset from Kaggle is employed [3]. A sample of the dataset is presented in figure 1. This dataset contains 4999 rows and 6 columns including ‘row_no’, ‘description’, ‘medical_speciality’, ‘sample_name’, ‘transcription’, and ‘keywords’.  The project goal is to classify ‘Medical Speciality’, so ‘medical_speciality’ will be the label. The ‘Transcription’ field contains the medical transcription and this file will be the feature field for the model. Other fields such as ‘sample_name’, and ‘keywords’ are not very informative for the project goal. 

![alt text](/Images/Data_samples.png)
Fig: 1 Snapshot of the dataset.

The dataset has 40 ‘medical specialty’ classes/categories and these classes are highly imbalanced. The class ‘surgery’ has more than 1100 observations on the other hand ‘Hospice - Palliative Care’ class has only 6 observations. This type of unbalanced dataset significantly impacts model performance. If the majority class is heavily overrepresented in the dataset, the model may become biased toward predicting that class. This can result in poor performance for the minority class. An imbalanced dataset can make it easier for the model to overfit, especially if the minority class is poorly represented. There are several strategies to address the impact of an imbalanced dataset on model performance, including oversampling the minority class, undersampling the majority class, or using more advanced techniques such as SMOTE (Synthetic Minority Over-sampling Technique). Unfortunately, SMOTE doesn’t work well with text data because the numerical vectors that are created from the text are very highly dimensional [4]. 

Over-sampling/up-sampling and under-sampling/down-sampling techniques are employed for this project. Most of the classes are minorities in this dataset. There are 11 classes with more than 100 observations, 8 classes with more than 200 observations, and 2 classes with more than 500 observations. For minority classes, the up-sampling technique is employed and the down-sampling technique is employed for only the ‘surgery’ class. Because this class is only oversampled. For down-sampling, the cosine similarity technique is used. If 2 observations are more than 70% similar, 1 observation is removed. For up-sampling, the observations of minority classes are duplicated once and added to the dataset. So the size of each minor class has become twice. 

To fit text data in machine learning models, several text preprocessing steps are required. Two major steps are data cleaning and data transformation. The data cleaning step includes removing punctuations, stop words and digits, lowercase text, generating tokens, etc. The second step in text data preprocessing is to transform the text into a format suitable for analysis. This involves converting the text into numerical features that machine learning algorithms can use. Common techniques for text transformation include bag-of-words representation, where each document is represented as a vector of word counts, and TF-IDF (Term Frequency-Inverse Document Frequency). In this project, the TF-IDF technique is used to represent the feature matrix for text transformation.

Text data is often represented as a sparse matrix. So in this project, PCA (Principal Component Analysis) is employed to reduce dimensionality. Though PCA may not work well on text data because text data has a unique structure and properties that differ from continuous data. After reducing the dimension, to classify ‘Medical Speciality’ Naive-Bayes classifier is employed.

**Result analysis:** It is anticipated that with all 40 classes, the accuracy of the model will be worse and the accuracy is just 1%. So in the next step, a few classes with the lowest observations are removed. There are 11 classes with more than 100 observations and the accuracy of the Naive Bayes classifier is just around 25%. Therefore, classes with more observations are required for better model performance. The performance of classification models can be impacted by the number of classes. As the number of classes increases, the complexity of the problem increases, making it more difficult for the model to classify instances accurately. For only 2 classes, the Naive Bayes model accuracy is more than 90%. So it can be easily interpreted that the model is working fine with the dataset for binary class classification. To explore more on the dataset I have decided to work with 8 classes that have more than 200 observations. Still, the dataset is imbalanced with 8 classes. First, measure the accuracy of 8 class classifications and the accuracy increased a little bit compared to 11 classifications. Then implemented PCA to reduce dimensionality but accuracy remains almost the same. The accuracy for these approaches is around 33%. In the next step, to resolve the imbalance of the dataset, up-sampling and down-sampling techniques are implemented. The number of observations for each class is shown in table 1.

![alt text](/Images/No_Observations.png)

In table 1, the number of observations are shown before and after implementing Up-sampling and down-sampling techniques. The ‘Surgery’ class is the only class with oversampled data so it is down-sampled. Though there is no major change due to the down-sample. The model performance is better with only up-sampling. In table 2 we can see that the accuracy is 41.2%. The ‘Consult - History and Phy.’ class has 516 observations. So it is excluded from up-sampling.

![alt text](/Images/Result_analysis.png)

**Alternative approaches:** There are other classification approaches which can be implemented but in my opinion, Naive Bayes is best fit for this dataset. For instance, kNN is a lazy learning algorithm that relies on the proximity of instances to make decisions and in this case, the features are overlapped. The Decision Tree is vulnerable to imbalanced data and easily overfits in an imbalanced dataset.

**Conclusion:** In conclusion, automated medical transcription classification can improve efficiency and accuracy in the healthcare industry. However, the imbalanced nature of medical datasets poses challenges for machine learning models. To address this, over-sampling and under-sampling techniques are employed. Text preprocessing steps are also necessary to fit text data into machine learning models. In this project, TF-IDF is used for text transformation, and PCA is used to reduce dimensionality. Finally, a Naive-Bayes classifier is employed to classify the ‘Medical Speciality’ of the transcription. Overall, these techniques can improve the accuracy and efficiency of medical transcription classification. This system can lead to better patient care for any hospital transcription management system.


**Reference:** 
1. https://www.coursera.org/articles/medical-transcriptionist
2. https://www.webmd.com/a-to-z-guides/what-is-a-medical-transcriptionist
3. https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions
4. https://towardsdatascience.com/how-i-handled-imbalanced-text-data-ba9b757ab1d8
