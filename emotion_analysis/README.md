# Emotion analysis on arabic tweets
## Usage example
### Install packages
```python
pip install simpletransformers
pip install wandb
pip install tweet-preprocessor
pip install farasapy
pip install pyarabic
pip install transformers
pip install nltk
pip install spacy
cd emotion_analysis
git clone https://github.com/aub-mind/arabert
```
### Train the model
```python
import pandas as pd
from emotion_analysis.emotion_clf import EmotionMultilabelClassification

# Preprocess the data -> train the model
emc = EmotionMultilabelClassification() # Init the model
emc.fit(train_data,eval_data) # Train the model
```
### Evaluate the model performance

```python
# Preprocess evaluation data -> init the model -> evaluate results
emc = EmotionMultilabelClassification(verbose=False) # Init the model, disable verbose (optional)
eval_df = emc.transform(test_data,evaluate=True,return_df=True) # Set 'evaluate=True' to get a performance report
print(emc.evaluation_report) # Print the model performance report
```
### Predict emotion on new data
```python
# Preprocess new data -> init the model -> predict results
unlabeled_data =["انها مثل هذا اليوم الجميل",
                 "كان لدي أسبوع مرهق ... لا بد لي من العودة إلى المنزل"]
emc = EmotionMultilabelClassification() # Init the model
# Returns full dataframe
# Save the full results (prediction and probabilities) to csv
prediction_df = emc.transform(unlabeled_data,return_df=True,results_to_csv=True)
# Returns only the array of probabilities
probabilities = emc.transform(unlabeled_data)
```