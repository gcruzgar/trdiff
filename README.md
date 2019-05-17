# trdiff

The overall objective is to understand what makes a text difficult to translate.
Can we modify or improve existing methods? 
Improve NLP methods by making alterations to input texts or by selecting how and when new examples are taught to the algorithm. 

### Things to try:

- Linear regression using Biber dimensions (predicting words per day):
  + UN texts  [work started]
  + WTO texts [work started]
- Prediction of MT errors and it's relation to text difficulty - domain adaptation [Need guidance but should be straight forward]
  + Build model and test on UN data to predict words per day
- Test QEBrain - MT errors without reference
- Semi-supervised approaches and NN: [Relatively new field - can learn basic NN and try some of these]
  + Curriculum learning - X. Zuang et al. 2018, "An Empirical Exploration of Curriculum Learning for Neural Machine Translation" 
  + denoising MT - W. Wang et al. 2018, "Denoising Neural Machine Translation Training with Trusted Data and Online Data Selection"
  + Bidirectional Language Model (biLM) - M. E. Peters et al 2018, "Deep contextualized word representations" 
  + Y. Yang et al 2019, "Improving Multilingual Sentence Embedding using Bi-directional Dual Encoder with Additive Margin Softmax"
  + K. Fan et al 2018, "Bilingial Expert Can Find Translation Errors"
  + D. Yogotama et al 2019, "Learning and Evaluating General Linguistic Intelligence"
  + G. Kostopoulos et al 2018, "Semi-Supervised Regression: A recent review"
- Visualisation of multidimensional data [Basic knowledge of this - would be interesting to see what we can do]

## Biber Dimensions - words per day

Biber dimensions (lexical, syntactic, tex-level... features in texts) can be used to build regression models predicting the rate of translation of documents (in words per day).

See [biberpy](https://github.com/ssharoff/biberpy) for extraction of Biber dimensions in `python`. 

The UNOG (around 200 documents) and WTO (around 100 documents) datasets contain metadata including time taken to translate each document. 

Preliminary results using ordinary least squares regression show a weak correlation between biber dimensions and words translated per day. However, there is still large error in the predicted values and the residuals are not completly random error. This could be due to uncertainty in the data itself and other factors affecting the rate of translation that havent been accounted for. The results can be improved slightly by using the total number of words in the document and the category or topic of the document (e.g. which department of the UN) up to an r2-score = 0.43. Using other linear regression methods such as Ridge Regression and Lasso Regression offer very similar results.  

![UN_OLS](img/un_wpd_ols.png)    
**Figure 1** Predicted against real values of words translated per day for the UNOG dataset (using biber dimensions and number of words in each document).

![UN_OLS_Residuals](img/un_wpd_ols_residuals.png)    
**Figure 2** Difference in predicted and real values for the UNOG dataset. Note the appearance of a trend, possibly due to a systematic error or the increased uncertainty in documents that took to long (external reasons) or too short (lowest timeframe visible is one day).

## Semi-Supervised Regression
Due to the limit in labeled data (currently have access to ~300 documents) and the easier access to unlabeled data, semi-supervised regression is a good candidate for improved regression models.

The general idea is to use labeled data to predict unlabeled data, accept the results above a confidence threshold, using these new labels to predict the remaining unlabeled data and so on and so forth. For example, the labeled UNOG dataset could be combined with the larger unlabeled UN corpus to predict the words translated per day, ultimately linked with text difficulty. 
