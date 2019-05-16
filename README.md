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
- Visualisation of multidimensional data [Basic knowledge of this - would be interesting to see what we can do]
