# translationdiff

The overall objective is to understand what makes a text difficult to translate.
Can we modify or improve existing methods? 
Improve NLP methods by making alterations to input texts or by selecting how and when new examples are taught to the algorithm. 

### Things to try:
- Linear regression using Biber dimensions (predicting words per day):
  + UN texts  [work started]
  + WTO texts [work started]
- Prediction of MT errors and it's relation to text difficulty - domain adaptation
  + Build model and test on UN data to predict words per day
- Test QEBrain - MT errors without reference
- Semi-supervised approaches and NN:
  + Curriculum learning?
  + denoising MT?
- Visualisation of multidimensional data
