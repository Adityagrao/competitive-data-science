## Recap of main ML algorithms
- [Explanation of Random Forest](http://www.datasciencecentral.com/profiles/blogs/random-forests-explained-intuitively)
- [Explanation/Demonstration of Gradient Boosting](http://arogozhnikov.github.io/2016/06/24/gradient_boosting_explained.html)
- [Example of kNN](https://www.analyticsvidhya.com/blog/2014/10/introduction-k-neighbours-algorithm-clustering/)
#### Additional Tools
- [Example from sklearn with different decision surfaces](http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html)
- [Arbitrary order factorization machines](https://github.com/geffy/tffm)

## Software/Hardware requirements
#### AWS spot option:
- [Overview of Spot mechanism](http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-spot-instances.html)
- [http://www.datasciencebowl.com/aws_guide/](http://www.datasciencebowl.com/aws_guide/)
#### Stack and packages:
- [Blog "datas-frame" (contains posts about effective Pandas usage)](https://tomaugspurger.github.io/)

## Feature preprocessing and generation with respect to models
#### Feature preprocessing
- [Preprocessing in Sklearn](http://scikit-learn.org/stable/modules/preprocessing.html)
- [Andrew NG about gradient descent and feature scaling](https://www.coursera.org/learn/machine-learning/lecture/xx3Da/gradient-descent-in-practice-i-feature-scaling)
- [Feature Scaling and the effect of standardization for machine learning algorithms](http://sebastianraschka.com/Articles/2014_about_feature_scaling.html)
#### Feature generation
- [Discover Feature Engineering, How to Engineer Features and How to Get Good at It](https://machinelearningmastery.com/discover-feature-engineering-how-to-engineer-features-and-how-to-get-good-at-it/)
- [Discussion of feature engineering on Quora](https://www.quora.com/What-are-some-best-practices-in-Feature-Engineering)

## Feature extraction from text and images
#### Bag of words
- [Feature extraction from text with Sklearn](http://scikit-learn.org/stable/modules/feature_extraction.html)
- [More examples of using Sklearn](https://andhint.github.io/machine-learning/nlp/Feature-Extraction-From-Text/)
#### Word2vec
- [Tutorial to Word2vec](https://www.tensorflow.org/tutorials/word2vec)
- [Tutorial to word2vec usage](https://rare-technologies.com/word2vec-tutorial/)
- [Text Classification With Word2Vec](http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/)
- [Introduction to Word Embedding Models with Word2Vec](https://taylorwhitten.github.io/blog/word2vec)
#### NLP Libraties
- [NLTK](http://www.nltk.org/)
- [spaCy](https://spacy.io/)
- [TextBlob](https://github.com/sloria/TextBlob)
#### Pretrained models
- [Using pretrained models in Keras](https://keras.io/applications/)
- [Image classification with a pre-trained deep neural network](https://www.kernix.com/blog/image-classification-with-a-pre-trained-deep-neural-network_p11)
#### Finetuning
- [How to Retrain Inception's Final Layer for New Categories in Tensorflow](https://www.tensorflow.org/tutorials/image_retraining)
- [Fine-tuning Deep Learning Models in Keras](https://flyyufelix.github.io/2016/10/08/fine-tuning-in-keras-part2.html)

## Exploratory data analysis
- [Biclustering algorithms for sorting corrplots](http://scikit-learn.org/stable/auto_examples/bicluster/plot_spectral_biclustering.html)

## Validation
- [Validation in Sklearn](http://scikit-learn.org/stable/modules/cross_validation.html)
- [Advices on validation in a competition](http://www.chioka.in/how-to-select-your-final-models-in-a-kaggle-competitio/)

## Data leakage
- [Perfect score script by Oleg Trott](https://www.kaggle.com/olegtrott/the-perfect-score-script) - used to probe leaderboard
- [Page about data leakages on Kaggle](https://www.kaggle.com/wiki/Leakage)

## Metrics optimization
#### Classification
- [Evaluation Metrics for Classification Problems: Quick Examples + References](http://queirozf.com/entries/evaluation-metrics-for-classification-quick-examples-references)
- [Decision Trees: “Gini” vs. “Entropy” criteria](https://www.garysieling.com/blog/sklearn-gini-vs-entropy-criteria)
- [Understanding ROC curves](http://www.navan.name/roc/)
#### Ranking
- [Learning to Rank using Gradient Descent](http://icml.cc/2015/wp-content/uploads/2015/06/icml_ranking.pdf) - original paper about pairwise method for AUC optimization
- [Overview of further developments of RankNet](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf)
- [RankLib](https://sourceforge.net/p/lemur/wiki/RankLib/) (implemtations for the 2 papers from above)
- [Learning to Rank Overview](https://wellecks.wordpress.com/2015/01/15/learning-to-rank-overview)
#### Clustering
- [Evaluation metrics for clustering](http://nlp.uned.es/docs/amigo2007a.pdf)

## Hyperparameter tuning
- [Tuning the hyper-parameters of an estimator (sklearn)](http://scikit-learn.org/stable/modules/grid_search.html)
- [Optimizing hyperparameters with hyperopt](http://fastml.com/optimizing-hyperparams-with-hyperopt/)
- [Complete Guide to Parameter Tuning in Gradient Boosting (GBM) in Python](https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/)

## Tips and tricks
- [Far0n's framework for Kaggle competitions "kaggletils"](https://github.com/Far0n/kaggletils)
- [https://www.dataquest.io/blog/jupyter-notebook-tips-tricks-shortcuts/](https://www.dataquest.io/blog/jupyter-notebook-tips-tricks-shortcuts/)

## Advanced features
#### Matrix Factorization:
- [Overview of Matrix Decomposition methods (sklearn)](http://scikit-learn.org/stable/modules/decomposition.html)
#### t-SNE:
- [Multicore t-SNE implementation](https://github.com/DmitryUlyanov/Multicore-TSNE)
- [Comparison of Manifold Learning methods (sklearn)](http://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html)
- [How to Use t-SNE Effectively (distill.pub blog)](https://distill.pub/2016/misread-tsne/)
- [tSNE homepage (Laurens van der Maaten)](https://lvdmaaten.github.io/tsne/)
- [Example: tSNE with different perplexities (sklearn)](http://scikit-learn.org/stable/auto_examples/manifold/plot_t_sne_perplexity.html#sphx-glr-auto-examples-manifold-plot-t-sne-perplexity-py)
#### Interactions:
- [Facebook Research's paper about extracting categorical features from trees](https://research.fb.com/publications/practical-lessons-from-predicting-clicks-on-ads-at-facebook/)
- [Example: Feature transformations with ensembles of trees (sklearn)](http://scikit-learn.org/stable/auto_examples/ensemble/plot_feature_transformation.html)
