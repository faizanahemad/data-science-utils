# Project/Publishable Ideas
For ideas to be viable it has to be computationally possible, and be programmable in Pytorch or keras or base python

### Deep Learning
1. Video Compression using DVAE, using super resolution techniques and Bidirectional LSTM time series
2. Snapshot Ensembling of Neural Networks
    - Instead of normal snapshot ensembling (where the idea is to increase LR after a local minima and then find a new minima and add it to existing snapshots) we propose a new method in which a new snapshot is only added to existing snapshots if it is significantly different from existing snapshots.
    - Lets say that our DNN has N parameters (weights), our error function is a function in the weight space. To determine if a snapshot is different we take it's Weight Wn and calculate cosine/euclid distances from all existing weights W1...n-1. If it is having significant difference above a threshold, then we take the snapshot.
    - A generalised process: Intelligent Snapshot selection to minimise overfitting and variance, and increase generalization. For example 1 process could be that the Validation error is lower than atleast half the existing snapshots and weight (Wn to W1..n-1 distance) distance should be above a threshold and predictions should be have less some threshold correlation.
    - https://medium.com/analytics-vidhya/snapshot-ensembles-leveraging-ensembling-in-neural-networks-a0d512cf2941


### Feature Engineering 
1. Categorical features where row/example has multiple categories
    - Method 1
        - Get WOE per category
        - See which categories the entity belongs to and take weighted (by num examples) / normal average of woe to arrive at final score.
    - Method 2
	    - Same as method 1 except that for each entity we also take nearest categories to its own categories for robustness
    - Method 3
        - Use Keras Embedding layer, feed categories as an array (like text), finally use GAP to get a vector
2. Categorical features where categories have few examples
    - for low number of entiities category, get similar categories and use woe of nearby categories.
    - Getting Similar categories How?
    - Use a weighted by examples woe or Combine categories own woe with nearby cats' woe but give more weight to its own woe.
    - This will add more robustness and prevent too much variance for categories with less examples.
3. Idea is to not treat each category as completely distinct entities but rather have vector representations of categories where categories which are similar are nearer in the N-D vector space.
    - Keras Embeddding layer be default is trained to minimise loss, and could lead to high variance
    - First we will train embedding layer to minimise category aggregate statistics.
    - Next We will train embedding layer to minimise loss in actual training
    - Keras Embedding layer can take only use only 1 categorical column, we will ensure that multiple categorical columns can be used at once.
        - Label Encoding and either Flatten or GAP  
    - This approach is very similar to auto-encoder approach but since it is also trained during main training it should find more interactions 

### Trees and RFs
1. Explainable Trees
    - Each feature can be used only once in a path from root to leaf 
    - Each node is a b-tree node instead of binary node
    - B-tree nodes for categorical features will create as many branches as categorical variables cardinality.
    - For the categories that have less than x% of total example, we will put all of them into misc
    - we can also specify for a categorical column which categories can be misc
    
### Training and Tuning
1. Probabilistic Pruning of GridSearch CV Runs for speed.
    - Inspired from https://towardsdatascience.com/pruned-cross-validation-for-hyperparameter-optimization-1c4e0588191a
    - We estimate the above pdf of Model performance over hyperparameter space after each CV fold run.
    - When a new Hyperparam set is being evaluated, we will say that mean of that hyperparam set is mean of whatever folds has been evaluated for that set till now. 
    - Global mean and std are derived from all runs till now. As more runs happen, global mean and std will become clearer, similarly for current set also as more folds are processed its overall score (mean) will become clearer. 
    - Our pruning method takes 
       - Num total runs
       - Score of each run
       - current hyperparam set
       - current hyperparam set's num folds evaluated and score
    - Output of our pruning method - probability of current hyperparam set's score being lower than global mean/global best.
    - We prune (stop remaining folds of current hyper-params set to run) based on this probability
    

### Multi-Algorithm
1. Cohort based Linear Regression
    - Create cohorts/sub-groups of data using either clustering or Decision Trees
    - Add a penalty term for num of clusters/depth of tree so we have shallow trees. 
    - Run LR separately within each cohort
    - Easier explainability and faster training+inference
2. XGB + RF
    - Use XGB as base learner
    - Use RF as the ensembling method for multiple XGB
    - This can avoid the XGB overfitting
    
    
# Resources
https://roamanalytics.com/2016/10/28/are-categorical-variables-getting-lost-in-your-random-forests/

Video Compression

https://dsp.stackexchange.com/questions/35953/can-deep-learning-be-applied-to-video-compression
https://arxiv.org/abs/1804.09869
http://cs231n.stanford.edu/reports/2017/pdfs/423.pdf
https://towardsdatascience.com/tag2image-and-image2tag-joint-representations-for-images-and-text-9ad4e5d0d99