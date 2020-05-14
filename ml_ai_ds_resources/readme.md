<!-- >>>>>> BEGIN GENERATED FILE (include): SOURCE ml_ai_ds_resources/readme_template.md -->

## Grasping Machine Learning with Mathematical Theory — Resources

Or a Bag of resources for ML and maths in ML

With the advent of multiple libraries machine learning work has been democratised in a big way. Its easy to now build your own model following a tutorial. But what remains missing is our ability to dive into research and contribute to solving harder problems. To be able to apply ML algorithms to novel situations we need to understand how they work and where they are suited.

This type of deep understanding of ML algorithms can be achieved by knowing the fundamentals, not just by using library APIs. As such this guide is focused on providing the fundamental resources on mathematics and machine learning to help us move forward.

Understanding some CS basics before starting ML is very useful. Learn Python programming too.

[Checkout Markdown Helper which helps doing includes](https://github.com/BurdetteLamar/markdown_helper)
Install using `sudo gem install markdown_helper` then in this folder run `markdown_helper include readme_template.md readme.md`

In my suggestion having 2 parallel threads, 1 on ML and 1 on Maths is best. Studying only mathematics will become monotonous, ML study and practice is essential since our goal is to gain deeper understanding of ML. For working on ML try combining theory and also do some practice challenges on [Kaggle](https://www.kaggle.com/).

## Article Contents
- [RoadMap References](#roadmap-references)
- [Machine Learning resources](#machine-learning-resources)
  - [Basic Starter](#basic-starter)
  - [Some Advanced Courses](#some-advanced-courses)
  - [Field Specific Stuff: Deep Learning](#field-specific-stuff-deep-learning)
  - [Field Specific Stuff: Computer Vision](#field-specific-stuff-computer-vision)
  - [Field Specific Stuff: NLP](#field-specific-stuff-nlp)
  - [Field Specific Stuff: Reinforcement Learning](#field-specific-stuff-reinforcement-learning)
  - [Field Specific Stuff: Recommendation Systems](#field-specific-stuff-recommendation-systems)
  - [Other Random Areas/Resources](#other-random-areasresources)
- [Mathematics resources](#mathematics-resources)
  - [Starter](#starter)
  - [Basic: Probability and Statistics](#basic-probability-and-statistics)
  - [Basic: Calculus](#basic-calculus)
  - [Basic: Linear Algebra](#basic-linear-algebra)
  - [Advanced: Differential Equations](#advanced-differential-equations)
  - [Advanced: Optimisation](#advanced-optimisation)
  - [Advanced: Probability, Bayesian and Statistical Inference](#advanced-probability-bayesian-and-statistical-inference)
  - [Advanced: Calculus, Multivariate Calculus and Matrix Calculus](#advanced-calculus-multivariate-calculus-and-matrix-calculus)
  - [Other Topics](#other-topics)
  - [More Math Resources](#more-math-resources)
- [Staying Updated](#staying-updated)
  - [Video Resources](#video-resources)
  - [Article Resources](#article-resources)
- [Finding good Papers and Paper Reading Tips](#finding-good-papers-and-paper-reading-tips)
  - [Finding Good Papers](#finding-good-papers)
  - [Finding Good papers](#finding-good-papers-1)
  - [Reading a Paper](#reading-a-paper)
- [Some Good Papers That I am reading](#some-good-papers-that-i-am-reading)
- [Engineering Aspects](#engineering-aspects)
  - [Before Starting a Project](#before-starting-a-project)
  - [DL](#dl)
  - [Experiment Tracking](#experiment-tracking)
  - [Deployments](#deployments)
  - [Code Examples for Learning](#code-examples-for-learning)
  - [EDA and plotting, visualizations](#eda-and-plotting-visualizations)
  - [Modelling](#modelling)
  - [NLP](#nlp)
  - [RL](#rl)
  - [CV](#cv)
  - [Recommendation Systems](#recommendation-systems)
  - [Explainable AI](#explainable-ai)
  - [Performance](#performance)
  - [Jupyter & Notebooks](#jupyter--notebooks)
  - [Recipes](#recipes)
- [Random Machine Learning Topics/Articles](#random-machine-learning-topicsarticles)
  - [General ML](#general-ml)
  - [Mathematical](#mathematical)
  - [Applied ML](#applied-ml)
  - [Interview Questions](#interview-questions)
  - [Explainability: ](#explainability-)
  - [Deep Learning](#deep-learning)
  - [NLP](#nlp-1)
  - [RL](#rl-1)
  - [Misc](#misc)
- [Datasets ](#datasets-)
- [Generic References](#generic-references)
- [CS Basics](#cs-basics)

<!-- >>>>>> BEGIN INCLUDED FILE (markdown): SOURCE ml_ai_ds_resources/roadmaps.md -->
## RoadMap References

* [ML subreddit Guide to Learning ML](https://www.reddit.com/r/MachineLearning/wiki/index)

* [The Open Source Data Science Masters](http://datasciencemasters.org/)

* [Open Machine Learning Course mlcourse.ai](https://mlcourse.ai/) and [their Roadmap](https://mlcourse.ai/roadmap)

* [Reddit Guides to what maths needs to be read](https://www.reddit.com/r/learnmachinelearning/comments/adwft2/all_the_math_you_might_need_for_machine_learning/)

* [AV DS path](https://www.analyticsvidhya.com/blog/2020/01/learning-path-data-scientist-machine-learning-2020/), [NLP path](https://www.analyticsvidhya.com/blog/2020/01/learning-path-nlp-2020/?utm_source=blog&utm_medium=learning-path-data-scientist-machine-learning-2020)

* [DataSchool starter](https://www.dataschool.io/start/)
<!-- <<<<<< END INCLUDED FILE (markdown): SOURCE ml_ai_ds_resources/roadmaps.md -->

<!-- >>>>>> BEGIN INCLUDED FILE (markdown): SOURCE ml_ai_ds_resources/ml_learning.md -->
## Machine Learning resources

While I have listed the courses here, it is best to visit them and see their syllabus, many have redundant content.

### Basic Starter

* One of these basic courses
    - [Stanford CS229](https://see.stanford.edu/course/cs229) ([2018 Videos](https://www.youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU), [2014 Videos](https://www.youtube.com/playlist?list=PLA89DCFA6ADACE599))[[Notes](http://cs229.stanford.edu/syllabus.html) [Cheatsheet](https://stanford.edu/~shervine/teaching/cs-229/cheatsheet-supervised-learning)](A more mathematical version of [Andrew ng’s Coursera Course](https://www.coursera.org/learn/machine-learning)|[Coursera course Videos](https://www.youtube.com/playlist?list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN)) or 
    - [ML carnegie melon 2011](http://www.cs.cmu.edu/~tom/10701_sp11/lectures.shtml) or 
    - [Stanford Statistical Learning](https://www.edx.org/course/statistical-learning) or 
    - [Caltech ML basics](https://work.caltech.edu/telecourse.html#lectures) ([Videos](http://work.caltech.edu/lectures.html)) or 
    - [Udacity GTech ML](https://www.udacity.com/course/machine-learning--ud262)
    - [Google ML Crash Course](https://developers.google.com/machine-learning/crash-course/ml-intro)
    - [EDX Machine Learning Fundamentals](https://courses.edx.org/courses/course-v1:UCSanDiegoX+DSE220x+3T2019/course/)

* [MIT Intro to Computational Thinking and Data Science](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-0002-introduction-to-computational-thinking-and-data-science-fall-2016/)

* Books
    - [ISL Book](http://faculty.marshall.usc.edu/gareth-james/ISL/) ([Lecture Vids and Slides](http://fs2.american.edu/alberto/www/analytics/ISLRLectures.html)) [ISL solutions](https://github.com/tdpetrou/Machine-Learning-Books-With-Python/tree/master/Introduction%20to%20Statistical%20Learning) ([Code 1](https://github.com/JWarmenhoven/ISLR-python))

* [Applied Courses: Fast.ai](https://www.fast.ai/)
* [A Few Useful Things to Know about Machine Learning](https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf)
* [Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml/) By Google

* [Kaggle tutorials](https://www.kaggle.com/learn/overview): 
    - [Python](https://www.kaggle.com/learn/python), 
    - [Pandas](https://www.kaggle.com/learn/pandas), 
    - [Data-Visualization](https://www.kaggle.com/learn/data-visualization), 
    - [ML](https://www.kaggle.com/learn/intro-to-machine-learning), 
    - [Intermediate ML](https://www.kaggle.com/learn/intermediate-machine-learning), 
    - [Titanic competition](https://www.kaggle.com/c/titanic), 
    - [Housing Prices](https://www.kaggle.com/c/home-data-for-ml-course)

* [Data Science: Inferential Thinking through Simulations](https://www.edx.org/course/foundations-of-data-science-inferential-thinking-b)

### Some Advanced Courses

* [COMS W4995 Applied Machine Learning Spring 2020](https://www.youtube.com/playlist?list=PL_pVmAaAnxIRnSw6wiCpSvshFyCREZmlM) [Website](https://www.cs.columbia.edu/~amueller/comsw4995s20/), 

* [Another Mathematical Treatment and general coverage of ML by Washington Uni](https://www.youtube.com/playlist?list=PLTPQEx-31JXhguCush5J7OGnEORofoCW9) ([Course Materials](http://mathofml.cs.washington.edu/)) or [CORNELL CS4780 “Machine Learning for Intelligent Systems”](http://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/)

* [CS50’s Introduction to Artificial Intelligence with Python](https://www.edx.org/course/cs50s-introduction-to-artificial-intelligence-with-python)

* [Advanced ML Coursera](https://www.coursera.org/specializations/aml)

* [EDX Machine Learning for Data Science and Analytics](https://courses.edx.org/courses/course-v1:ColumbiaX+DS102X+1T2017/course/)

* [Statistical ML](https://www.youtube.com/playlist?list=PLTB9VQq8WiaCBK2XrtYn5t9uuPdsNm7YE), [Statistical ML 2 ](https://www.youtube.com/playlist?list=PLjbUi5mgii6B7A0nM74zHTOVQtTC9DaCv)

* [Intro to AI Udacity](https://www.udacity.com/course/intro-to-artificial-intelligence--cs271), [AI Columbia Univ EDX ](https://www.edx.org/course/artificial-intelligence-ai)([Link](https://courses.edx.org/courses/course-v1:ColumbiaX+CSMM.101x+1T2020/course/)), [MIT AI Course](https://www.youtube.com/playlist?list=PLUl4u3cNGP63gFHB6xb-kVBiQHYe_4hSi) [[MIT AI OCW](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-034-artificial-intelligence-fall-2010/)], [Gtech Knowledge-Based AI: Cognitive Systems](https://www.udacity.com/course/knowledge-based-ai-cognitive-systems--ud409)

* [MIT 9.520 Statistical Learning Theory](https://www.youtube.com/playlist?list=PLyGKBDfnk-iDj3FBd0Avr_dLbrU8VG73O) 

* [Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/)

* [ESL book](https://web.stanford.edu/~hastie/ElemStatLearn/printings/ESLII_print12.pdf) ([Site](https://web.stanford.edu/~hastie/ElemStatLearn/)) [ESL Solutions](https://waxworksmath.com/Authors/G_M/Hastie/WriteUp/Weatherwax_Epstein_Hastie_Solution_Manual.pdf) ([soln 2](https://blog.princehonest.com/stat-learning/) [[Backup](https://github.com/asadoughi/stat-learning)])([Notes 1](https://github.com/empathy87/The-Elements-of-Statistical-Learning-Python-Notebooks), [2](https://github.com/dgkim5360/the-elements-of-statistical-learning-notebooks), [3](https://github.com/maitbayev/the-elements-of-statistical-learning), [4](https://getd.libs.uga.edu/pdfs/ma_james_c_201412_ms.pdf))

* [Probabilistic Graphical Models Specialization](https://www.coursera.org/specializations/probabilistic-graphical-models), [Probabilistic Graphical Models by CMU](https://www.cs.cmu.edu/~epxing/Class/10708-20/), [Graphical Models: Jeffrey A. Bilmes](https://www.youtube.com/channel/UCvPnLF7oUh4p-m575fZcUxg/videos)

* John Hopkins Advanced Linear Models for Data Science [Part 1](https://www.coursera.org/learn/linear-models) and [Part 2](https://www.coursera.org/learn/linear-models-2).

* [ML by Mathematical Monk on YT](https://www.youtube.com/playlist?list=PLD0F06AA0D2E8FFBA), [Information Theory by Jeffrey A. Bilmes](https://www.youtube.com/channel/UCvPnLF7oUh4p-m575fZcUxg/videos)

* [Statistical Methods in Machine Learning Syllabus](http://www.stat.cmu.edu/~larry/=sml/syllabus.pdf) ([Notes and Exercises](http://www.stat.cmu.edu/~larry/=sml/))

* [Mining Massive Datasets - Stanford University](https://www.youtube.com/playlist?list=PLLssT5z_DsK9JDLcT8T62VtzwyW9LNepV)

### Field Specific Stuff: Deep Learning

* [Deep Learning Basics](https://www.coursera.org/specializations/deep-learning) ([deeplearning.ai](https://www.youtube.com/channel/UCcIXc5mJsHVYTZR1maL5l9w) [1st course](https://www.youtube.com/playlist?list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0))

* [A basic Implementation of NN in python with backprop](https://github.com/ahmedbesbes/Neural-Network-from-scratch/)

* [Stanford CS230 Deep Learning](https://www.youtube.com/playlist?list=PLoROMvodv4rOABXSygHTsbvUz4G_YQhOb) ([Slides and Syllabus](https://cs230.stanford.edu/syllabus/))

* [DEEP LEARNING DS-GA 1008 · SPRING 2020 · NYU CENTER FOR DATA SCIENCE](https://atcold.github.io/pytorch-Deep-Learning/)

* [Analyses of Deep Learning (STATS 385) Stanford University, Fall 2019](https://stats385.github.io/) ([Videos](https://www.researchgate.net/project/Theories-of-Deep-Learning))

* [IFT 6085: Theoretical principles for deep learning](http://mitliagkas.github.io/ift6085-dl-theory-class-2019/)

* [DL Brown Uni Assignments](http://cs.brown.edu/courses/cs1470/assignments.html)

* [EPFL EE-559 – DEEP LEARNING](https://fleuret.org/ee559/)

* Books for Deep Learning
    - [Deep Learning Book](https://www.deeplearningbook.org/) ([Chapter Summaries](https://github.com/dalmia/Deep-Learning-Book-Chapter-Summaries), [Videos](https://www.youtube.com/channel/UCF9O8Vj-FEbRDA5DcDGz-Pg/playlists), )
    - [Dive into Deep Learning Book](https://d2l.ai/), [Berkeley Site](https://courses.d2l.ai/berkeley-stat-157/index.html), [Pytorch Codes](https://github.com/dsgiitr/d2l-pytorch)
    - [Neural Network Design](http://hagan.okstate.edu/nnd.html)
    - [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) [GitHub](https://github.com/mnielsen/neural-networks-and-deep-learning) [solutions](https://github.com/reachtarunhere/nndl/blob/master/2016-11-22-ch1-sigmoid-2.md)
    

* [Neural Networks by Hugo Larochelle](https://www.youtube.com/playlist?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH) ([Course Website](http://info.usherbrooke.ca/hlarochelle/neural_networks/content.html))

* [CS294-158-SP20 Deep Unsupervised Learning Spring 2020](https://sites.google.com/view/berkeley-cs294-158-sp20/home)

* [MIT Introduction to Deep Learning](http://introtodeeplearning.com/) ([Videos](https://www.youtube.com/playlist?list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI))

* [Carnegie Mellon University Deep Learning](https://www.youtube.com/channel/UC8hYZGEkI2dDO8scT8C5UQA/playlists), [CMU DL website](https://deeplearning.cs.cmu.edu/)

### Field Specific Stuff: Computer Vision

* [Stanford CS 231 Conv Nets](http://cs231n.stanford.edu/2018/) ([github](https://cs231n.github.io/)) ([2016 by Andrej Karpathy](https://www.youtube.com/playlist?list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC), [2017 Version](https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv))

### Field Specific Stuff: NLP

* [Stanford CS224N: Natural Language Processing with Deep Learning | Winter 2019](http://web.stanford.edu/class/cs224n/) ([Vids](https://www.youtube.com/playlist?list=PLoROMvodv4rOhcuXMZkNm7j3fVwBBY42z))
* [Standford CS224U](https://web.stanford.edu/class/cs224u/), [Stanford ML book](https://web.stanford.edu/~jurafsky/slp3/)
* [Natural Language Processing | University of Michigan](https://www.youtube.com/playlist?list=PLLssT5z_DsK8BdawOVCCaTCO99Ya58ryR), 
* [Coursera Natural Language Processing | Dan Jurafsky, Christopher Manning](https://www.youtube.com/playlist?list=PLQiyVNMpDLKnZYBTUOlSI9mi9wAErFtFm) [Slides](https://web.stanford.edu/~jurafsky/NLPCourseraSlides.html)

### Field Specific Stuff: Reinforcement Learning

* Intros
    - [DRL FreeCodeCamp](https://medium.com/free-code-camp/an-introduction-to-reinforcement-learning-4339519de419)
    - [Deep Reinforcement Learning: Pong from Pixels](http://karpathy.github.io/2016/05/31/rl/)
    - [Deep Reinforcement Learning, Decision Making, and Control](https://vimeo.com/240428644)
    - [Neural Information Processing Systems Conference - NIPS 2016 Deep Reinforcement Learning Through Policy Optimization, Jan 23, 2017 at 11:13AM by Pieter Abbeel](https://channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Deep-Reinforcement-Learning-Through-Policy-Optimization)
    - [MIT 6.S091: Introduction to Deep Reinforcement Learning (Deep RL)](https://www.youtube.com/watch?v=zR11FLZ-O9M&feature=youtu.be)
    - [Deep Reinforcement Learning](http://videolectures.net/rldm2015_silver_reinforcement_learning/)
    
* [Deep RL Bootcamp 26-27 August 2017   |   Berkeley CA](https://sites.google.com/view/deep-rl-bootcamp/lectures)    

* [CS234: Reinforcement Learning | Winter 2019](https://www.youtube.com/playlist?list=PLoROMvodv4rOSOPzutgyCTapiGlY2Nd8u) ([http://web.stanford.edu/class/cs234/index.html](http://web.stanford.edu/class/cs234/index.html))

* [CS 285 at UC Berkeley Deep Reinforcement Learning](http://rail.eecs.berkeley.edu/deeprlcourse/)

* [A Free course in Deep Reinforcement Learning](https://simoninithomas.github.io/Deep_reinforcement_learning_Course/)

* UCL course 
    - [David-Silver-Reinforcement-learning](https://github.com/dalmia/David-Silver-Reinforcement-learning) ([davidsilver](https://www.davidsilver.uk/teaching/), [Videos](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ), [Easy21 Soln 1](https://github.com/luofan18/Reinforcement-learning-playing-easy21), [Easy21 Soln 2](https://github.com/kvfrans/Easy21-RL)), 
    - [Newer Videos - Slow and Easier - Advanced Deep Learning & Reinforcement Learning UCL](https://www.youtube.com/playlist?list=PLqYmG7hTraZDNJre23vqCGIVpfZ_K2RZs)

* [IIT-M RL](https://www.cse.iitm.ac.in/~ravi/courses/Reinforcement%20Learning.html), [YT](https://www.youtube.com/playlist?list=PLuWx2S0SyaDctJtVKHhmjYACmHZ3nX9ew)

* [dennybritz/reinforcement-learning](https://github.com/dennybritz/reinforcement-learning), [NervanaSystems/coach](https://github.com/NervanaSystems/coach), [rlcode/reinforcement-learning](https://github.com/rlcode/reinforcement-learning)

* [https://mpatacchiola.github.io/](https://mpatacchiola.github.io/blog/)

* [Richard S. Sutton and Andrew G. Barto Reinforcement Learning: An Introduction Book](http://incompleteideas.net/book/the-book-2nd.html)

* [Gtech RL](https://www.udacity.com/course/reinforcement-learning--ud600)

* [More Resources]
    - [Awesome RL](https://github.com/aikorea/awesome-rl)
    - [Spinning Up in Deep RL!](https://spinningup.openai.com/en/latest/)
    - [Deep Learning and Reinforcement Learning Summer School, Toronto 2018](http://videolectures.net/DLRLsummerschool2018_toronto/), [Notes](https://yobibyte.github.io/rlss17.html#rlss17)
    - [Using Keras and Deep Deterministic Policy Gradient to play TORCS](https://yanpanlau.github.io/2016/10/11/Torcs-Keras.html)
    - [Using Keras and Deep Q-Network to Play FlappyBird](https://yanpanlau.github.io/2016/07/10/FlappyBird-Keras.html)
    - [Reddit: free resources to learn Deep Reinforcement Learning](https://www.reddit.com/r/reinforcementlearning/comments/ci1bvy/opinions_on_free_resources_to_learn_deep/)
    

### Field Specific Stuff: Recommendation Systems

<Yet to Add>

### Other Random Areas/Resources
* [Curated: awesome-ml-courses](https://github.com/luspr/awesome-ml-courses)

* [deep-learning-drizzle](https://deep-learning-drizzle.github.io/) ([Github Page](https://github.com/kmario23/deep-learning-drizzle/blob/master/README.md))

* [HarvardX’s Fundamentals of Neuroscience XSeries](https://www.edx.org/xseries/harvardx-fundamentals-of-neuroscience)

* [https://www.edx.org/course/subject/data-science](https://www.edx.org/course/subject/data-science)

<!-- <<<<<< END INCLUDED FILE (markdown): SOURCE ml_ai_ds_resources/ml_learning.md -->

<!-- >>>>>> BEGIN INCLUDED FILE (markdown): SOURCE ml_ai_ds_resources/math_4_ml.md -->

## Mathematics resources

In general follow MIT OCW’s resources and search google/youtube for good explanations.

### Starter
* [The Matrix Calculus You Need For Deep Learning](https://explained.ai/matrix-calculus/index.html)

* [Essence of Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr) and [Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab), [3Blue1Brown Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi), [DE by 3Blue1Brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDNPOjrT6KVlfJuKtYTftqH6)

* [MIT Mathematics for Computer Science](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-042j-mathematics-for-computer-science-spring-2015/) [[Book](https://courses.csail.mit.edu/6.042/spring17/mcs.pdf)] [[Youtube](https://www.youtube.com/playlist?list=PLUl4u3cNGP60UlabZBeeqOuoLuj_KNphQ)] [[OCW](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-042j-mathematics-for-computer-science-spring-2015/)]

* [Maths for ML open book](https://mml-book.github.io/)

* [http://neuralnetworksanddeeplearning.com/chap1.html](http://neuralnetworksanddeeplearning.com/chap1.html)

### Basic: Probability and Statistics

* [Probability](https://www.youtube.com/playlist?list=PL2SOU6wwxB0uwwH80KTQ6ht66KWxbzTIo) (STAT 110 Harvard), [MIT 6–012 Introduction to Probability](https://ocw.mit.edu/resources/res-6-012-introduction-to-probability-spring-2018/) [Youtube Playlist](https://www.youtube.com/playlist?list=PLUl4u3cNGP60A3XMwZ5sep719_nh95qOe) or [EDX Harvard Probability](https://www.edx.org/course/introduction-to-probability) ([HarvardX: STAT110x EDX Introduction to Probability](https://courses.edx.org/courses/course-v1:HarvardX+STAT110x+1T2020/course/))

* [Probability Theory Lectures](https://www.youtube.com/playlist?list=PL9v9IXDsJkktefQzX39wC2YG07vw7DsQ_), Follow PROBABILITY THEORY: THE LOGIC OF SCIENCE

* [Intermediate Statistics by Larry Wasserman](https://www.youtube.com/playlist?list=PLJPW8OTey_OZk6K_9QLpguoPg_Ip3GkW_) ([Course page with Slides and assignments](http://www.stat.cmu.edu/~larry/=stat705/)| [All of statistics Book pdf](https://drive.google.com/open?id=1p0YcNL1zHbrjKtGaSgu9bMI88dF-_qWg))

* [Statistics for Applications by MIT](https://ocw.mit.edu/courses/mathematics/18-650-statistics-for-applications-fall-2016/) [[Youtube Playlist](https://www.youtube.com/playlist?list=PLUl4u3cNGP60uVBMaoNERc6knT_MgPKS0)], [CMU Intermediate Statistics](http://www.stat.cmu.edu/~larry/=stat705/) [[YouTube Playlist](https://www.youtube.com/playlist?list=PLJPW8OTey_OZk6K_9QLpguoPg_Ip3GkW_)], [MIT Probabilistic Systems Analysis and Applied Probability 6–041sc](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-041sc-probabilistic-systems-analysis-and-applied-probability-fall-2013/) ([Vids-youtube](https://www.youtube.com/playlist?list=PLUl4u3cNGP60A3XMwZ5sep719_nh95qOe))

* [Khan Academy https://www.khanacademy.org/math/statistics-probability](https://www.khanacademy.org/math/statistics-probability)

* [SticiGui](https://www.stat.berkeley.edu/~stark/SticiGui/index.htm)

* [onlinestatbook](http://onlinestatbook.com/2/index.html)

### Basic: Calculus

* [18.01x Single Variable Calculus](https://www.edx.org/xseries/mitx-18.01x-single-variable-calculus)

* [Calculus Basic Exhaustive coverage](https://www.youtube.com/playlist?list=PL0o_zxa4K1BWYThyV4T2Allw6zY0jEumv)

* [Introductory Calculus 1,2,3](https://www.youtube.com/user/amarchese22/playlists)

* [Differential Calculus](https://www.khanacademy.org/math/differential-calculus) and [Integral Calculus](https://www.khanacademy.org/math/integral-calculus) || [Calculus 1](https://www.khanacademy.org/math/calculus-1) and [Calculus 2](https://www.khanacademy.org/math/calculus-2) by Khan Academy

* [Coursera Calculus](https://www.coursera.org/learn/discrete-calculus), [Another](https://www.coursera.org/learn/introduction-to-calculus)

* [Calculus cheat sheet](http://tutorial.math.lamar.edu/pdf/Calculus_Cheat_Sheet_All.pdf)

### Basic: Linear Algebra

* [[TextBook+Practice Questions](http://math.mit.edu/~gs/linearalgebra/)][[Practice](https://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/assignments/) [Practice-2](http://web.mit.edu/18.06/www/old.shtml) ] [[OCW](https://ocw.mit.edu/courses/mathematics/18-06sc-linear-algebra-fall-2011/) [Youtube Playlist](https://www.youtube.com/playlist?list=PL221E2BBF13BECF6C)]

* [Matrix Algebra](https://www.coursera.org/learn/matrix-algebra-engineers) For Engineers

* [EDX-Linear Algebra — Foundations to Frontiers](https://courses.edx.org/courses/course-v1:UTAustinX+UT.5.05x+2T2017/course/)

* [MIT 18.065 Matrix Methods in Data Analysis, Signal Processing, and Machine Learning](https://www.youtube.com/playlist?list=PLUl4u3cNGP63oMNUHXqIUcrkS2PivhN3k) ([Course page](https://ocw.mit.edu/courses/mathematics/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/))

* [Linear Algebra Review and Reference: CS229](http://cs229.stanford.edu/section/cs229-linalg.pdf)

* [Linear Algebra step by step](https://www.amazon.com/Linear-Algebra-Step-Kuldeep-Singh/dp/0199654441) ([Solutions and slides](https://global.oup.com/booksites/content/9780199654444/), [Student solutions](https://global.oup.com/booksites/content/9780199654444/studentsolutions/)) or Sheldon Axler Linear algebra done right

* [Khan academy Linear Algebra](https://www.khanacademy.org/math/linear-algebra)

* [Linear algebra cheat sheet](https://www.souravsengupta.com/cds2016/lectures/Savov_Notes.pdf)

* [Computational Linear Algerbra](https://www.youtube.com/playlist?list=PLtmWHNX-gukIc92m1K0P6bIOnZb-mg0hY) by Fast.ai

* [Applications of Linear Algebra Part 2](https://www.edx.org/course/applications-of-linear-algebra-part-2)

* [Coding the Matrix - Linear Algebra through Computer Science Applications](https://www.youtube.com/playlist?list=PLEhMEyM9jSinRHXJgRCOLZUiu9847V2g0), [Book](https://codingthematrix.com/)

### Advanced: Differential Equations

* [18.03x Differential Equations](https://www.edx.org/xseries/mitx-18.03x-differential-equations)

* [Differential Eqn](https://www.coursera.org/learn/differential-equations-engineers) [DE MIT 18.03sc](https://www.youtube.com/playlist?list=PL64BDFBDA2AF24F7E), [DE MIT](https://www.youtube.com/playlist?list=PLUl4u3cNGP63oTpyxCMLKt_JmB0WtSZfG), [MIT Differential Equations](https://ocw.mit.edu/courses/mathematics/18-03sc-differential-equations-fall-2011/) [[Older version for more problems](https://ocw.mit.edu/courses/mathematics/18-03-differential-equations-spring-2010/)]

* [ODE Coursera](https://www.coursera.org/learn/ordinary-differential-equations)

* [Mit Refresher on ODE](https://ocw.mit.edu/courses/mechanical-engineering/2-087-engineering-math-differential-equations-and-linear-algebra-fall-2014/) [[Youtube Playlist](https://www.youtube.com/playlist?list=PLUl4u3cNGP63w3DE9izYp_3fpAdi0Wkga)] [[OCW Small Video Playlist on Same](https://ocw.mit.edu/resources/res-18-009-learn-differential-equations-up-close-with-gilbert-strang-and-cleve-moler-fall-2015/)] [[Differential Eqns and Linear Algebra book site with solutions](http://Differential Equations and Linear Algebra) [Amazon.in book link](https://www.amazon.in/Differential-Equations-Linear-Algebra-Gilbert/dp/0980232791)]

* [Introduction to Control System Design — A First Look](https://www.edx.org/course/introduction-to-control-system-design-a-first-look)

### Advanced: Optimisation

* [Optimization Methods in Business Analytics](https://courses.edx.org/courses/course-v1:MITx+15.053x+3T2016/course/) ([Link](https://www.edx.org/course/optimization-methods-for-business-analytics))

* [Optimisation Coursera](https://www.coursera.org/learn/discrete-optimization), [Opt 1](https://www.coursera.org/learn/basic-modeling), [Opt 2](https://www.coursera.org/learn/advanced-modeling), [Opt-3](https://www.coursera.org/learn/solving-algorithms-discrete-optimization), [Approximation Algos](https://www.coursera.org/learn/approximation-algorithms), [Ap-1](https://www.coursera.org/learn/approximation-algorithms-part-1), [Ap-2](https://www.coursera.org/learn/approximation-algorithms-part-2)

### Advanced: Probability, Bayesian and Statistical Inference

* [Probability by Elif Uysal](https://www.youtube.com/playlist?list=PLH72EKtvoHuZw0GuZ3YDd765H3S8nPJ2E), [EDX Computational Probability and Inference](https://courses.edx.org/courses/course-v1:MITx+6.008.1x+3T2016/course/) ([Link](https://www.edx.org/course/computational-probability-and-inference))

* [Bayesian Reasoning for Intelligent People](http://tuvalu.santafe.edu/~simon/br.pdf)

* Statistical Inference 2nd Ed by Casella and Berger

* [Bayesian Statistics Coursera](https://www.coursera.org/learn/bayesian-statistics), [Youtube Bayesian Statistics By Ben Lambert](https://www.youtube.com/playlist?list=PLwJRxp3blEvZ8AKMXOy0fc0cqT61GsKCG)

* Introduction to Probability Models by Sheldon M Ross

* [Coursera Stochastic processes](https://www.coursera.org/learn/stochasticprocesses)

* [Inferential Statistics with R by duke university](https://www.coursera.org/learn/inferential-statistics-intro)

* [Bayesian Statistics: From Concept to Data Analysis](https://www.coursera.org/learn/bayesian-statistics) and [Bayesian Statistics: Techniques and Models](https://www.coursera.org/learn/mcmc-bayesian-statistics)

* [Computer Age Statistical Inference](https://web.stanford.edu/~hastie/CASI/)

* [All of Nonparametric Statistics](http://www.stat.cmu.edu/~larry/all-of-nonpar/index.html)

* [BioStatistics 1](https://www.coursera.org/learn/biostatistics) and [2](https://www.coursera.org/learn/biostatistics-2), [clinical-trials](https://www.coursera.org/learn/clinical-trials)

### Advanced: Calculus, Multivariate Calculus and Matrix Calculus

* [Matrix Calculus for Deep Learning](https://explained.ai/matrix-calculus/index.html), 

* Vector Differentiation [1](https://www.youtube.com/watch?v=iWxY7VdcSH8), [2](https://www.youtube.com/watch?v=uoejt0FCWWA), [3](https://www.youtube.com/watch?v=i6fqfH5hx60), 

* Multivariable Calculus [1](https://www.youtube.com/playlist?list=PL8erL0pXF3JYm7VaTdKDaWc8Q3FuP8Sa7), [2-Derivatives](https://www.youtube.com/playlist?list=PL8erL0pXF3JZZTnqjginERYYEi1WpLE_V), [3-Integrals](https://www.youtube.com/playlist?list=PL8erL0pXF3JaJdUcmc_PeGV-vG5z87BkD), 

* [MIT Multivariable Calculus](https://www.youtube.com/playlist?list=PL4C4C8A7D06566F38) ([Homework vids](https://www.youtube.com/playlist?list=PLF07555F3CC669D01), [OCW page](https://ocw.mit.edu/courses/mathematics/18-02sc-multivariable-calculus-fall-2010/), [2007 Page](https://ocw.mit.edu/courses/mathematics/18-02-multivariable-calculus-fall-2007/)), 

* [Calculus with lots of problems by Organic Tutor YouTube](https://www.youtube.com/playlist?list=PL0o_zxa4K1BWYThyV4T2Allw6zY0jEumv), 

* [Khan Academy Multivariable calculus](https://www.youtube.com/playlist?list=PLSQl0a2vh4HC5feHa6Rc5c0wbRTx56nF7) ([Multivariable calculus](https://www.khanacademy.org/math/multivariable-calculus))

* [Vector Calculus](https://www.coursera.org/learn/vector-calculus-engineers)

* [Numerical Analysis Coursera](https://www.coursera.org/learn/intro-to-numerical-analysis), [Complex Analysis](https://www.coursera.org/learn/complex-analysis)

* [Calculus I](https://www.youtube.com/playlist?list=PL8erL0pXF3JYm7VaTdKDaWc8Q3FuP8Sa7), [II](https://www.youtube.com/playlist?list=PL8erL0pXF3JZZTnqjginERYYEi1WpLE_V), [III](https://www.youtube.com/playlist?list=PL8erL0pXF3JaJdUcmc_PeGV-vG5z87BkD): Cover calculus from a inter-disciplinary perspective.

* [Calculus Applied!](https://www.edx.org/course/calculus-applied)

### Other Topics

* [Information Theory for Intelligent People](http://tuvalu.santafe.edu/~simon/it.pdf), [Visual Information Theory](https://colah.github.io/posts/2015-09-Visual-Information/), [Information Theory](https://www.coursera.org/learn/information-theory), 

* [DSP](https://www.coursera.org/learn/dsp) and [ASP](https://www.coursera.org/learn/audio-signal-processing), [Discrete-Time Signal Processing](https://www.edx.org/course/discrete-time-signal-processing-4)

* [Discrete Maths](https://www.youtube.com/playlist?list=PLHXZ9OQGMqxersk8fUxiUMSIx0DBqsKZS)

* [Numerical Methods using Python](https://www.coursera.org/learn/computers-waves-simulations)

* [Computational Thinking for Modeling and Simulation](https://www.edx.org/course/computational-thinking-for-modeling-and-simulation)

* [Combinatorics](https://www.coursera.org/learn/analytic-combinatorics)

* [Coursera Game Theory — 1](https://www.coursera.org/learn/game-theory-1), [2](https://www.coursera.org/learn/game-theory-2), [Game thoery](https://www.coursera.org/learn/game-theory-introduction), [Games without Chance: Combinatorial Game Theory](https://www.coursera.org/learn/combinatorial-game-theory)

* [Differential Geometry](https://www.youtube.com/playlist?list=PLIljB45xT85DWUiFYYGqJVtfnkUFWkKtP)

* [Topology and geometry](https://www.youtube.com/playlist?list=PLTBqohhFNBE_09L0i-lf3fYXF5woAbrzJ)

* [MathTheBeautiful](https://www.youtube.com/channel/UCr22xikWUK2yUW4YxOKXclQ/playlists) (Calculus and Linear Algebra Lectures)

* [Math for Game devs](https://www.youtube.com/playlist?list=PLW3Zl3wyJwWOpdhYedlD-yCB7WQoHf-My)

* [Topics in Mathematics with Applications in Finance](https://ocw.mit.edu/courses/mathematics/18-s096-topics-in-mathematics-with-applications-in-finance-fall-2013/)

* [Engineering Maths Refresher MIT](https://ocw.mit.edu/courses/mathematics/18-085-computational-science-and-engineering-i-fall-2008/), [Engineering Maths — II](https://ocw.mit.edu/courses/mathematics/18-086-mathematical-methods-for-engineers-ii-spring-2006/)

* [Spectral Graph Theory](https://www.youtube.com/playlist?list=PLi4h0n4UP8d9VGPqO8vLQga9ZApO65TLW)

* [Principles of Economics with Calculus](https://www.edx.org/course/principles-of-economics-with-calculus)

### More Math Resources

* [Algebra, Topology, Differential Calculus, and Optimization Theory For Computer Science and Machine Learning —  Jean Gallier and Jocelyn Quaintance](https://www.cis.upenn.edu/~jean/math-deep.pdf)

* [Mathematical Monk](https://www.youtube.com/user/mathematicalmonk) ([Information Theory](https://www.youtube.com/playlist?list=PLE125425EC837021F), [ML](https://www.youtube.com/playlist?list=PLD0F06AA0D2E8FFBA), [Probability](https://www.youtube.com/playlist?list=PL17567A1A3F5DB5E4))

* [Coursera Maths For ML](https://www.coursera.org/specializations/mathematics-machine-learning) (YouTube- [Multivariate Calc](https://www.youtube.com/playlist?list=PLiiljHvN6z193BBzS0Ln8NnqQmzimTW23), [Linear Algebra](https://www.youtube.com/playlist?list=PLiiljHvN6z1_o1ztXTKWPrShrMrBLo5P3)), [Coursera Maths for Data Science](https://www.coursera.org/specializations/mathematics-for-data-science)

* [Deep Learning Book](https://www.deeplearningbook.org/) [[PDF available](https://github.com/janishar/mit-deep-learning-book-pdf)]Chapters 2,3,4 for the mathematics refresher

* Search “[Undergraduate Texts in Mathematics](https://www.amazon.com/s?k=Undergraduate+Texts+in+Mathematics&i=stripbooks-intl-ship&ref=nb_sb_noss) / ”, [More Coursera Probability and Statistics Courses](https://www.coursera.org/browse/data-science/probability-and-statistics), [More Math on Coursera](https://www.coursera.org/browse/math-and-logic), [More MIT math OCW](https://ocw.mit.edu/courses/audio-video-courses/#mathematics), [https://www.edx.org/course/subject/math](https://www.edx.org/course/subject/math)

* [Understanding Math](https://github.com/nbro/understanding-math), [Awesome Math](https://github.com/rossant/awesome-math)

* [Evan Chen’s the napkin project](https://web.evanchen.cc/napkin.html)
<!-- <<<<<< END INCLUDED FILE (markdown): SOURCE ml_ai_ds_resources/math_4_ml.md -->

<!-- >>>>>> BEGIN INCLUDED FILE (markdown): SOURCE ml_ai_ds_resources/staying_updated.md -->

## Staying Updated

At a minimum keep checking reddit and Stack Exchange sites. [StatQuest by Josh Stamer](https://www.youtube.com/user/joshstarmer) is highly recommended as a beginner’s watch.

### Video Resources
* Other
    - [VideoLectures.net](http://videolectures.net/Top/Computer_Science/Machine_Learning/#l=en), 
    - [SDS Podcast](https://www.superdatascience.com/podcast)
    - [MLSS 2014](https://www.youtube.com/playlist?list=PLZSO_6-bSqHQCIYxE3ycGLXHMjK3XV7Iz), [2019](https://sites.google.com/view/mlss-2019/lectures-and-tutorials?authuser=0)
* YT
    - [Arxiv Insights](https://www.youtube.com/channel/UCNIkB2IeJ-6AmZv7bQ1oBYg), 
    - [StatQuest by Josh Stamer](https://www.youtube.com/user/joshstarmer) 
    - [Yannic Kilcher](https://www.youtube.com/channel/UCZHmQk67mSJgfCCTn7xBfew/playlists), 
    - [Two Minute Papers](https://www.youtube.com/channel/UCbfYPyITQ-7l4upoX8nvctg), 
    - [Ahlad.K on Youtube](https://www.youtube.com/user/kumarahlad/playlists), 
    - [Robert Miles](https://www.youtube.com/channel/UCLB7AzTwc6VFZrBsO2ucBMg), 
    - [Computerphile](https://www.youtube.com/user/Computerphile)
    - [Lex Fridman](https://www.youtube.com/user/lexfridman/featured)
    


### Article Resources
* *Discussion Forum* 
    - [r/ML](https://www.reddit.com/r/MachineLearning/) , 
    - [r/learnML](https://www.reddit.com/r/learnmachinelearning/), 
    - [r/DeepLearning](https://www.reddit.com/r/deeplearning/), 
    - [r/DS](https://www.reddit.com/r/datascience/), 
    - [r/RL](https://www.reddit.com/r/reinforcementlearning/), 
    - [r/AskStats](https://www.reddit.com/r/AskStatistics/), 
    - [r/learnMaths](https://www.reddit.com/r/learnmath)
    - [https://discuss.analyticsvidhya.com/top](https://discuss.analyticsvidhya.com/top)
    - [https://www.fast.ai/tag/technical/](https://www.fast.ai/tag/technical/)
    

* StackExchange Sites
    - [Cross Validated](https://stats.stackexchange.com/?tab=month), 
    - [MathExchange Questions tagged [machine-learning]](https://math.stackexchange.com/questions/tagged/machine-learning?sort=votes&pageSize=15), 
    - [SE Questions tagged [machine-learning]](https://stackoverflow.com/questions/tagged/machine-learning?sort=votes&pageSize=15), 
    - [SE Questions tagged [deep-learning]](https://stackoverflow.com/questions/tagged/deep-learning), 
    - [AI SE](https://ai.stackexchange.com/), 
    - [DS SE](https://datascience.stackexchange.com/)
    
* NewLetters and feeds
    - [https://distill.pub/](https://distill.pub/) 
    - [https://openai.com/blog/](https://openai.com/blog/), 
    - [https://ai.googleblog.com/](https://ai.googleblog.com/), 
    - [Explained](https://explained.ai/)
    - [NTU Graph Deep Learning Lab](https://graphdeeplearning.github.io/)
    - [http://newsletter.ruder.io/](http://newsletter.ruder.io/), 
    - [https://ruder.io/](https://ruder.io/), 
    - [https://virgili0.github.io/Virgilio/](https://virgili0.github.io/Virgilio/), 
    - [https://bair.berkeley.edu/blog/](https://bair.berkeley.edu/blog/), 
    - [The Batch](https://www.deeplearning.ai/thebatch/)
    - [Data Science Weekly Newsletter Archive](https://www.datascienceweekly.org/newsletters)
    - [AI FB](https://ai.facebook.com/)
    - [AV DL](https://www.analyticsvidhya.com/blog/category/deep-learning/)
    - [http://www.wildml.com/](http://www.wildml.com/)
    - [https://www.topbots.com/](https://www.topbots.com/)
    - [https://rubikscode.net/category/ai/this-week-in-ai/](https://rubikscode.net/category/ai/this-week-in-ai/)
    - [TDS Learning Section](https://towardsdatascience.com/learn-on-towards-data-science-52245bc91451)
    - [Data science blogs](https://github.com/rushter/data-science-blogs)
    - [datasciencecentral](https://www.datasciencecentral.com/)
    - [fivethirtyeight](https://fivethirtyeight.com/)
    - [101.datascience](https://101.datascience.community/)
    
* Personal Blogs
    - [https://colah.github.io/](https://colah.github.io/),
    - [https://setosa.io/ev/](https://setosa.io/ev/)
    - [https://www.benfrederickson.com/blog/](https://www.benfrederickson.com/blog/), 
    - [http://nicolas-hug.com/blog/](http://nicolas-hug.com/blog/), 
    - [https://shaoanlu.wordpress.com/](https://shaoanlu.wordpress.com/)
    - [https://www.dlology.com/](https://www.dlology.com/)
    - [https://blog.otoro.net/](https://blog.otoro.net/)
    - [http://www.offconvex.org/](http://www.offconvex.org/)
    - [Karpathy Blog](https://karpathy.github.io/)
    - [computervisionblog](https://www.computervisionblog.com/)
    - [jalammar](http://jalammar.github.io/)
    - [the morning paper](https://blog.acolyer.org/)

* Projects
    - [SOTA by paperswithcode](https://paperswithcode.com/sota)
    - [https://www.kaggle.com/](https://www.kaggle.com/)
    - [MadeWithML](https://madewithml.com/)
    - [ML coding from scratch](https://github.com/python-engineer/MLfromscratch) and [ML coding from scratch Youtube](https://www.youtube.com/playlist?list=PLqnslRFeH2Upcrywf-u2etjdxxkL8nl7E) and [Another From scratch guide](https://github.com/pmuens/lab#implementations)
    
* **Recommender** 
    - [Arxiv Sanity](http://www.arxiv-sanity.com/) 
    - [https://s2-sanity.apps.allenai.org/](https://s2-sanity.apps.allenai.org/)
    - [https://scirate.com/arxiv/cs.AI](https://scirate.com/arxiv/cs.AI)
    - [https://scirate.com/arxiv/cs.CV](https://scirate.com/arxiv/cs.CV)
    - [https://scirate.com/arxiv/cs.LG](https://scirate.com/arxiv/cs.LG)

* More Lists
    - [Awesome Deep Learning](https://github.com/ChristosChristofidis/awesome-deep-learning), 
    - [Awesome ML](https://github.com/RatulGhosh/awesome-machine-learning), 
    - [Awesome RL](https://github.com/aikorea/awesome-rl), 
    - [TopDL](https://github.com/aymericdamien/TopDeepLearning), 
    - [Trending-Deep-Learning](https://github.com/mbadry1/Trending-Deep-Learning), 
    - [More Github Repos with cool learning](https://towardsdatascience.com/top-10-popular-github-repositories-to-learn-about-data-science-4acc7b99c44) 
    - [Statistical Learning](https://github.com/topics/statistical-learning), 
    - [Awesome AI](https://github.com/owainlewis/awesome-artificial-intelligence), 
    - [More papers](https://github.com/tirthajyoti/Papers-Literature-ML-DL-RL-AI), 
    - [Github ML list](https://github.com/collections/machine-learning), 
    - [ML resource list](https://github.com/ujjwalkarn/Machine-Learning-Tutorials), 
    - [Survey_of_Deep_Metric_Learning](https://github.com/kdhht2334/Survey_of_Deep_Metric_Learning)
    - [SOTA papers till 2018 in many DL tasks](https://www.eff.org/ai/metrics)
    - [A gallery of interesting Jupyter Notebooks](https://github.com/jupyter/jupyter/wiki/A-gallery-of-interesting-Jupyter-Notebooks)

<!-- <<<<<< END INCLUDED FILE (markdown): SOURCE ml_ai_ds_resources/staying_updated.md -->

<!-- >>>>>> BEGIN INCLUDED FILE (markdown): SOURCE ml_ai_ds_resources/papers.md -->

## Finding good Papers and Paper Reading Tips
All This should get you to good speed and provide you background to understand most papers. I use a 5 pass strategy to paper reading which I mention in the next section.

### Finding Good Papers

When you are new to this field, ensure that you take up those papers which are corner stone of this field. Not the latest ones, but rather the well established ones. These are usually well written, with less errors, and have long term value. Stick to ones published in big conferences like NeurIPS, ICML, ICLR and journals like JMLR. Also to guage value of the paper look at citation count and see if you can find a discussion in reddit or [Open review](https://openreview.net/).

### Finding Good papers

* [List of Top publications by Google Scholar](https://scholar.google.es/citations?view_op=top_venues&hl=en&vq=eng_artificialintelligence), [best_paper_awards](https://jeffhuang.com/best_paper_awards.html), [CS-Rankings of Confs and Institutes](http://csrankings.org/#/index?all)

* [ICML | 2019](https://icml.cc/Conferences/2019/Schedule) ([Full ICML Open Review Thread](https://openreview.net/group?id=ICML.cc)), [ICLR 2019](https://openreview.net/group?id=ICLR.cc/2019/Conference) ([Full Open Review thread of ICLR](https://openreview.net/group?id=ICLR.cc)), [NeurIPS](https://papers.nips.cc/) ([Open Review new thread](https://openreview.net/group?id=NeurIPS.cc), [old](https://openreview.net/group?id=NIPS.cc))

* [JMLR](http://www.jmlr.org/papers/) and [MLR](http://proceedings.mlr.press/), [JAIR](https://www.jair.org/index.php/jair)

* [Computer Vision Foundation](http://openaccess.thecvf.com/menu.py)

* [Reddit’s WAYR in r/ML](https://www.reddit.com/r/MachineLearning/comments/g4eavg/d_machine_learning_wayr_what_are_you_reading_week/)

* [AAAI](https://aaai.org/Library/conferences-library.php), [IJCAI 2019](https://www.ijcai.org/Proceedings/2019/) ([Past](https://www.ijcai.org/past_proceedings)), [KDD](https://www.kdd.org/conferences), [ECAI](https://www.eurai.org/library/ECAI_proceedings), [WWW](https://dl.acm.org/conference/www) ([2019](https://www2019.thewebconf.org/proceedings), [2018](https://www2018.thewebconf.org/proceedings/)), [ACM Recsys](https://recsys.acm.org/#)

* [EMNLP](https://www.aclweb.org/anthology/venues/emnlp/)

* [SOTA from papers with code](https://paperswithcode.com/sota), [Follow Papers with code site for good list of Papers](https://paperswithcode.com/)

* Other Notable Conferences: UAI, AISTATS, ECML, ACML, ICDM, COLT, [check this for more](http://www.guide2research.com/topconf/)

* [Wider CS Papers: papers-we-love](https://github.com/papers-we-love/papers-we-love)

* [Use Google Scholar to find more papers in your area](https://scholar.google.com/), also search “Advances in Field X” or “Recent Breakthroughs in X”, “Review X”, “Overview of X”. For each subject/book or field you are working on go to it’s discussion on reddit and look for other forums as well. Forum discussions are very good place to learn more about a topic.

### Reading a Paper

Follow a 5 pass strategy to read a paper. For most papers released post 2019 and with low citation counts -  Read the paper, extract the idea but if no code is present those numbers are most likely faulty.

* **First pass**: understanding what the paper is trying to convey, important results, basically getting the story. For this read the abstract, titles/heading and the conclusion

* **Second Pass**: Read the Intro and go over the experiments, identify the datasets and understand the evaluation metrics.

* **Third Pass**: Read the paper without focus on the mathematical contents, skim the equations, its ok if you don’t understand them. Read the english, and understand how they did it intuitively.

* **Fourth Pass**: The long pass, read the mathematics, dive into the equations.

* **Fifth Pass**: if a reference implementation exists then look at it.

[More info here](http://blizzard.cs.uwaterloo.ca/keshav/home/Papers/data/07/paper-reading.pdf) and [here](https://elfsternberg.com/2018/10/27/so-you-want-to-get-into-theoretical-computer-science/) and [here](https://web.stanford.edu/class/ee384m/Handouts/HowtoReadPaper.pdf) and [here](https://www.sciencemag.org/careers/2016/03/how-seriously-read-scientific-paper)

## Some Good Papers That I am reading
- Attention and Transformers
    - [NMT Paper by Bahdanau ICLR 2015](https://arxiv.org/abs/1409.0473)
    - [Show, Attend and Tell](https://arxiv.org/abs/1502.03044)
    - [A Structured Self-attentive Sentence Embedding](https://arxiv.org/abs/1703.03130)
    - [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
    - [BERT](https://arxiv.org/abs/1810.04805), [A Primer in BERTology: What we know about how BERT works](https://arxiv.org/abs/2002.12327)
    - Making Transformers Smaller and fixing other issues with them.
        - [Transformer-XL](https://arxiv.org/abs/1901.02860)
        - [Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451)
        - [Pay Less Attention with Lightweight and Dynamic Convolutions](https://arxiv.org/abs/1901.10430)
        - [Adaptive Attention Span in Transformers](https://arxiv.org/abs/1905.07799)
        - [Time-aware Large Kernel Convolutions](https://arxiv.org/abs/2002.03184)
    - Other Interesting Papers using Attention
        - [Stand-Alone Self-Attention in Vision Models](https://arxiv.org/abs/1906.05909)
    - [TRANSFORMERS FROM SCRATCH](http://www.peterbloem.nl/blog/transformers)
    - [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
    - [Transformers are Graph Neural Networks](https://graphdeeplearning.github.io/post/transformers-are-gnns/)
    - [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
    
- Graph Neural Networks
    - [A Comprehensive Survey on Graph Neural Networks 2019](https://arxiv.org/abs/1901.00596)
    - [Machine Learning on Graphs: A Model and Comprehensive Taxonomy 2020](https://arxiv.org/abs/2005.03675)
    - [GCMC]()
    - [GraphSage]()
    - Scaling Graph networks
        - [PyTorch-BigGraph: A Large-scale Graph Embedding System](https://arxiv.org/abs/1903.12287)
        - [SIGN: Scalable Inception Graph Neural Networks](https://arxiv.org/abs/2004.11198)


- Deep Reinforcement Learning
    - [A Brief Survey of Deep Reinforcement Learning](https://arxiv.org/abs/1708.05866)
    - [Deep Reinforcement Learning: An Overview](https://arxiv.org/abs/1701.07274)
    - [Key Papers in Deep R](https://spinningup.openai.com/en/latest/spinningup/keypapers.html)
    - [deep-reinforcement-learning-papers](https://github.com/muupan/deep-reinforcement-learning-papers)

- Deep Reinforcement Learning For SQL optimisations
    - [Learning to Optimize Join Queries With Deep Reinforcement Learning](https://arxiv.org/abs/1808.03196), [Blog by Berkeley Lab](https://rise.cs.berkeley.edu/blog/sql-query-optimization-meets-deep-reinforcement-learning/)
    - [Towards a hands-free query optimizer through deep learning](https://blog.acolyer.org/2019/01/18/towards-a-hands-free-query-optimizer-through-deep-learning/), [Deep Reinforcement Learning for Join Order Enumeration](https://arxiv.org/abs/1803.00055)
    - [Neo: A Learned Query Optimizer](https://arxiv.org/abs/1904.03711)
    
- Explainable AI
    - [Explainable Deep Learning: A Field Guide for the Uninitiated](https://arxiv.org/abs/2004.14545)
    
- Adversarial 
    - [On Adaptive Attacks to Adversarial Example Defenses](https://arxiv.org/abs/2002.08347)
    - [Adversarial Attacks and Defences: A Survey 2018](https://arxiv.org/abs/1810.00069)
    - [A Complete List of All (arXiv) Adversarial Example Papers](https://nicholas.carlini.com/writing/2019/all-adversarial-example-papers.html)
    

<Yet to Add>

<!-- <<<<<< END INCLUDED FILE (markdown): SOURCE ml_ai_ds_resources/papers.md -->

<!-- >>>>>> BEGIN INCLUDED FILE (markdown): SOURCE ml_ai_ds_resources/engineering_aspects.md -->

## Engineering Aspects

### Before Starting a Project
- https://towardsdatascience.com/20-questions-to-ask-prior-to-starting-data-analysis-6ec11d6a504b
- https://towardsdatascience.com/how-to-construct-valuable-data-science-projects-in-the-real-world-203a4f520d54

### DL
- [Pytorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)

### Experiment Tracking
- [Trains](https://github.com/allegroai/trains)
- [MLFlow]()
- [DVC: Open-source Version Control System for Machine Learning Projects](https://dvc.org/)

### Deployments

### Code Examples for Learning
- [Numpy-ML](https://github.com/ddbourgin/numpy-ml)
- [dennybritz/reinforcement-learning](https://github.com/dennybritz/reinforcement-learning)


### EDA and plotting, visualizations
- [Lolviz: Visualize DataStructures](https://github.com/parrt/lolviz)
- [pandas-profiling](https://github.com/pandas-profiling/pandas-profiling)
- [Missing-No](https://github.com/ResidentMario/missingno)
- Plotting Libs: [Altair](https://github.com/altair-viz/altair), [Plotly]()
- [Good theme for Matplotlib](https://towardsdatascience.com/a-new-plot-theme-for-matplotlib-gadfly-2cffc745ff84)

### Modelling
- [Scikit Categorical encoders](https://github.com/scikit-learn-contrib/category_encoders)

### NLP
- [https://allennlp.org/](https://allennlp.org/)
- [Spacy](https://spacy.io/)

### RL
- [vitchyr/rlkit](https://github.com/vitchyr/rlkit)

### CV
- [Keract Activation Heatmap](https://github.com/philipperemy/keract)
- [image-super-resolution](https://github.com/idealo/image-super-resolution)

### Recommendation Systems

### Explainable AI
- [Lime](https://github.com/marcotcr/lime)
- [Keract Activation Heatmap](https://github.com/philipperemy/keract)
- [DTree Viz](https://github.com/parrt/dtreeviz)
- [SHAP](https://github.com/slundberg/shap)
- [CNN Feature Maps](https://github.com/lewis-morris/mapextrackt), [Grad-CAM](https://arxiv.org/abs/1610.02391), [Grad-CAM++](https://arxiv.org/abs/1710.11063)
- [ELI5](https://eli5.readthedocs.io/en/latest/)

### Performance
- [Ray](https://docs.ray.io/en/latest/)
- [Dask](https://docs.dask.org/en/latest/)
- [Modin Dataframe](https://github.com/modin-project/modin), [CuDF](https://github.com/rapidsai/cudf)
- [Vaex: Out-of-Core DataFrames](https://vaex.readthedocs.io/en/latest/)
- [Intel SDC: Numba for Pandas](https://intelpython.github.io/sdc-doc/latest/getting_started.html)
- [SnakeViz](https://jiffyclub.github.io/snakeviz/#snakeviz), [Vprof](https://github.com/nvdv/vprof), [PyInstrument](https://github.com/joerick/pyinstrument)

### Jupyter & Notebooks
- [jupytemplate](https://github.com/xtreamsrl/jupytemplate)
- [NB-Dev](https://github.com/fastai/nbdev)
- [Dashboard with Jupyter Notebook](https://github.com/voila-dashboards/voila/)
- [Debugging in Jupyter](https://github.com/jupyter-xeus/xeus-python) ([Article](https://towardsdatascience.com/jupyter-is-now-a-full-fledged-ide-c99218d33095))

### Recipes
- [Vectorizing Conditions in Numpy & Pandas](https://www.youtube.com/watch?v=nxWginnBklU)
<!-- <<<<<< END INCLUDED FILE (markdown): SOURCE ml_ai_ds_resources/engineering_aspects.md -->

<!-- >>>>>> BEGIN INCLUDED FILE (markdown): SOURCE ml_ai_ds_resources/random_topics.md -->

## Random Machine Learning Topics/Articles

### General ML

* [Probability Calibration](https://scikit-learn.org/stable/modules/calibration.html) [[Ref-1](https://towardsdatascience.com/probability-calibration-for-boosted-trees-24cbd0f0ccae), [Ref-2](http://danielnee.com/2014/10/calibrating-classifier-probabilties/), [Ref-3](http://fastml.com/classifier-calibration-with-platts-scaling-and-isotonic-regression/)]
* [Bayesian Optimisation](https://towardsdatascience.com/a-conceptual-explanation-of-bayesian-model-based-hyperparameter-optimization-for-machine-learning-b8172278050f) [[Paper](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf)]
* Reducing Model size in deep learning by knowledge distillation/Teacher Student training
* [Metric Learning](http://contrib.scikit-learn.org/metric-learn/introduction.html) ([A Survey](https://www.mdpi.com/2073-8994/11/9/1066/htm), [pytorch-metric-learning](https://github.com/KevinMusgrave/pytorch-metric-learning))
* KD Tree for KNN: [Part 1](https://www.youtube.com/watch?v=u4M5rRYwRHs), [Part 2](https://www.youtube.com/watch?v=XqXSGSKc8NU), [Part 3](https://www.youtube.com/watch?v=DlPrTGbO19E&t=2s), [Part 4](https://www.youtube.com/watch?v=SD6bO8eu5RM) ([Full Playlist](https://www.youtube.com/playlist?list=PLguYJK7ydFE7R7KqRRVXw23kOrn6jiwqi))


### Mathematical
* [Simpson’s Paradox](https://towardsdatascience.com/simpsons-paradox-how-to-prove-two-opposite-arguments-using-one-dataset-1c9c917f5ff9)
* [How to lie with Statistics Summary](https://towardsdatascience.com/lessons-from-how-to-lie-with-statistics-57060c0d2f19)
* [PCA with Maths](https://www.youtube.com/playlist?list=PLBv09BD7ez_5_yapAg86Od6JeeypkS4YM) ([Another PCA by same guy](https://www.youtube.com/playlist?list=PLBv09BD7ez_4InDh85LM_43Bsw0cFDHdN)), [Clustering](https://www.youtube.com/watch?v=b9gPL6NvsnA&list=PLBv09BD7ez_6lYVoZ1RzVcOPIT5Lfjo0Y)
* [A Paper explaining CNN with Maths](https://pdfs.semanticscholar.org/450c/a19932fcef1ca6d0442cbf52fec38fb9d1e5.pdf)


### Applied ML
* [Instagram Recommender](https://ai.facebook.com/blog/powered-by-ai-instagrams-explore-recommender-system/)


### Interview Questions
- [Interview Questions](https://towardsdatascience.com/40-statistics-interview-problems-and-answers-for-data-scientists-6971a02b7eee), 
- [2](https://towardsdatascience.com/over-100-data-scientist-interview-questions-and-answers-c5a66186769a), 
- [3](https://towardsdatascience.com/giving-some-tips-for-data-science-interviews-after-interviewing-60-candidates-at-expedia-395fff7e073b), 
- [4](https://towardsdatascience.com/googles-data-science-interview-brain-teasers-7f3c1dc4ea7f), 
- [5](https://towardsdatascience.com/how-i-became-an-ai-consultant-interview-questions-answers-689ba03a2620), 
- [6](https://towardsdatascience.com/40-statistics-interview-problems-and-answers-for-data-scientists-6971a02b7eee)
- [7](https://towardsdatascience.com/facebooks-data-science-interview-practice-problems-46c7263709bf)
- [8](https://towardsdatascience.com/the-amazon-data-scientist-interview-93ba7195e4c9)
- [9](https://towardsdatascience.com/over-100-data-scientist-interview-questions-and-answers-c5a66186769a)
- [10](https://towardsdatascience.com/50-deep-learning-interview-questions-part-1-2-8bbc8a00ec61)

* [Anyone have experience combining ML models with expert rules](https://www.reddit.com/r/MachineLearning/comments/gbdz4n/discussion_anyone_have_experience_combining_ml/)

### Explainability: 
- [deep-dive-into-catboost-functionalities-for-model-interpretation](https://towardsdatascience.com/deep-dive-into-catboost-functionalities-for-model-interpretation-7cdef669aeed)
- [Partial Dependence Plot for Feature Importance and Exploration](https://towardsdatascience.com/looking-beyond-feature-importance-37d2807aaaa7)

### Deep Learning
* [a-hackers-guide-to-efficiently-train-deep-learning-models](https://medium.com/@ahmedbesbes/a-hackers-guide-to-efficiently-train-deep-learning-models-b2cccbd1bc0a)
* [Variational_AutoEncoder](https://www.youtube.com/watch?v=w8F7_rQZxXk&list=PLdxQ7SoCLQANizknbIiHzL_hYjEaI-wUe) ([Arxiv Insight Video](https://www.youtube.com/watch?v=9zKuYvjFFS8&vl=en), [Keras AutoEncoders](https://blog.keras.io/building-autoencoders-in-keras.html))
* [Augmenting Neural Networks with Constraints Optimization](https://towardsdatascience.com/augmenting-neural-networks-with-constraints-optimization-ac747408432f)
* [Do We Really Need Model Compression?](http://mitchgordon.me/machine/learning/2020/01/13/do-we-really-need-model-compression.html)
* [Andrej Karpathy blog: A Recipe for Training Neural Networks](http://karpathy.github.io/2019/04/25/recipe/)

### NLP
- [Synonyms](https://blog.kensho.com/how-to-build-a-smart-synonyms-model-1d525971a4ee) ([Dataset](https://www.kaggle.com/kenshoresearch/kensho-derived-wikimedia-data), [Kernel](https://www.kaggle.com/kenshoresearch/kdwd-aliases-and-disambiguation))

### RL
- [Training AI Without Writing A Reward Function, with Reward Modelling](https://www.youtube.com/watch?v=PYylPRX6z4Q)

### Misc
* [Technical Writing by Google](https://developers.google.com/tech-writing)
* [DS Resume](https://towardsdatascience.com/data-science-resume-mistakes-to-avoid-2867460659ac)

## Datasets 
* [CV datasets](https://lionbridge.ai/datasets/20-best-image-datasets-for-computer-vision/)
* [3D shape data for Geometric Deep Learning - shapenet](https://shapenet.org/)

## Generic References

- [**How to Learn Machine Learning, The Self-Starter Way**](https://elitedatascience.com/learn-machine-learning)
- [**Data Science** Resources](https://wrdrd.github.io/docs/consulting/data-science#mathematical-notation)
- [nbro/understanding-math](https://github.com/nbro/understanding-math)
- [**List of logic symbols - Wikipedia**](https://en.wikipedia.org/wiki/List_of_logic_symbols)
- [**List of mathematical symbols - Wikipedia**](https://en.wikipedia.org/wiki/List_of_mathematical_symbols)
- [**Mathematics for Data Science**](https://towardsdatascience.com/mathematics-for-data-science-e53939ee8306)
- [how_do_you_read_mathheavy_machine_learning](https://www.reddit.com/r/MachineLearning/comments/6rj9r4/d_how_do_you_read_mathheavy_machine_learning/)
- [math_undergrad_how_do_you_efficiently_study](https://www.reddit.com/r/learnmath/comments/7k72lb/math_undergrad_how_do_you_efficiently_study/)
- [Confession as an AI researcher; seeking advice](https://www.reddit.com/r/MachineLearning/comments/73n9pm/d_confession_as_an_ai_researcher_seeking_advice/)
- [How Do You Read Large Numbers Of Academic Papers Without Going Crazy?](https://www.reddit.com/r/MachineLearning/comments/de5wam/d_how_do_you_read_large_numbers_of_academic/)
- [How to deal with my research not being acknowledged ?](https://www.reddit.com/r/MachineLearning/comments/dh0aak/d_how_to_deal_with_my_research_not_being/)
- [a_clear_roadmap_for_mldl](https://www.reddit.com/r/learnmachinelearning/comments/cxrpjz/a_clear_roadmap_for_mldl/)
- [instagram.com/machinelearning](https://www.instagram.com/machinelearning/)
- [Advice and Suggestions needed on my Roadmap to Machine Learning/AI Pro](https://www.reddit.com/r/learnmachinelearning/comments/90ohlm/advice_and_suggestions_needed_on_my_roadmap_to/)
- [What math classes are relevant for machine learning?](https://www.reddit.com/r/MachineLearning/comments/f4i3v6/discussion_what_math_classes_are_relevant_for/)
- [What are some habits of highly effective ML researchers?](https://www.reddit.com/r/MachineLearning/comments/f4oxuj/discussion_what_are_some_habits_of_highly/)
- [**An Opinionated Guide to ML Research**](http://joschu.net/blog/opinionated-guide-ml-research.html)
- [A Compilation of Useful, Free, Online Math Resources](https://www.reddit.com/r/math/comments/2mkmk0/a_compilation_of_useful_free_online_math_resources/)
- [**llSourcell/learn_math_fast**](https://github.com/llSourcell/learn_math_fast)
- [**ShuaiW/data-science-question-answer**](https://github.com/ShuaiW/data-science-question-answer)

- [MIT HowToReadAScientificPaper](https://be.mit.edu/sites/default/files/documents/HowToReadAScientificPaper.pdf)
- [Tips For First Paper?](https://www.reddit.com/r/MachineLearning/comments/g7nemh/d_tips_for_first_paper/)
- [Advice for Beginners in ML and Data Science](https://www.reddit.com/r/learnmachinelearning/comments/gc5z6b/advice_for_beginners_in_ml_and_data_science/)
- [Speech and Language Processing (3rd ed. draft)](https://web.stanford.edu/~jurafsky/slp3/)
- [The Big Bad NLP Database](https://datasets.quantumstat.com/)
- [Some Tutorials](https://github.com/madewithml/lessons)
- [Some Good Papers](https://github.com/GokuMohandas/casual-digressions)
- [https://nlpprogress.com/](https://nlpprogress.com/)
- [Guide to ML Research](http://joschu.net/blog/opinionated-guide-ml-research.html)
<!-- <<<<<< END INCLUDED FILE (markdown): SOURCE ml_ai_ds_resources/random_topics.md -->

<!-- >>>>>> BEGIN INCLUDED FILE (markdown): SOURCE ml_ai_ds_resources/cs.md -->

## CS Basics

* [https://towardsdatascience.com/beginners-learning-path-for-machine-learning-5a7fb90f751a](https://towardsdatascience.com/beginners-learning-path-for-machine-learning-5a7fb90f751a)

* [Learn Python](https://www.learnpython.org/), [How to Think Like a Computer Scientist: Learning with Python 3](http://openbookproject.net/thinkcs/python/english3e/#), [Automate the Boring Stuff with Python](https://automatetheboringstuff.com/#toc)

* [https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-0001-introduction-to-computer-science-and-programming-in-python-fall-2016/index.htm](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-0001-introduction-to-computer-science-and-programming-in-python-fall-2016/index.htm)

* [Course on Data Structures](https://www.coursera.org/specializations/data-structures-algorithms), [Algorithms Youtube Stanford](https://www.youtube.com/playlist?list=PLXFMmlk03Dt7Q0xr1PIAriY5623cKiH7V), [MIT 6.006 Intro to algorithms](https://www.youtube.com/playlist?list=PLUl4u3cNGP61Oq3tWYp6V_F-5jb5L2iHb), [MIT algorithms 2015](https://www.youtube.com/playlist?list=PLkToMFwOtNHiJtcBu0piSLKnLVGOF9vaV), [IO-efficient algos](https://www.coursera.org/learn/io-efficient-algorithms), [MIT Introduction to Computational Thinking and Data Science](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-0002-introduction-to-computational-thinking-and-data-science-fall-2016/), [Algorithms in Python Implementations](https://github.com/TheAlgorithms/Python)

* [Some Knowledge of System design](https://github.com/donnemartin/system-design-primer) (not mandatory)

* [Using Linux Unix](https://www.coursera.org/learn/unix), Pip, apt-get, cat, grep, history, ls, cd and other tools/commands
<!-- <<<<<< END INCLUDED FILE (markdown): SOURCE ml_ai_ds_resources/cs.md -->








<!-- <<<<<< END GENERATED FILE (include): SOURCE ml_ai_ds_resources/readme_template.md -->
