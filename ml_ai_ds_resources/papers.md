
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
        - [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150)
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

