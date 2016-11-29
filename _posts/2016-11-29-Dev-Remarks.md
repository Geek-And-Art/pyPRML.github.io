---
layout: post
comments: true
title:  "Development Remarks"
excerpt: "Dev notes to trace experience progress"
date:   2016-11-29
mathjax: true
---

# Prelude

In the process of developing all kinds of machine learning algorithms from scratch, I found it's really tough to make progress with my own expectation. You want the code to be generative, but it just sucks in the mess data structure. You want the code to express some quick feedback, but it just sucks with performance issues, etc,. So, I think it may be valuable to trace the problems I met, the solutions I found, and the ways I finally use to apply to my own project.

For current Python ecosystem, **SciPy**, the [`scikit-learn`](http://scikit-learn.org/stable/) is one wonderful project to be referenced. Lots of problems that beyond our own domain knowledge can be found and researched there.

# Naive Bayes

When I first create the NB implementation with one method to synthesize data, it's a trap to consider both routines at the same time. My first implementation has to be overthrown as there're too many trails of sythesized method, especially the interface design of NB's train() method.

From the succinct perspective view, the training process only requires input data `X` and output/target data `y`. Thus a clear interface:

```python
def train(X, y):
    """
    Input
    -----
    - X: 2-D numpy array with shape (num_train, num_attrs)
    - y: 1-D numpy array with shape (num_train, )
    """
    pass
```

is all we need. You don't need to give the `X` or `y` more generative data format, like dictionary, which is the preprocessing work of users or other layers. At this basic layer, we only need to choose the data format based on performance quality.

For Python, it's much like MatLab, it's more good at matrix computation instead of other dynamic allocation like `dict`, which is due to the fixed continuous data allocation in memeory. When you convert contents of one text into `Bag-of-Words` format, numpy array format may give your **1000x** performance improvement comparing the use of `dict`.
