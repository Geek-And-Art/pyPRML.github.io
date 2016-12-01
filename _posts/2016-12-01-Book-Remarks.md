---
layout: post
comments: true
title:  "Book Remarks"
excerpt: "PRML book remarks"
date:   2016-12-01
mathjax: true
---

# Chapter 1.2.5

After long preparations of math, basic machine learning concepts, this chapter begins to talk about the **probabilistic perspective**, which is the basis for further Bayesian treatments in following chapter 1.2.6.

Here, the author uses the example of polynomial curve fitting in terms of error minimization as introduction. This is another common situation, which is called *discriminant model* in later chapters.

Pay attension, from here, it's the most important boundary between discriminant model and probabilistic model. In discriminant model, our purpose is to estimate the function $$f: X \rightarrow Y$$ directly. But in probabilistic model, our goal is no longer this function map estimation any more. Instead, we'd like to know the distribution of $$P(Y\vert X)$$. By knowing this distribution, we can know new data's prediction $$\hat{y}$$ easily:

$$\hat{y} = \underset{y}{\operatorname{argmax}} P(Y=y \vert X)$$

Also we can notice the distribution of $$P(Y\vert X)$$ gives us more information about the uncertinty of our prediction, comparing the discriminant model can only provide point estimation. 

By understanding the probabilistic perspective, we can discuss this chapter's details.

By using the model assumtion that the noisy error is Gaussian, we construct our model format as

$$p(t\vert x, w, \beta) = \mathcal{N}(y(x, w), \beta^{-1})$$

. Here it uses trick that, by setting $$e = t - y(x,w)$$, we get $$p(e\vert x, w, \beta) = \mathcal{N}(0, \beta^{-1})$$. Then, we need to estimate this distribution.

Assume the noisy distribution as Gaussian is not trivial. It's related to the considerations from **Information Theory**. From its entropy discussion, even the most efficient information transmission, its process contains the uncertainty loss. And the distribution of the most efficient transimission is Gaussian. That is to say, by considering the natural information loss, its ideal simplest assumption can be treated as Gaussian.



