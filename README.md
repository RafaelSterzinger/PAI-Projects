# Probabilistic Artificial Intelligence
## Gaussian Process Regression
Implementation of a Gaussian Process and applying it on an inference regression problem based on drinking water pollution.
<br/><br/>
## Bayesian Neural Network
Implementation of a Bayesian Neural network based on the theory shown in [Variational Inference for Neural Networks](https://www.cs.toronto.edu/~graves/nips_2011.pdf) and applying it on the [Rotated MNIST](https://github.com/ChaitanyaBaweja/RotNIST) dataset.
<br/><br/>
## Bayesian Optimization
Implementation of a custom [Bayesian optimization algorithm to an hyperparameter tuning problem](https://papers.nips.cc/paper/2012/file/05311655a15b75fab86956663e1819cd-Paper.pdf). In particular, the goal was to perform global optimization of a black-box function subject to a constraint. 
<br/><br/>
## Actor Critic Reinforcement Learning
<img align="right" height="120" src="https://www.nestorsag.com/assets/static/lunarlander.74b2aba.8e40388ecbca12ef48106778e710b728.gif"></img>
The task was to implement an algorithm that, by practicing on a simulator, learns a control policy for the _Lunar Lander_ problem. The method suggested is a variant of policy gradient with two additional features, namely (1) [Rewards-to-go](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#implementing-reward-to-go-policy-gradient), and (2) [Generalized Advantage Estimatation](https://arxiv.org/pdf/1506.02438.pdf), both aiming at decreasing the variance of the policy gradient estimates while keeping them unbiased.
