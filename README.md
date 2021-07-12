# Memory Efficient Online Meta Learning

This is implementation of [Memory Efficient Online Meta Learning](http://proceedings.mlr.press/v139/acar21b/acar21b.pdf).

### Requirements

Please install the required packages. The code is compiled with Python 3.8 dependencies in a virtual environment via

```pip install -r requirements.txt```

### Instructions

An example code for S-MNIST 1000 tasks setting is given. Run

```python main.py```

The code,

- Constructs a online meta learning dataset,

- Trains all methods,

- Plots the CTM and LTM on tensorboard.

You can change the configurations and save the losses for each round to run get hedge results by running

```python hedge.py```
### Citation

```
@InProceedings{pmlr-v139-acar21b,
  title = 	 {Memory Efficient Online Meta Learning},
  author =       {Acar, Durmus Alp Emre and Zhu, Ruizhao and Saligrama, Venkatesh},
  booktitle = 	 {Proceedings of the 38th International Conference on Machine Learning},
  pages = 	 {32--42},
  year = 	 {2021},
  editor = 	 {Meila, Marina and Zhang, Tong},
  volume = 	 {139},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {18--24 Jul},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v139/acar21b/acar21b.pdf},
  url = 	 {http://proceedings.mlr.press/v139/acar21b.html},
  abstract = 	 {We propose a novel algorithm for online meta learning where task instances are sequentially revealed with limited supervision and a learner is expected to meta learn them in each round, so as to allow the learner to customize a task-specific model rapidly with little task-level supervision. A fundamental concern arising in online meta-learning is the scalability of memory as more tasks are viewed over time. Heretofore, prior works have allowed for perfect recall leading to linear increase in memory with time. Different from prior works, in our method, prior task instances are allowed to be deleted. We propose to leverage prior task instances by means of a fixed-size state-vector, which is updated sequentially. Our theoretical analysis demonstrates that our proposed memory efficient online learning (MOML) method suffers sub-linear regret with convex loss functions and sub-linear local regret for nonconvex losses. On benchmark datasets we show that our method can outperform prior works even though they allow for perfect recall.}
}

```