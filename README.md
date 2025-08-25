# SearchingForCauses

This is the repository for the implementations of the paper [*Searching for actual causes: approximate algorithms with adjustable precision*](https://arxiv.org/abs/2507.07857).

## About this project

Causality has gained popularity in recent years. It has helped improve the performance, reliability, and interpretability of machine learning models. However, recent literature on explainable artificial intelligence (XAI) has been subject to criticism. The classical XAI and causality literature focuses on understanding which factors contribute to which consequences. While such knowledge is valuable for researchers and engineers, it is not what non-expert users expect as explanations. Instead, these latter often await facts that cause the target consequences, i.e., actual causes. Formalizing this notion is still an open problem. However,  identifying actual causes is reportedly an NP-complete problem, and there are too few practical solutions to approximate formal definitions. We propose a method derived from the beam search algorithm to identify actual causes with a polynomial complexity and an adjustable level of precision and exhaustiveness. Our experiments indicate that our algorithm (1) identifies causes for different categories of systems that are not handled by existing approaches (i.e., non-boolean, black-box, and stochastic systems), (2) can be adjusted to gain more precision and exhaustiveness with more computation time.

## How to use

To use this code, you can clone the repository. You can setup an environment for jupyter notbook using:

```
    source setup.sh
```

The code for the method can be found in the external [actualcauses](https://pypi.org/project/actualcauses/) package. The code for the experiments is in the [experiment.py](src/experiments.py) file. You can find examples of usage in [Examples.ipynb](src/Examples.ipynb) or how to generate the result figures in [Experiments.ipynb](src/Experiments.ipynb). 

To reproduce the experiment from the paper, run:

```
    python src/experiments.py
```

## Citation

If you use this code for an academic work, please use the following citation:

Reyd, S., Diaconescu, A., & Dessalles, J. L. (2025). Searching for actual causes: Approximate algorithms with adjustable precision. arXiv preprint arXiv:2507.07857.

@misc{reyd2025searchingactualcauses,
      title={Searching for actual causes: Approximate algorithms with adjustable precision}, 
      author={Samuel Reyd and Ada Diaconescu and Jean-Louis Dessalles},
      year={2025},
      eprint={2507.07857},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2507.07857}, 
}
