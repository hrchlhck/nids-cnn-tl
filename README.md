# Network-based Intrusion Detection Through Image-based CNN and Transfer Learning

## Authors
Pedro Horchulhack, Eduardo Kugler Viegas, Altair Olivo Santin, João André Simioni

## Abstract
Machine learning (ML) techniques for network intrusion detection is still limited in production environments despite promising results reported in the literature. Network traffic behavior exhibits considerable variability and evolves over time, requiring periodic model updates. This paper proposes a new approach to intrusion detection modeling based on CNN and transfer learning to reduce updating overhead. Its implementation is twofold. First, CNN is implemented using flow-based feature expansion derived from neural flattened hyperdimensional space. This expanded space representation contributes to a longer model lifetime and maintains system accuracy over time. Second, the required training data and computational cost are significantly reduced by performing periodic model updates based on a transfer learning approach. Experiments on a novel dataset with over 2.6 TB of data and one year of real-world network traffic demonstrate the feasibility of the proposal. Our proposal improves the average F1 score by up to 0.19 when no model updates are performed. While improving the system’s accuracy, model updates impose only 42.8% of the computational cost.

## Citation
```bibtex
@inproceedings{horchulhack2024,
  author={Horchulhack, Pedro and Viegas, Eduardo K. and Santin, Altair O. and Simioni, João A.},
  booktitle={2024 International Wireless Communications and Mobile Computing (IWCMC)}, 
  title={Network-based Intrusion Detection Through Image-based CNN and Transfer Learning}, 
  year={2024},
  volume={},
  number={},
  pages={386-391},
  keywords={Wireless communication;Accuracy;Computational modeling;Transfer learning;Training data;Network intrusion detection;Telecommunication traffic;Intrusion Detection;CNN;Transfer Learning},
  doi={10.1109/IWCMC61514.2024.10592364}
}
```
