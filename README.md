# Who's the (Multi-)Fairest of Them All: Rethinking Interpolation-Based Data Augmentation Through the Lens of Multicalibration

Existing work on algorithmic fairness and data augmentation measures fairness improvements with binary metrics that do not capture model uncertainty. At the same time, methods of improving metrics that do consider uncertainty involve decreasing the amount of initial training data to create a holdout set for post-processing. A promising direction in data augmentation for fairness is fair mixup, a variant of interpolation-based data augmentation that accommodates the incorporation of any penalty metric between two computationally identifiable groups in its training loss function. Thus far, however, fair mixup has been implemented and tested on relatively large marginalized groups, with performance reporting and training-time penalty metrics based on the binary metrics of demographic parity and equalized odds. Furthermore, fair mixup-based training has only been done on one marginalized group and its complement at a time. This paper uses multicalibration (MC), which reflects probabilistic predictions, as an alternative lens to examine data augmentation for classification fairness for marginalized groups and their intersections of various sizes. Using two structured data classification problems drawn from the \folktables package, we stress-test four versions of fair mixup with up to 81 marginalized groups, evaluating multicalibration violations and balanced accuracy tradeoffs against nine other baselines, including post-processing, balanced batching, and regular mixup. We compare performance and fairness across five differently sized sets of demographic groups and across data from 40 combinations of the ten most populous US states and the four most recent data collection years. We surprisingly find that while Fair Mixup regimes worsen both baseline balanced accuracy and multicalibration violations in nearly all experimental settings, regular mixup substantially improves both metrics, especially when calibrating on very small groups. Furthermore, combining regular mixup with post-processing for multicalibration enforcement yields even stronger improvements than either augmentation or post-processing alone.

**Who's the (Multi-)Fairest of Them All: Rethinking Interpolation-Based Data Augmentation Through the Lens of Multicalibration** 
[paper](https://arxiv.org/abs/2412.10575?fbclid=IwZXh0bgNhZW0CMTEAAR1tU7r1LabFOLvQc6-pBRIZui3Jk0JUINQn1KWQ62hceazMB7vkUuSHd9E_aem_tL6m5iekgiREwqOONGmaSw)

## Prerequisites
- Python 3.7 
- PyTorch 1.3.1
- aif360
- sklearn

## Implementation
We implement several experiments based on the `folktables` package. The various `.sh` files run our experiments in sequence.
