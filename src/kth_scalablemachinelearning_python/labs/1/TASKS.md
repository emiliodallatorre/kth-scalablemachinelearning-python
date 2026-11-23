# Lab assignment 1: Building and evaluating convolutional neural networks

You can use Lab 2 notebook Part B to implement this assignment.

Use a practical ML workflow to build image classifiers with AlexNet and ResNet, investigate how design choices affect learning, and compare accuracy, robustness, and computational cost.

## Dataset

Use CIFAR-10, which contains labelled colour images from ten classes. The dataset is available from the [official website](https://www.cs.toronto.edu/~kriz/cifar.html) and can be loaded with torchvision.

## Notebook goals

- [ ] Prepare the image data for reproducible training and evaluation
- [ ] Train and compare AlexNet and ResNet
- [ ] Investigate model capacity and regularization
- [ ] Evaluate performance on clean, corrupted, and adversarial inputs
- [ ] Support a recommendation using measured accuracy, robustness, and cost

## Task 1: Study and prepare the data

- [ ] Study CIFAR-10 before building the models
- [ ] Create training and validation partitions from the provided training data and reserve the test data for final evaluation
- [ ] Prepare reusable data loading and preprocessing steps

Start from and extend the following questions:

- [ ] What does an example represent, and how are the classes distributed?
- [ ] Which preprocessing steps are needed for the two architectures?
- [ ] How is the dataset preprocessed and split?
- [ ] How is randomness controlled in the experiment?

## Task 2: Implement and compare the models

- [ ] Implement AlexNet and ResNet using PyTorch, starting with the recommended CIFAR-10 adaptations described in Part A9
- [ ] Explain the changes from the original architectures and justify any further modifications
- [ ] Train both models using a reproducible procedure and compare them under a clearly stated compute budget

Describe and justify:

- [ ] The layers, feature-map dimensions, and parameter counts
- [ ] The input preparation and classification head
- [ ] The convolution operations and the roles they play in the visual task
- [ ] The optimization and checkpoint-selection procedure
- [ ] The role of residual connections in ResNet

You may explore other variants after establishing the two baseline models. Explain any changes so that the reader can distinguish an original architecture from an experimental variant.

## Task 3: Investigate capacity and regularization

- [ ] Explore how model capacity and regularization affect learning and generalization
- [ ] Compare a small set of different widths or depths and investigate L2 regularization and dropout
- [ ] Compare a condition without dropout with two dropout settings of your choice

For each comparison:

- [ ] State what is changed and what is held fixed
- [ ] Show training and validation behaviour
- [ ] Explain evidence of underfitting, overfitting, or optimization difficulty
- [ ] Discuss the effect on accuracy and computational cost

Choose settings that are feasible for your available computational resource. Support your interpretation with controlled comparisons rather than a large, unexplained parameter search.

## Task 4: Evaluate robustness

- [ ] Investigate whether training with data augmentation improves robustness
- [ ] Choose transformations appropriate for the images and construct a reproducible corrupted evaluation set, for example using rotation, geometric distortion, or noise

Compare the original and augmented training conditions in terms of:

- [ ] Performance on clean images
- [ ] Performance on corrupted images
- [ ] The types of errors introduced by different transformations
- [ ] The additional training or evaluation cost

Use the same evaluation images and corruption settings for the models being compared. Explain your choices and discuss whether stronger transformations can make labels ambiguous.

## Task 5: Analyse predictions and adversarial examples

- [ ] Evaluate the models quantitatively and inspect representative successful and unsuccessful predictions
- [ ] Use suitable metrics and visualizations to explain which classes or image characteristics are difficult

Your analysis should include:

- [ ] A numerical comparison and training diagnostics
- [ ] Examples of correct and incorrect predictions
- [ ] An adversarial example generated against a trained model
- [ ] The perturbation constraint, attack method, and observed effect

Distinguish sensitivity to random corruption from a deliberately optimized perturbation. If your attempted attack does not change a prediction within the chosen constraint, report that result and explain the limitations of the attempt.

## Task 6: Benchmark and discuss the tradeoffs

- [ ] Compare the configurations under the compute budget you defined
- [ ] Measure training cost and inference performance, and relate them to accuracy and robustness

Discuss:

- [ ] Which model would you choose under your available computational budget, and why?
- [ ] Which changes improve accuracy or robustness at an acceptable cost?
- [ ] Which conclusions are sensitive to the training duration, random seed, or hardware?
- [ ] What further experiment would provide the most useful evidence?

Support your conclusions with measurements from your implementation.

## Optional challenges

Attempt either or both after the main tasks. Choose and justify a feasible method and evaluation protocol. Include your implementation, evidence, and discussion in Part B.

### Challenge 1: Model compression

Explore a method to make a trained model smaller or faster while retaining acceptable predictive performance. Define an accuracy tolerance and efficiency measure, and compare with the uncompressed reference.

- [ ] Describe the method and any additional training or fine-tuning
- [ ] Measure accuracy and relevant costs consistently, distinguishing parameter or file-size reduction from actual inference speed
- [ ] Discuss the tradeoff and account for the additional compression compute

### Challenge 2: Generalization

Evaluate a selected model on an unseen corruption family or a different, suitably labelled image domain. Keep these conditions separate from model and augmentation selection.

- [ ] Explain the shift, label compatibility, and preprocessing
- [ ] Use held-out evaluation and paired image identities when applicable
- [ ] Discuss transfer, failure cases, severity, and limits of generalization

## Deliverables

Submit one executed notebook containing:

- [ ] The complete implementation, configuration, and reproduction instructions
- [ ] The experimental results, figures, and numerical comparisons
- [ ] The dataset, architecture, documentation, and code references used
- [ ] A report containing discussions and conclusions for Tasks 1–6 and challenges attempted

The notebook should be executed so it shows the final results used in the report. If the Colab limit is restricted to run all tasks, you may submit separate notebooks addressing different tasks, but you need to discuss all tasks in a single report.