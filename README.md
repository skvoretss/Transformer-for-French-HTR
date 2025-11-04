# Transformer-for-French-HTR
Fine-Tuning a Transformer Model for Handwritten French Text Recognition

## Description
This project focuses on fine-tuning the pre-trained transformer model ```microsoft/trocr-small-handwritten``` for the task of recognizing handwritten text in French. We utilize the RIMES 2011 dataset, which contains a diverse range of handwritten text samples, allowing us to improve recognition quality and adapt the model to the specifics of French handwriting.

### Main goals
- Model Adaptation: Fine-tune the pre-trained model ```microsoft/trocr-small-handwritten``` to enhance the accuracy of handwritten French text recognition.
- Performance Evaluation: Conduct an evaluation of the model's performance on the test set from RIMES 2011 dataset and compare the results with existing solutions.

### Dataset
The project uses the [RIMES 2011 dataset](https://storage.teklia.com/public/rimes2011/RIMES-2011-Lines.zip), which consists of:

- Handwritten texts in French, both word and line levels.
- Annotations that allow for mapping images to corresponding texts.

This dataset has a default split for train, test and validation sets, therefore these sets are used to have a comprehensive and valid comparison with existing results.
### Metrics
Classic metrics [WER](https://en.wikipedia.org/wiki/Word_error_rate) and [CER](https://huggingface.co/spaces/evaluate-metric/cer) were used not only because they provide deep analyze of the results, but also because they are commonly used by authors for both printed and handwritten text recognition. 

### Technologies Used

- Python: The primary programming language for implementing the project.
- Transformers: A library from Hugging Face used for working with transformer models.
- PyTorch: A machine learning framework used for training the model.
- Metrics: A library [evaluate](https://github.com/huggingface/evaluate) from Hugging Face to calculate WER and CER.

## Results
#### Example 1

#### Example 2

#### Conclusion
After a series of experiments, the best results are WER = 14.4\% and CER = 5.8\% after ~8 hours of training, using hyperparameters from example 2. Comparable results with WER =  14.9% and CER = 6.7% were obtained after only ~6 hours of training with hyperparameters from example 1. Other sets of hyperparameters showed poor results compared to experiments, mentioned above, and to the results of other authors on the chosen dataset.
