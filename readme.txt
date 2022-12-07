The training and testing data is generated using the QuaDRiGa channel model:
  1. Download the source of QuaDRiGa channel model at https://quadriga-channel-model.de/
  2. Run 'generate_training_and_testing_channel.m' to generate the training dataset and the testing data. 

Training Phase
  1. Run 'wmmse_deepunfolding_final_v1_reproduce.py' to learn the step-sizes and store the learned step-sizes.
  2. The training loss is given below:
  ![image]()
  
  
  
Testing Phase
  1. Use the step-sizes to evaluate the generalization of the proposed NN model in the testing phase.
  


