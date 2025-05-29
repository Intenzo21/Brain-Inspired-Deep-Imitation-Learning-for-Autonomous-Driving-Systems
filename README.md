# Brain-Inspired Deep Imitation Learning for Autonomous Driving Systems

[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.simpa.2021.100165-blue?style=plastic)](https://doi.org/10.1016/j.simpa.2021.100165)

## Project Abstract
Autonomous driving has attracted great attention from both academics and industries. To realise autonomous driving, Deep Imitation Learning (DIL) is treated as one of the most promising solutions, because it improves autonomous driving systems by automatically learning a complex mapping from human driving data, compared to manually designing the driving policy. However, existing DIL methods cannot generalise well across domains, that is, a network trained on the data of source domain gives rise to poor generalisation on the data of target domain. We propose a novel brain-inspired deep imitation method that builds on the evidence from human brain functions, to improve the generalisation ability of DNN so that autonomous driving systems can perform well in various scenarios. Specifically, humans have a strong generalisation ability which is beneficial from the structural and functional asymmetry of the two sides of the brain. We design dual Neural Circuit Policy (NCP) architectures in DNN based on the asymmetry of human neural networks. Experimental results demonstrate that our brain-inspired method outperforms existing methods regarding generalisation when dealing with unseen data.

***Note: The instructions described below are related to Windows OS. They are pretty straightforward given that the user has already downloaded the necessary project files and meets the dependencies and requirements mentioned in `maintenance_manual.pdf`.***

## Data Preprocessing
[The comma.ai dataset](https://archive.org/download/comma-dataset) is split into 3 types of raw recordings prior
to being preprocessed. This process is carried out manually and no script is developed for the
purpose. We present the actions required to preprocess these 3 types of recording data. In contrast,
the Udacity dataset is preprocessed on the go while training the models via a batcher function thus
no additional steps are required.

- Step 1. Enter a command prompt terminal window;

- Step 2: Navigate to the directory of the comma.ai dataset model training scripts by using the `cd` command; 
   - Alternatively, one could go to the model training program files directory via Windows File Explorer, use the `Ctrl+L` keyboard shortcut to go to the address bar and enter `cmd`, as shown in Figure A.1, to launch a command prompt from the current file location;

- Step 3: Run the data preprocessing script by entering the following command line arguments
into the terminal:

    ```bash
    python preprocess_data.py rec_type [batch_size]
    ```
  - The `rec_type` should be set to the value of either `sunny`, `cloudy` or `night` since we have 3 types of video files. On the other hand, the optional `batch_size` argument above can be replaced with the desired numerical size value of each batch. It is set to 64 by default.

You can skip the above steps and directly download the 64-batched, preprocessed dataset [here](https://archive.org/details/imitation_learning_files).

**For additional details about the project (how to reproduce, dependencies etc.), please refer to the `maintenance_manual.pdf` and `user_manual.pdf` files.**
