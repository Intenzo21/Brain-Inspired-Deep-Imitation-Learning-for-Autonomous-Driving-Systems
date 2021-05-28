# Deep-Imitation-Learning-for-Autonomous-Driving-Vehicles
Autonomous driving vehicles have drawn a great deal of interests from both academia (e.g. Oxford, MIT) and industry (e.g. Google, Tesla). However, we find that it is very difficult to directly achieve fully autonomous driving (SAE Level 5) due to generalised knowledge. To deal with the problem, deep imitation learning is a promising solution which learns knowledge from the demonstration of human. In this project, we worked on how to use deep imitation learning to achieve vehicle dynamic control (e.g. steering angle, speed). We used a dataset and simulator provided by Udacity (https://github.com/udacity/self-driving-car-sim) and the real-world comma.ai dataset. 
 
You can download the comma.ai driving dataset [here](https://archive.org/download/comma-dataset).

The instructions described are related to the Windows OS. They are pretty straightforward given that the user has already downloaded the necessary project files and meets the dependencies and requirements mentioned in Chapter B. One can obtain the program scripts from our GitHub repository.

## Data Preprocessing
 The comma.ai dataset is split into 3 types of raw recordings prior
 to being preprocessed. This process is carried out manually and no script is developed for the
 purpose. We present the actions required to preprocess these 3 types of recording data. In contrast,
 the Udacity dataset is preprocessed on the go while training the models via a batcher function thus
 no additional steps are required.
 
 Step 1: Enter a command prompt terminal window;
 Step 2: Navigate to the directory of the comma.ai dataset model training scripts by using the cd command;
 Alternatively, one could go to the model training program files directory viaWindows
 File Explorer, use the Ctrl +L keyboard shortcut to go to the address bar and enter
 cmd, as shown in Figure A.1, to launch a command prompt from the current file
 location;
