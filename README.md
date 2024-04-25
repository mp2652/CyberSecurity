This project is an improvement of an existing Traffic Type Classification, including protocol classification, application classification, and traffic type classification.
In the past, port numbers were commonly used for protocol classification, but this method is limited because many services do not have assigned port numbers.
More recently, Deep Packet Inspection (DPI) has been used, which analyzes packet payload patterns and keywords instead of port numbers. However, DPI struggles with encrypted traffic, such as HTTPS.
To address these challenges, machine learning (ML) models based on flow-level statistical features have been developed. 
These models extract features, evaluate them, and use classification algorithms like Decision Trees and KNN. However, optimizing each step individually does not necessarily lead to the best overall solution.
Given the success of deep learning (DL) in other domains, DL techniques are being applied to traffic classification.
DL offers an end-to-end strategy and can automatically learn discriminative features from raw input. However, DL models are often seen as black boxes, without the possibility to understand the predictions.
We have a lack of interpretability.
Another point is that ML and DL models typically require analyzing entire flows or large portions of data, which can be time-consuming and may violate requirements for online classification.
In fact, ML  input Data can be Statistical Features which is statistical analysis of packet flows, including metrics such as packet count, size distribution, and flow rate.
DL input Data is  raw byte data that treat each byte as a feature. Sometimes, DL methods may incorporate additional information alongside raw byte data to enrich input representations.
To address these issues, Self-Attentive Mechanism (SAM) is proposed. SAM resolves previous problems by using fine-grained packet-level data for classification. 
Since packets arrive gradually, SAM can meet the requirements for online classification. 
Additionally, SAM provides more interpretability by assigning weights to different parts of the input, helping to understand why certain features are discriminated against by the model.
The SAM has four main components: Embedding, self-attention, 1D-CNN, and classifier.
The·Embedding is used for learning semantic representations and enriches the representation of packet-level input in SAM.
The Self-attention Assigns attentive weights to different parts of the input, indicating their importance. 
 The Classifier is the final component that applies the softmax function to output the predicted class probabilities.
The implementation of SAM faces two main technical challenges which are the Packet-level input selection. In fact, due to the abundance of information, not all input data are equally important, requiring careful selection. 
The second challenge is the Adaptation of self-attention mechanism. Because SAM is based on a mechanism designed for language processing, it necessitates adjustments for traffic classification.

For the input data, SAM utilizes the initial 𝐿 bytes of an IP packet as its input vector.In fact, in  the TCP/IP model, packets are structured with layers including the link layer, Internet layer, transport layer, and application layer. 
So, the input  vector encompasses information from the Internet layer, the transport layer, and a restricted segment of the packet payload to ensure user privacy. 
Some data fields, such as local network-specific IP addresses, are omitted as they do not contribute to the classification process.
This data refinement facilitates the fulfillment of online classification requirements.
Also, the final output of SAM is  weights where close to 0 signify unimportant information, while those close to 1 denote significant details. 
This approach offers several interpretability advantages compared to DL methods lacking self-attention mechanisms, allowing for the identification of discriminative features across various classification tasks.
In evaluating SAM's performance, it is compared with other ML and DL-based schemes using statistical features, such as Decision Tree, k-Nearest Neighbors, Random Forest, 1D-CNN, and CNN+RNN, through a 10-fold cross-validation.
Metrics including accuracy, precision, recall, and F1-score are utilized for assessment.
Results demonstrate SAM's superior stability and performance across various classification tasks, particularly in protocol and application classification, where it outperforms alternative methods consistently.
Despite a slight performance drop in traffic type classification, SAM remains highly competitive, exhibiting better stability and improved accuracy compared to other approaches. 
Additionally, analyses of byte input size, training time, GPU usage, and classification performance under different batch sizes reveal trade-offs between resource consumption and model performance, guiding optimal parameter selection for efficient classification.
Moreover, in our evaluation we aimed to identify the most crucial features for various classification tasks, particularly focusing on which bytes receive the most attention from SAM. 
During the testing phase, we extracted the attentive weights from SAM's self-attention layer and visualized the results. 
It's interesting to note that the significant features differ depending on the classification task.
Also, we observed that SAM converges faster compared to other methods, with significant reduction in loss after just the first epoch of training.
Despite training for only five epochs, SAM's efficiency is evident as its accuracy remains consistent or even improves slightly with continued training.
In this project, we tried some improvements in order to improve the SAM results.We start by adding a residual connection as the last layer of the SAM model. 
A residual connection, also known as a skip connection, is a neural network architecture element that involves adding the input of a certain layer to the output of a later layer.
This helps mitigate the vanishing gradient problem and facilitates the training of deeper networks by allowing the model to learn residual information directly.
In fact, The vanishing gradient problem is a challenge in deep learning where gradients (derivatives of the loss with respect to model parameters) become extremely small during backpropagation, hindering the training of deep neural networks. 
This can lead to ineffective updates of weights in early layers, causing those layers to learn slowly or not at all. Thus, residual connections and other techniques are used to address this issue.
In our improvement, we included a 1D convolutional layer with a kernel size of 1, which acts as a linear transformation. The input to this layer is the output from previous layers in the model, and the output is then added element-wise to the original input of the layer.
This helps create a shortcut connection that enables the network to learn residual information, aiding in the training of deeper models. The activation function ReLU is applied after the convolution.
We obtained an improvement with the residual connection of on average 0.4%. So in proportion to the good result of the SAM that is 94%, it is an interessant improvement.
Secondly, we tried to add an LSTM layer to the SAM. This  is a type of recurrent neural network (RNN) layer designed to capture and remember long-term dependencies in sequential data. 
It consists of memory cells with input, output, and forget gates. These gates regulate the flow of information within the cell, allowing it to selectively remember or forget information over time.
LSTMs are particularly effective in handling vanishing gradient problems associated with traditional RNNs, making them suitable for tasks involving sequential data such as natural language processing and time series prediction.
Here, in our context the input is  a sequence of bytes,  the LSTM layer can effectively capture dependencies and patterns over different positions in the sequence. However, in our experiment, the addition of the LSTM layer gave worse results. 
We suppose that the number of the input byte (50 in this case) is too short. The length of the input sequence has to be longer in general.
Then, another improvement that is possible is to use a Multi Head Attention. It applies the self attention mechanism using the SAM model several times on the same input and then concatenates the results together for the next inner layer.
However, this structure also introduces computation complexity so we do not experiment with this structure.
An additional improvement that we apply is batch normalization. The problem which is avoided as a result of the batch norm is the internal covariate shift problem, that is changing in the distribution of the input to each layer during the training.
As the parameters of the earlier layers in the network are updated, the distribution of the inputs to subsequent layers may shift. The shift can slow down the training process, as the network has to continually adapt to the changing input distribution, which might lead to suboptimal learning.
Normalizing the input to each layer during training, helps in maintaining a stable distribution of inputs to the layers, which accelerates training and can lead to better generalization performance.
With the batch normalization, we reach an accuracy of almost 96%, an improvement of about 1.5%, so it is a very significant improvement.

There are also some little changes that we made like to replace the Adam Optimizer by the Stochastic gradient descent or RMSprop. We also increased and decreased the kernel filter size, changed the dropout value and tried spatial Dropout but without finding better results. 
However, the replacement of Adam optimizer by AdamW has increased the accuracy to more than 95%. 
In fact, Adam and AdamW are both optimization algorithms used in training neural networks, and AdamW is essentially an extension or modification of the original Adam optimizer.
The primary difference lies in the way that AdamW addresses the issue of weight decay more explicitly. Weight decay is a regularization term that penalizes large weights to prevent overfitting.
In the context of optimization, it is equivalent to adding a penalty term to the loss function based on the magnitude of the weights.
The explicit handling of weight decay in AdamW led to better results  in this task.

In conclusion, the research makes a significant contribution to the domain of online traffic classification by introducing a self-attentive deep learning method that not only advances classification accuracy but also illuminates the problems of interpretability and fulfills the requirements of Online classification. 
For the improvements, we could see that using Batch normalization, AdamW optimizer and a Residual connection have improved the results of the SAM model while others changes in some parameters like the use of Spatial Dropout, increase of the Dropout value, the kernel size, or the use of others optimizers didn’t really improve the SAM result.
This approach bridges the gap between complex neural network representations and actionable insights for network management and decision-making, paving the way for increased efficiency in classifying online traffic in computer networking contexts.

Bibliography:
●	https://www.sciencedirect.com/science/article/abs/pii/S1389128621002930
●	https://github.com/xgr19/SAM-for-Traffic-Classification
●	https://chat.openai.com/
●	https://developer.nvidia.com/cuda-zone
●	https://pytorch.org/
