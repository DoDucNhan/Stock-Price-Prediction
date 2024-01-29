# Stock Price Prediction

Stock market prediction is a common task that every stock expert wants to be good at so that the predicted price is as close to the actual price as possible. The successful prediction of a stock's future price could yield significant profit. The most commonly used network for stock price problems is the **Recurrent Neural Network**. This network can effectively extract patterns in the sequence data and take information from previous inputs to influence the current input and output, which makes it a powerful tool for time series data and stock price prediction. In this experiment, as the goal is to predict stock price and gain deeper insights into the Recurrent Neural Network, this network architecture will be self-implemented using Pytorch library and trained on the stock price data of Google. Furthermore, this network is compared to **Long Short-Term Memory** and **Gated Recurrent Unit** networks to investigate the performance differences. The evaluation is performed on three models with the same architecture complexity (except its core unit) and training configuration, to determine the best model based on the lowest loss on the validation set.

## Introduction
### Deep Learning
_Deep learning (DL)_ can be considered a subset of _machine learning_. While machine learning uses straightforward concepts, deep learning works with a broader algorithm called artificial neural networks. This algorithm is inspired and designed by the structure and function of the human brain to imitate how humans think and learn. The learning process of DL can be _supervised, semi-supervised_ or _unsupervised_. DL has various architectures, such as _Deep Neural Networks, Convolutional Neural Networks, Recurrent Neural Networks_ and _Transformers_. These architectures have been applied to computer vision, speech recognition, natural language processing, medical image analysis and produced stunning results which can surpass human expert performance in some cases. 

### Stock Market
A _stock market_ is a public market that can be bought and sold shares for publicly listed companies. The stocks, also known as equities, represent ownership in the company. The stock exchange is the mediator that allows the buying and selling of shares. Recently, stock price prediction has been a critical area of research and is one of the top applications of machine learning. Stock market prediction is the process of trying to determine the future value of company stock or other financial instruments traded on an exchange. The successful prediction of a stock's future price could yield significant profit. Many factors might be responsible for determining a particular stock's price, such as the market trend, supply and demand ratio, global economy, public sentiments, sensitive financial information, earning declaration, historical price and many more. These factors explain the challenge of accurate prediction. However, with the advantage of modern technologies like data mining and deep learning, which can help analyze big data and develop an accurate prediction model that avoids some human errors and stock market prediction has since moved into the technological realm. Since the stock price is the _time series data_ - a sequence of data points in chronological order used by businesses to analyze past data and make future predictions, the most prominent technique for this task involves the use of _Recurrent Neural Networks_ since this network has the capacity to remember previous input through time. 

## Methods
### Recurrent Neural Network
A recurrent neural network (RNN) is a class of artificial neural networks. This architecture is robust because of its node connections, which create a cycle, allowing output from some nodes to affect subsequent input to the same nodes. This feature also allows it to exhibit temporal dynamic behaviour that can work with sequential or time series data. In the training process, RNNs utilize training data to learn, like feedforward and convolutional neural networks (CNNs). However, this network is distinguished by having memory since it takes information from previous inputs to influence the current input and output. While traditional deep neural networks assume that inputs and outputs are independent of each other, the output of RNN depends on the prior elements within the sequence. This process makes them applicable to ordinal or temporal tasks such as connected handwriting recognition, speech recognition, language translation, natural language processing (NLP), and image captioning.

![image](https://github.com/DoDucNhan/Stock-Price-Prediction/assets/44297479/9556b052-e750-429e-a2cb-873ba0a3490a)
*RNN architecture*

![image](https://github.com/DoDucNhan/Stock-Price-Prediction/assets/44297479/5f1e7cc9-3b85-4c9b-9b1c-3248123e2658)
*RNN block architecture*

### Long Short-Term Memory
One of the appeals of RNN is the idea that it is able to connect previous information to the present task. However, when the prediction needs prior information from further away from the beginning of the sequence, the gap between the relevant information and the point where it is needed becomes very large. Unfortunately, as that gap grows, RNNs become unable to learn to connect the information. This problem is also known as the vanishing/exploding gradient. The vanishing and exploding gradient phenomena happen when a multiplicative gradient that can be exponentially decreasing/increasing with respect to the number of layers causes the challenge of capturing long-term dependencies. Long Short-Term Memory (LSTM) is an advanced RNN designed to avoid the long-term dependency problem. The primary point making LSTM differ from RNN is the LSTM unit. 

![image](https://github.com/DoDucNhan/Stock-Price-Prediction/assets/44297479/404b5cbd-e244-47ac-a16d-deaf91d91eeb)

### Gated Recurrent Unit
A Gated Recurrent Unit (GRU) is a variant of the RNN architecture and uses gating mechanisms to control and manage the flow of information between cells in the neural network. GRUs were introduced only in 2014 by Cho, et al. and can be considered a relatively new archi tecture, especially compared to the widely-adopted LSTM. This RNN variant is similar to the LSTMs as it is able to effectively retain long-term dependencies in sequential data and also works to address the short-term memory problem of RNN models. The workflow of the GRU is the same as the RNN and LSTM. However, instead of using a cell state to regulate information, it uses hidden states, and instead of four gates like LSTM, it has two gates, a reset gate and an update gate. Similar to the gates within LSTMs, the reset and update gates control how much and which information to retain. GRU is relatively new, but it is on track to outshine LSTM due to its superior speed while achieving similar accuracy and effectiveness. If the dataset is small, GRU is a better option than LSTM.

![image](https://github.com/DoDucNhan/Stock-Price-Prediction/assets/44297479/9e0f1a3c-acd9-4869-8309-383dad01031f)

## Experiment Analysis
In this experiment, RNN, GRU and LSTM are compared to show the efficiency of variations of RNN in stock price prediction. In addition, three models are shared the same architecture and training configuration to determine how well each variation performs in the stock price dataset from https://www.alphavantage.co. The evaluation metric for model performance is based on Mean Squared Error (MSE):
 
$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

where $y_i$ is the actual price and $\hat{y}_i$ is predicted price.

### Dataset
The dataset used in this experiment is the stock price of Google taken from the Alpha Vantage API. It is a weekly financial market dataset comprising 951 records from `2004-08-27` to `2022-11-08` and 7 columns:
- **Open**: opening price
- **High**: maximum price during the day
- **Low**: minimum price during the day
- **Close**: close price adjusted for splits\
- **Adj** Close: adjusted close price adjusted for both dividends and splits
- **Volume**: the number of shares that changed hands during a given day
- **Dividend**: payments a company makes to share profits with its stockholders

In this experiment, the adjusted close price is the value predicted by three models, the $80\%$ of the data is used as a train set, and the remaining $20\%$ is a validation set. 
![image](https://github.com/DoDucNhan/Stock-Price-Prediction/assets/44297479/673c71ff-12a9-4c8b-941a-ec8c6a5b124b)


### Data Preprocessing
The price value can be minimal and tremendous over time, potentially skewing the model performance in unexpected ways. This problem is where data normalization comes in. Normalization can increase the model's accuracy and help the optimization algorithm converge more quickly towards the target minima. By bringing the input data on the same scale and reducing its variance, the model can more efficiently learn from the data and store patterns in the network. Furthermore, RNN and its variations are intrinsically sensitive to the scale of the input data. For all the above reasons, it is crucial to normalize the data using formula as follow:

$$ x' = \frac{x - \bar{x}}{\sigma} $$

where $\bar{x}$ is the mean of $x$  and $\sigma$ is its standard deviation.

### Data Preparation
The model is trained to predict the $21^{st}$ day price based on the past $20$ days' close prices. The number of days, $20$, was selected based on a few reasons:
- The length of sequence used in RNN/GRU/LSTM model typically ranges from 15 to 20 words
- Very long input sequences may result in vanishing gradients
- Longer sequences tend to have much longer training times

After transforming the dataset into input features and output labels, the shape of feature $X$ is $(951, 20)$, $950$ for the number of rows, each row containing a sequence of the past $20$ days' prices. The corresponding label $Y$ data shape is $(951,)$, which matches the number of rows in $X$. These processed data will then be split into train and validation sets.

### Model Architecture & Training Configuration
In this comparison, three different models are initialized with the same architecture as follows:
- **RNN/GRU/LSTM unit**: input size is $1$, hidden dimension is $32$, number of layers is $2$.
- **Dropout**: dropout with $p=0.2$ to prevent overfitting.
- **Fully-connected (FC) layer**: one FC layer with input size of $\text{number of layers} \times \text{hidden dimension} (2 \times 32)$, and output size is $1$.

The training process is performed with the same configuration: 
- **Batch size**: $64$
- **Epochs**: $100$
- **Learning rate**: $0.01$ 
- **Optimizer**: Adaptive Moment Estimation (Adam) 
- **Loss**: Mean Squared Error (MSE)

The learning rate scheduler is also applied to the training process in order to gain higher performance. This scheduler will decay the learning rate of each parameter group by $0.1$ every $40$ epochs. 

### Training Results

| Model      | Loss | No. Prams |
| ----------- | ----------- | ---- |
| RNN      | 0.004079       | 3,297 |
| LSTM   | 0.003603        | 12,993 |
| GRU   | 0.002126        | 9,761 |


After training with $100$ epochs, all three models can generalise data well and notice the price trend over time. Even though RNN is a vanilla model, it can still predict the price not too far from the actual price. Nevertheless, the price prediction from LSTM and GRU is more promising and closer to the actual price compared to RNN. This result is understandable since two later-developed models have been refined to address the vanishing gradient problem through their ability to capture information from the prior input. Surprisingly, the GRU model with fewer parameters, $9,761$ compared to $12,993$ in LSTM, has a higher performance with only $0.002126$ loss in the validation set. This result has justified the superior performance of GRU, and this model is a good option if memory resources are limited and fast results are desired. However, when it comes to a larger dataset, it would be better to choose LSTM since the dataset might have a long-term trend, and LSTM can exploit that information better for forecasting.

![image](https://github.com/DoDucNhan/Stock-Price-Prediction/assets/44297479/ff0e4418-d92b-484a-98db-aeb96695a0d2)
*Prediction of RNN*

![image](https://github.com/DoDucNhan/Stock-Price-Prediction/assets/44297479/7eb86e17-e8c1-4473-a070-57861515c5e5)
*Prediction of LSTM*

![image](https://github.com/DoDucNhan/Stock-Price-Prediction/assets/44297479/944cc36e-d978-492b-a44d-8c5583382073)
*Prediction of GRU*

## Conclusion
Through experiment and comparison, this project has shown how RNN, LSTM and GRU work and how to implement these models in Pytorch to solve the stock price prediction task. In the data preprocessing for stock price prediction, it is crucial to apply normalisation to avoid skewing the result since RNN, and its variants are sensitive to the scale of input data. Furthermore, the model comparison shows that the RNN model is less effective at forecasting time series data than GRU and LSTM. In addition, GRU is relatively new, but its performance is on par with LSTM and computationally more efficient. However, there is yet to be a clear answer to which variant performs better, which still depends on the task and dataset. 

Since the experiment was performed only on pretty straightforward models' architecture, there is still plenty of room to make changes to enhance the results obtained. For future direction, because the stock price can be easily affected by its company earnings or another company status, one approach to achieve better results is adding more features for training. Another way is to increase the model complexity by adding the FC layer before GRU/LSTM unit to map input values into a high-dimensional feature space, transforming the features for GRU/LSTM unit. 

## References
- *Data preparation and visualisation*: https://github.com/jinglescode/time-series-forecasting-pytorch
