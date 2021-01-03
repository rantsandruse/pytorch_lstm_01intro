# Learning Pytorch in Ten Days: Day 1 - Train a basic LSTM tagger

##  Foreword

As I was teaching myself pytorch for applications in deep learning/NLP, I noticed that there are certainly no lacking of 
tutorials and examples. However, there's a consistent pattern of focusing more on the how's than why's. But learning 
**the how's without understanding why's is very dangerous**, especially for deep learning:

First of all, unlike experimental sciences where your experiments either succeed or fail, your first forays into deep 
learning may look like them succeed because the code "runs" and the result looks "reasonably good". But it could be suboptimal 
or worse still, incorrect. And it might be hard to pinpoint what went wrong, given the relatively low interpretability 
of deep learning models. 

Second (but no less important), knowing the how's alone will let you practice deep learning, but won't help you understand 
deep learning and further iterate & innovate. In order to innovate on the existing methodologies (or at least apply them 
correctly), you need to dig a little (or a lot) deeper: Understand the theoretical and practical motivations. Identify 
the alternatives. Evaluate the trade-offs. Grind over the shortfalls of your model and how you may improve them. Observe
and summarize the research trends over time and think about what's next. 

Unfortunately, the burden of a) verify that your code/method is correct b) identify the best resource to further your 
learning falls on the bright-eyed, bushy-tailed students, who most likely have yet to develop the mindset of scientific 
skepticism or the intuition to discern noise from signal. 

So I developed this set of beginner to intermediate level tutorials, where I would provide examples you can run, AND 
discuss the theoretical and practical motivations, the alternatives, the trade-offs, the potential improvements, and the 
trends. We will progress from the simple 10-line tutorial from [pytorch.org](https://pytorch.org) to applying transfer learning with one 
of the state-of-art models to a sentiment classification problem. 

While I tried my best to convey the hows' and the whys' in a series of short blog posts, there are a lot of important 
aspects that I skimmed over. Nevertheless, I hope that these tutorials would serve as a starting point that 
would inspire you to think critically, to dig effectively and to research thoroughly. I am also offering "further reading suggestions" 
after each tutorial, which hopefully can help you achieve these goals. 

Without further ado, let's start with **the Day 1 tutorial**.      
The Day 1 tutorial is at beginner level, with the majority of the content based on [the pytorch LSTM tutorial](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html) from [the pytorch official website](pytorch.org).
The original tutorial is quite straightforward to follow. But while I was going through it, I noticed that a few non-trivial
details were skimmed over. Googling these details suggested that they are oftentimes a source of confusion, and results 
in lengthy, back-and-forth discussions on multiple message boards. 

So let me break down these points for you, in Q & A format below:  
### Setup and Preparation 
I ran all my tutorials from *Ubuntu 18.04* with *python 3.8.5*. For all the required libraries, please check out *requirements.txt*. You can install them via:

      conda install --file requirements.txt

(Note: you could also use pip, but I find it easier to do conda install for pytorch and related libraries)
If you've never heard of LSTM, I'd also recommend taking a look at [Colah's blog](https://colah.github.io/posts/2015-08-Understanding-LSTMs/), which 
provides great visualizations and an intuitive understanding of RNN/LSTM architectures. Briefly, LSTM is a type of recurrent neural network 
architecture amenable to processing sequences, and has grown to be an extremely popular choice for a wide range of applications, 
including but not limited to speech recognition, natural language translation, and intrusion systems detection. 

### How to preprocess inputs 

Instead of hard coding the data into your python script, let's assume that you've been provided with the following csv file
as your raw input, where the first field is the raw text, followed by the corresponding speech tags in the second field: 

      text,tag
      "The dog ate the apple","DET NN V DET NN"
      "Everybody read the book","NN V DET NN"

Next, you will read in the raw data, break down each sequence into individual tokens, and convert your training data
into a list of tuples: 

      # read in raw data
      training_data_raw = pd.read_csv("./train.csv")
      # create mappings
      
      # split texts and tags into training data. 
      texts = [t.split() for t in training_data_raw["text"].tolist()]
      tags_list = [t.split() for t in training_data_raw["tag"].tolist()]

      training_data = list(zip(texts, tags_list))

TODO: provide sample output 
You will also create two dictionaries: *word_to_ix* provides mapping from words to indices; *tag_to_ix* provides mapping 
from tags to indices. 
      
      def seqs_to_dictionary(training_data: list):
      '''
      Parameters
      ----------
      training_data: training data as a list of tuples.

      Returns
      -------
      word_to_ix: a dictionary mapping words to indices
      tag_to_ix: a dictionary mapping tags to indices
      '''
      word_to_ix = {}
      tag_to_ix = {}
      count1 = count2 = 0

      for sent, tags in training_data:
         for word in sent:
            if word not in word_to_ix:
                word_to_ix[word] = count1
                count1 += 1
         for tag in tags:
            if tag not in tag_to_ix:
                tag_to_ix[tag] = count2
                count2 += 1

    return word_to_ix, tag_to_ix

    word_to_ix, tag_to_ix = seqs_to_dictionary(training_data)

TODO: provide sample output 
### How to set up a basic LSTM model
In pytorch, describing a basic model includes initialization and setting up the forward pass. You do not need to worry
about back propagation, as it is gracefully handled by [torch.autograd](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html). 

In this case, we call our LSTM model *LSTMTagger*. Here the embedding dimension(*embedding_dim*) describes the length of 
the input feature (In this case, the length of word embedding. e.g. it would be 300 if we were using a standard GLOVE embedding), the hidden dimension(*hidden_dim*) describes 
The number of features in the hidden state. You can also think of it as the input dimension of a single LSTM cell. 

      class LSTMTagger(nn.Module):
          
          def __init__(self, embedding_dim, hidden_dim, vocab_size, output_size, is_nll_loss=False):
              '''
              embedding_dim: Glove is 300. We are using 6 here.
              hidden_dim: can be anything, usually 32 or 64. We are using 6 here.
              vocab_size: vocabulary size includes an index for padding
              output_size: We need to exclude the index for padding here.
              '''
              super().__init__()
              self.hidden_dim = hidden_dim
              self.embedding_dim = embedding_dim
              # In this case, vocab_size is 9, embedding_dim is 6.
              self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
      
              # The LSTM takes word embeddings as inputs, and outputs hidden states
              # with dimensionality hidden_dim.
              self.lstm = nn.LSTM(embedding_dim, hidden_dim)
      
              # The linear layer that maps from hidden state space to tag space
              self.hidden2tag = nn.Linear(hidden_dim, output_size)
              self.is_nll_loss = is_nll_loss

Next, we will define our forward pass as part of the *LSTMTagger* model: 

          def forward(self, sentence):
              # Note: we can implement cross entropy loss in two ways:
              # 1. Use NLL + log softmax:
              # NLLoss = - (input)
              # where input = log_softmax(x)
              # 2. Use Cross entropy directly:
              # CrossEntropyLoss = -log_softmax(input)
              # input = x
              # You can pass raw logits for the latter but need log_softmax for the former.
              embeds = self.word_embeddings(sentence)
              # the dimension should be: seq_len, batch_size, -1
              lstm_out, (h, c) = self.lstm(embeds.view(len(sentence), 1, -1))
              print("lstm out shape:", lstm_out.shape)
              print("hshape:", h.shape)
              print("cshape:", c.shape)
              tag_scores = self.hidden2tag(lstm_out.view(len(sentence), -1))
              if self.is_nll_loss:
                  tag_scores = F.log_softmax(tag_scores, dim=1)
              return tag_scores

At this point, you might feel a little overwhelmed by **NLL Loss**, **Cross entropy loss** and **softmax** function. And we will 
go over them together: 
#### 1. What is NLL (Negative log loss) Loss in pytorch? 
   **The short answer**: The NLL loss function in pytorch is **NOT really** the NLL Loss.

   The textbook definition of NLL Loss is the sum of negative log of the correct class:
      
      
 <img src="https://render.githubusercontent.com/render/math?math=NLLLoss=-\\sum(y_i* log(p_i))">
   
   In comparison, the pytorch implementation takes for granted that x<sub>i</sub> = log(p<sub>i</sub>), where x<sub>i</sub> 
   is the input. (Note: The default pytorch implementation calculates the mean loss (*reduction=mean*). What you usually
   see in textbooks/wikipedia is the sum of all losses (i.e. without 1/n) (*reduction=sum* in pytorch)):  
   

 <img src="https://render.githubusercontent.com/render/math?math=NLLLoss=-\\sum(y_i* x_i)">

   
   where <img src="https://render.githubusercontent.com/render/math?math=x=log(p_i)">, and y is still the ground truth.  
   This means your input **has already gone through the log_softmax transformation BEFORE** you feed it into the NLL function 
   in pytorch.

To gain more intuition, take a look at the example provided in [main_example.py](https://github.com/rantsandruse/pytorch_lstm_01intro/blob/main/main_example.py): 

      softmax_prob = torch.tensor([[0.8, 0.2], [0.6, 0.4]])
      log_softmax_prob = torch.log(softmax_prob)
      print("Log softmax probability:", log_softmax_prob)
      target = torch.tensor([0,0])
      nll_loss = F.nll_loss(log_softmax_prob, target)
      print("NLL loss is:", nll_loss)

In this example, When *target = [0,0]*, both ground truth classifications belong to the first class:

      y1 = [1,0], y2 = [1,0]
      x1 = log(p1) = [log(0.8), log(0.2)] = [-0.22, -1.61]
      x2 = log(p2) = [log(0.6), log(0.4)] = [-0.51, -0.91]
      pytorch_NLL_loss = -1/n (sum(yi * xi)) = 1/2 * (-0.22*1 - 0.51*1) = 0.36
#### 2. What is softmax? 
   Softmax function, or the normalized exponential function, is often used as the last activation function in a neural network. The
standard softmax function is given as: 

   <img src="https://render.githubusercontent.com/render/math?math=$P_i=\frac{e^{-bi}}{\sum(e^{-bi})}">

   If it doesn't look familar to you, think of n = 2, i.e. a neural network with binary outputs: 

   <img src="https://render.githubusercontent.com/render/math?math=$P_0=\frac{1}{1%2Be^{-b0}}$">
   <img src="https://render.githubusercontent.com/render/math?math=$P_1=\frac{e^{-b0}}{1%2Be^{-b0}}$">

As you might have realized by now, softmax function is simply a more generalized logistic function in 2 or more dimensions. 

   
#### 2. What is the relationship between NLL Loss, log_softmax and cross entropy loss in pytorch? 
   **The short answer**:  *NLL_loss* + *log_softmax* = *cross_entropy_loss* in pytorch. 

   The [original tutorial](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html) is actually using 
   cross entropy loss via *NLL Loss* + *log_softmax*, where the *log_softmax* operation was applied to the final layer 
   of the LSTM network (in [model_lstm_tagger.py](https://github.com/rantsandruse/pytorch_lstm_01intro/blob/main/model_lstm_tagger.py)): 
   
         tag_scores = F.log_softmax(tag_scores, dim=1)

   And then NLL Loss is applied on the tag_scores (in [main_v1.py](https://github.com/rantsandruse/pytorch_lstm_01intro/blob/main/model_v1.py)):

         loss_function = nn.NLLLoss()
         loss = loss_function(tag_scores, targets)

   Whether you are using the cross entropy loss or the NLL loss + log_softmax, you should be able to 
   recover softmax probability fairly easily: Given that NLL loss takes log_softmax as input, 
   you need to apply an exponential in your model inference stage, in order to recover the probabilities. 
   In contrast, the cross entropy loss function takes the raw input before applying softmax. As a result, you only need 
   to apply softmax in order to recover your probabilities. This is demonstrated here (in model_lstm_tagger.py):
        
        if model.is_nll_loss:
            # Use NLL loss
            print("Using NLL Loss:")
            tag_prob = tag_scores.exp()
        else:
            # Use cross entropy loss
            print("Using cross entropy loss")
            tag_prob = F.softmax(tag_scores)

        
#### 3. Why doesn't pytorch provide us with a loss function, so that we can obtain softmax from the forward pass output directly? 
   **The short answer**: Numerical stability. 

   At this point, this is a natural follow-up question. It always seems a little cumbersome that neither NLL loss/log_softmax 
   combo nor Cross entropy loss function takes the softmax probability as input, so we always have to transform it 
   after the forward pass. Wouldn't it be much easier if pytorch provides us with a loss function based on log(x) (similar 
   to what [this 
   stackflow post](<https://stackoverflow.com/questions/58122505/suppress-use-of-softmax-in-crossentropyloss-for-pytorch-neural-net>) suggested, where x = softmax(raw_logits)? This way once obtain softmax proabability from the forward pass, we can pass it directly to the 
   loss function without further transformations: 

   Loss function | Input | Transformation into probability score | 
   --- | --- | --- |
   NLL loss | x = log_softmax(raw_logits) | exp(x) |
   Cross entropy loss | x = raw_logits | softmax(x) | 
   How about this? | x = softmax(raw_logits) | x (no transformation needed!)  

   So... why not? This is **intentionally avoided by** pytorch due to numerical instability of softmax and risk of overflowing, especially 
   if your features are not normalized. As this medium post entitled ["Are you messing with me softmax"](https://medium.com/swlh/are-you-messing-with-me-softmax-84397b19f399) pointed 
   out, the largest value without overflowing in numpy is 709. Using log_softmax, just as what we did in our loss function mitigates the risk of numerical instability
   by using the log-sum-exp trick. 
   
   Briefly, it calculates: 

<img src="https://render.githubusercontent.com/render/math?math=y=log\\sum(e^{x_i})=a %2B log\\sum(e^{x_i-a})">

   Thus avoiding the numerical explosion of e<sup>xi</sup>. More details and derivations are given [here](https://www.xarg.org/2016/06/the-log-sum-exp-trick-in-machine-learning/).
   So shall we use log_softmax + NLL loss or Cross entropy loss moving forward? As explained in [this pytorch discussion thread](https://discuss.pytorch.org/t/trouble-getting-probability-from-softmax/26764/4), 
   using either of these options are perfectly fine and equivalent, but using log + softmax + NLL (i.e. separating log from softmax) could lead to numerical 
   instability and is therefore **strongly discouraged**. For future tutorials, I will stick to the cross entropy loss functions provided by pytorch.    


#### 4. Why should we choose cross entropy loss function over others, e.g. MSE loss, hinge loss, etc?  
   **The short answer**: You can be derive it via MLE, or think of it in terms of KL divergence. 

   If you ignore the LSTM layer of this LSTM tagger for a second (so you are left with just the linear layer), then it 
   becomes obvious that this is essentially a logistic regression problem, where we aim to find a tag probability between 
   0 and 1 for each of the words. Cross entropy loss is well established for logistic regression models. You can either 
   derive it via MLE ([maximum likeihood estimation](https://ocw.mit.edu/courses/mathematics/18-05-introduction-to-probability-and-statistics-spring-2014/readings/MIT18_05S14_Reading10b.pdf)), by maximizing the joint predicted likelihood function. Alternatively, 
   you can think of it as minimizing the [KL divergence](https://nowak.ece.wisc.edu/ece830/ece830_spring15_lecture7.pdf), which measures the difference between the predicted probability 
   distribution and the actual probability distribution. The connections of cross entropy loss to both MLE and KL divergence 
   is well illustrated in [this blog post by Lei Mao](https://leimao.github.io/blog/Cross-Entropy-KL-Divergence-MLE/) . 


### How to train your model and use it for prediction 
We will add a *train* and a *test* function to train our model and perform inference, respectively: 



#### 1. What is NLL (Negative log loss) Loss in pytorch? 
   **The short answer**: The NLL loss function in pytorch is **NOT really** the NLL Loss.

   The textbook definition of NLL Loss is the sum of negative log of the correct class:
      
      
 <img src="https://render.githubusercontent.com/render/math?math=NLLLoss=-\\sum(y_i* log(p_i))">
   
   In comparison, the pytorch implementation takes for granted that x<sub>i</sub> = log(p<sub>i</sub>), where x<sub>i</sub> 
   is the input. (Note: The default pytorch implementation calculates the mean loss (*reduction=mean*). What you usually
   see in textbooks/wikipedia is the sum of all losses (i.e. without 1/n) (*reduction=sum* in pytorch)):  
   

 <img src="https://render.githubusercontent.com/render/math?math=NLLLoss=-\\sum(y_i* x_i)">

   
   where <img src="https://render.githubusercontent.com/render/math?math=x=log(p_i)">, and y is still the ground truth.  
   This means your input **has already gone through the log_softmax transformation BEFORE** you feed it into the NLL function 
   in pytorch.

To gain more intuiton, take a look at the example provided in [main_example.py](https://github.com/rantsandruse/pytorch_lstm_01intro/blob/main/main_example.py): 

      softmax_prob = torch.tensor([[0.8, 0.2], [0.6, 0.4]])
      log_softmax_prob = torch.log(softmax_prob)
      print("Log softmax probability:", log_softmax_prob)
      target = torch.tensor([0,0])
      nll_loss = F.nll_loss(log_softmax_prob, target)
      print("NLL loss is:", nll_loss)

In this example, When *target = [0,0]*, both ground truth classifications belong to the first class:

      y1 = [1,0], y2 = [1,0]
      x1 = log(p1) = [log(0.8), log(0.2)] = [-0.22, -1.61]
      x2 = log(p2) = [log(0.6), log(0.4)] = [-0.51, -0.91]
      pytorch_NLL_loss = -1/n (sum(yi * xi)) = 1/2 * (-0.22*1 - 0.51*1) = 0.36

#### 2. What is the relationship between NLL Loss, log_softmax and cross entropy loss in pytorch? 
   **The short answer**:  *NLL_loss* + *log_softmax* = *cross_entropy_loss* in pytorch. 

   The [original tutorial](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html) is actually using 
   cross entropy loss via *NLL Loss* + *log_softmax*, where the *log_softmax* operation was applied to the final layer 
   of the LSTM network (in [model_lstm_tagger.py](https://github.com/rantsandruse/pytorch_lstm_01intro/blob/main/model_lstm_tagger.py)): 
   
         tag_scores = F.log_softmax(tag_scores, dim=1)

   And then NLL Loss is applied on the tag_scores: 

         loss_function = nn.NLLLoss()
         loss = loss_function(tag_scores, targets)

   Whether you are using the cross entropy loss or the NLL loss + logsoftmax, you should be able to 
   recover softmax probability fairly easily: Given that NLL loss takes log_softmax as input, 
   you need to apply an exponential in your model inference stage, in order to recover the probabilities. 
   In contrast, the cross entropy loss function takes the raw input before applying softmax. As a result, you only need 
   to apply softmax in order to recover your probabilities. This is demonstrated here (in model_lstm_tagger.py):
        
        if model.is_nll_loss:
            # Use NLL loss
            print("Using NLL Loss:")
            tag_prob = tag_scores.exp()
        else:
            # Use cross entropy loss
            print("Using cross entropy loss")
            tag_prob = F.softmax(tag_scores)

        
#### 3. Why doesn't pytorch provide us with a loss function, so that we can obtain softmax from the forward pass output directly? 
   **The short answer**: Numerical stability. 

   At this point, this is a natural follow-up question. It always seems a little cumbersome that neither NLL loss/log_softmax 
   combo nor Cross entropy loss function takes the softmax probability as input, so we always have to transform it 
   after the forward pass. Wouldn't it be much easier if pytorch provides us with a loss function based on log(x) (similar 
   to what [this 
   stackflow post](<https://stackoverflow.com/questions/58122505/suppress-use-of-softmax-in-crossentropyloss-for-pytorch-neural-net>) suggested, where x = softmax(raw_logits)? This way once obtain softmax proabability from the forward pass, we can pass it directly to the 
   loss function without further transformations: 

   Loss function | Input | Transformation into probability score | 
   --- | --- | --- |
   NLL loss | x = log_softmax(raw_logits) | exp(x) |
   Cross entropy loss | x = raw_logits | softmax(x) | 
   How about this? | x = softmax(raw_logits) | x (no transformation needed!)  

   So... why not? This is **intentionally avoided by** pytorch due to numerical instability of softmax and risk of overflowing, especially 
   if your features are not normalized. As this medium post entitled ["Are you messing with me softmax"](https://medium.com/swlh/are-you-messing-with-me-softmax-84397b19f399) pointed 
   out, the largest value without overflowing in numpy is 709. Using log_softmax, just as what we did in our loss function mitigates the risk of numerical instability
   by using the log-sum-exp trick. 
   
   Briefly, it calculates: 

<img src="https://render.githubusercontent.com/render/math?math=y=log\\sum(e^{x_i})=a %2B log\\sum(e^{x_i-a})">

   Thus avoiding the numerical explosion of e<sup>xi</sup>. More details and derivations are given [here](https://www.xarg.org/2016/06/the-log-sum-exp-trick-in-machine-learning/).
   So shall we use log_softmax + NLL loss or Cross entropy loss moving forward? As explained in [this pytorch discussion thread](https://discuss.pytorch.org/t/trouble-getting-probability-from-softmax/26764/4), 
   using either of these options are perfectly fine and equivalent, but using log + softmax + NLL (i.e. separating log from softmax) could lead to numerical 
   instability and is therefore **strongly discouraged**. For future tutorials, I will stick to the cross entropy loss functions provided by pytorch.    


#### 4. Why should we choose cross entropy loss function over others, e.g. MSE loss, hinge loss, etc?  
   **The short answer**: You can be derive it via MLE, or think of it in terms of KL divergence. 

   If you ignore the LSTM layer of this LSTM tagger for a second (so you are left with just the linear layer), then it 
   becomes obvious that this is essentially a logistic regression problem, where we aim to find a tag probability between 
   0 and 1 for each of the words. Cross entropy loss is well established for logistic regression models. You can either 
   derive it via MLE ([maximum likeihood estimation](https://ocw.mit.edu/courses/mathematics/18-05-introduction-to-probability-and-statistics-spring-2014/readings/MIT18_05S14_Reading10b.pdf)), by maximizing the joint predicted likelihood function. Alternatively, 
   you can think of it as minimizing the [KL divergence](https://nowak.ece.wisc.edu/ece830/ece830_spring15_lecture7.pdf), which measures the difference between the predicted probability 
   distribution and the actual probability distribution. The connections of cross entropy loss to both MLE and KL divergence 
   is well illustrated in [this blog post by Lei Mao](https://leimao.github.io/blog/Cross-Entropy-KL-Divergence-MLE/) . 
    
### A few additional points are described below: 
1. The quick example (see [main_example.py](https://github.com/rantsandruse/pytorch_lstm_01intro/blob/main/main_example.py)) suggests two alternatives for feeding in data. The first solution allows 
   for feeding individual words sequentially. The second one allows you to provide all the inputs in one tensor. I would 
   personally recommend the second one. 
2. I have two versions of the main function. The first version is [main_v1.py](https://github.com/rantsandruse/pytorch_lstm_01intro/blob/main/main_v1.py) 
   with a few improvements from the original tutorial: 
   
   a. I used a *seqs_to_dictionary* function instead of hard code for tag_to_ix dictionary.
   
   b. I added an example of inference for non-training data. 
   
   c. I added a demonstration of the relationship between NLL loss and Cross entropy loss
3. The second version is [main_v2.py](https://github.com/rantsandruse/pytorch_lstm_01intro/blob/main/main_v2.py). The main improvements from v1 are as follows:
   
   a. I added code to read from an .csv file instead of hard coding the input data inline. 
   
   b. I implemented very basic "train" and "test" functions.
      Training and testing utilities are essential to any machine learning pipelines. The "train" function takes in custom 
      data set and custom loss function as arguments, and trains model over a number of epochs. The "test" function freezes 
      the gradient with torch.nograd, and outputs the inference scores.  
   
### What's next 
In the next tutorial, I will show the how's and why's of training an LSTM tagger in mini-batches.    

### Further reading 
1. https://medium.com/swlh/are-you-messing-with-me-softmax-84397b19f399
2. https://www.xarg.org/2016/06/the-log-sum-exp-trick-in-machine-learning/   
3. https://ocw.mit.edu/courses/mathematics/18-05-introduction-to-probability-and-statistics-spring-2014/readings/MIT18_05S14_Reading10b.pdf
4. https://nowak.ece.wisc.edu/ece830/ece830_spring15_lecture7.pdf
5. https://leimao.github.io/blog/Cross-Entropy-KL-Divergence-MLE 


    
