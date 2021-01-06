# Learning Pytorch in Ten Days: Day 1 - How to train a basic LSTM tagger (and how to use NLL Loss, Cross entropy loss and softmax)

##  Introduction 

As I was teaching myself pytorch for applications in deep learning/NLP, I noticed that there are certainly no lacking of 
tutorials and examples. However, I consistently find a lot more explanations of the hows' than the whys'. 

I believe that **knowing the how's without understanding the whys' is quite dangerous**. It's like applying a chainsaw without 
reading the section about when not to use it. Besides, unlike experimental sciences where your experiments either succeed 
or fail, botched deep learning experiments **may look like they succeed** if the code "runs" and the result looks "reasonably good". 
By "botched" I mean suboptimal or worse still, fundamentally incorrect. 

The other reason for knowing the whys' is innovation. Without understanding the fundamentals, it's hard to grasp the 
theoretical and practical motivations or identify the limitations of the state-of-art methodologies, both of which are critical steps to 
improving the status quo. 

So I developed this set of beginner-to-intermediate level tutorials, where I hope to provide a balanced account of the hows' **AND** 
the whys'. At the end of 10 days, I hope that you will be inspired to: Understand the theoretical and practical motivations. 
Identify the alternatives. Evaluate the trade-offs. Grind over the shortfalls of your model and how you may improve them. Observe
and summarize the research trends over time and brainstorm over what's next. 


(Links to be provided below)

**Day 1** - How to train a basic LSTM Tagger (and why use NLL Loss, Cross entropy loss and softmax functions)

**Day 2** - How to train an LSTM Tagger with minibatch (and implement proper padding, masking & initialization)          

**Day 3** - How to convert an LSTM Tagger to an LSTM Classifier (easy day)

**Day 4** - Why doesn't your neural network model overfit (experiment on intrinsic dimensions)

**Day 5** - How to choose your optimizer (and why not ADAM) 

**Day 6** - How to apply various regularization techniques 

**Day 7** - How to modify your unidirectional LSTM into bidirectional (plus some stacking)

**Day 8** - How to implement self-attention LSTM model with regularization from Bengio et al 

**Day 9** - How to add ELMO layers to your LSTM model 

**Day 10** - How to tune your BERT transformer (explore the effect of batch size) 


Without further ado, let's start with **the Day 1 tutorial**.      

This tutorial is at beginner level, with the majority of the content based on [the basic pytorch LSTM tutorial](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html) from [the pytorch official website](pytorch.org).
Instead of repeating what was given there, I will focus on i. improving/providing alternatives for the existing code 2. explain the relevant whys'

### Setup and prep for beginners    
To set up your environment with *pytorch* and associated libraries, you can install them via:

      conda install --file requirements.txt

(Note: you could also use pip, but I found it easier to do conda install for pytorch and related libraries)
If you've never heard of RNN/LSTM, I'd also recommend taking a look at [Colah's blog](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) first.
For some intuitive understanding of negative log loss and softmax function, you can check out [this blogpost](https://ljvmiranda921.github.io/notebook/2017/08/13/softmax-and-the-negative-log-likelihood/). 
You should also try to read/run main_v2.ipynb.  
### Improvements & modification over original code: 

#### How to preprocess inputs 
Instead of hard coding the input data into your python script (which the original tutorial did for illustrative purposes), I set up a small csv 
file for input instead. This is to set the stage for reading real data instead of mock inputs.  

      text,tag
      "The dog ate the apple","DET NN V DET NN"
      "Everybody read the book","NN V DET NN"

Next, I read in the raw data, resulting in *training_data* in the same format as the original tutorial, i.e. a list of tuples. 

      # read in raw data
      training_data_raw = pd.read_csv("./train.csv")
      # create mappings
      
      # split texts and tags into training data. 
      texts = [t.split() for t in training_data_raw["text"].tolist()]
      tags_list = [t.split() for t in training_data_raw["tag"].tolist()]

      training_data = list(zip(texts, tags_list))

TODO: provide output 

Then I consolidated the process of mapping token to index into a single function for both words and tags: 
      
      def seqs_to_dictionary(training_data: list):
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
For the LSTMTagger model setup, very little modifications were made, except that I added a couple of lines 
to show how to use NLL loss or cross entropy loss: 

Next, we will define our forward pass as part of the *LSTMTagger* model: 

          def forward(self, sentence):
              ... 
              tag_scores = self.hidden2tag(lstm_out.view(len(sentence), -1))
              # modification starts  
              if self.is_nll_loss:
                  tag_scores = F.log_softmax(tag_scores, dim=1)
              # modification ends.    
              return tag_scores


### How to train your model and use it for prediction
In comparison to the original tutorial, we consolidated the code into *train* and *test* functions in [train.py](https://github.com/rantsandruse/pytorch_lstm_01intro/blob/main/main_example.py): 

      def train(model, loss_fn, training_data, word_to_ix, tag_to_ix, optimizer, epoch=10):
         for epoch in range(epoch):  # again, normally you would NOT do 300 epochs, it is toy data
           for sentence, tags in training_data:
               print(sentence)
               print(tags)
               # Step 1. Remember that Pytorch accumulates gradients.
               # We need to clear them out before each instance
               model.zero_grad()
   
               # Step 2. Get our inputs ready for the network, that is, turn them into
               # Tensors of word indices.
               sentence_in = seq_to_embedding(sentence, word_to_ix)
               targets = seq_to_embedding(tags, tag_to_ix)
   
               # Step 3. Run our forward pass.
               tag_scores = model(sentence_in)
               # Step 4. Compute the loss, gradients, and update the parameters by
               #  calling optimizer.step()
               loss = loss_fn(tag_scores, targets)
               print("loss for epoch ", epoch, ":", loss)
               loss.backward()
               optimizer.step()

For running inference, we define the test function:
      
      def test(testing_data, model, word_to_ix):
         
          with torch.no_grad():
              inputs = seq_to_embedding(testing_data.split(), word_to_ix)
              tag_scores = model(inputs)
              # Now evaluate probabilistic output
              # For either NLL loss or cross entropy los
              if model.is_nll_loss:
                  # Use NLL loss
                  print("Using NLL Loss:")
                  tag_prob = tag_scores.exp()
              else:
                  # Use cross entropy loss
                  print("Using cross entropy loss")
                  tag_prob = F.softmax(tag_scores)
      
            print(tag_prob)
            return tag_prob

### Relationship between NLL Loss, softmax and cross entropy loss 
To fully understanding the model loss function and forward pass, a few terms (**NLL loss, softmax, cross entropy loss**) 
their relationship need to be clarified. 
#### 1. What is NLL (Negative log loss) Loss in pytorch? 
   **The short answer:** The NLL loss function in pytorch is **NOT really** the NLL Loss.

   The textbook definition of NLL Loss is the sum of negative log of the correct class:
      
      
 <img src="https://render.githubusercontent.com/render/math?math=NLLLoss_{textbook}=-\\sum(y_i* log(p_i))">
   
   Where y<sub>i</sub>=1 for the correct class, and y<sub>i</sub>=0 for the incorrect class. 
   In comparison, the pytorch implementation takes for granted that x<sub>i</sub> = log(p<sub>i</sub>), where x<sub>i</sub> 
   is the input. (Note: The default pytorch implementation calculates the mean loss (*reduction=mean*), if you want the 
   the textbook/wikipedia version, use *reduction=sum* instead):  

 <img src="https://render.githubusercontent.com/render/math?math=NLLLoss_{pytorch}=-\\sum(y_i* x_i)">
 
   where <img src="https://render.githubusercontent.com/render/math?math=x=log(p_i)">, and y is still the ground truth label.  
   This means **your input has already gone through the log_softmax transformation BEFORE you feed it into the NLL function 
   in pytorch**.

To gain more intuition, take a look at the example provided in [main_example.py](https://github.com/rantsandruse/pytorch_lstm_01intro/blob/main/main_example.py): 

      softmax_prob = torch.tensor([[0.8, 0.2], [0.6, 0.4]])
      log_softmax_prob = torch.log(softmax_prob)
      print("Log softmax probability:", log_softmax_prob)
      target = torch.tensor([0,0])
      nll_loss = F.nll_loss(log_softmax_prob, target)
      print("NLL loss is:", nll_loss)

In this example, When *target = [0,0]*, both ground truth labels belong to the first class:

      y1 = [1,0], y2 = [1,0]
      x1 = log(p1) = [log(0.8), log(0.2)] = [-0.22, -1.61]
      x2 = log(p2) = [log(0.6), log(0.4)] = [-0.51, -0.91]
      pytorch_NLL_loss = -1/n (sum(yi * xi)) = 1/2 * (-0.22*1 - 0.51*1) = 0.36

#### 2. What is the relationship between NLL Loss, log_softmax and cross entropy loss in pytorch? 
   **The short answer:**  *NLL_loss(log_softmax(x))* = *cross_entropy_loss(x)* in pytorch. 

   The *LSTMTagger* in the [original tutorial](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html) **is using 
   cross entropy loss** via *NLL Loss* + *log_softmax*, where the *log_softmax* operation was applied to the final layer 
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
   **The short answer**: Mitigate numerical instability. 

   It always seems a little cumbersome that neither NLL loss/log_softmax combo nor Cross entropy loss function takes the softmax 
   probability as input, so we always have to transform it after the forward pass. Wouldn't it be much easier if pytorch provides us with a loss function based on log(x), 
   similar to what [this 
   stackflow post](<https://stackoverflow.com/questions/58122505/suppress-use-of-softmax-in-crossentropyloss-for-pytorch-neural-net>) suggested, where x = softmax(raw_logits)? 
   This way once we obtain softmax proabability from the forward pass, we can pass it directly to the 
   loss function without further transformations: 

   Loss function | Input | Transformation into probability score | 
   --- | --- | --- |
   NLL loss | x = log_softmax(raw_logits) | exp(x) |
   Cross entropy loss | x = raw_logits | softmax(x) | 
   How about this? | x = softmax(raw_logits) | x (no transformation needed... but do NOT use it!!!)  

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


#### 4. Why should we choose cross entropy loss function over the others, e.g. MSE loss, hinge loss, etc?  
   **The short answer**: You can be derive it via MLE([maximum likeihood estimation](https://ocw.mit.edu/courses/mathematics/18-05-introduction-to-probability-and-statistics-spring-2014/readings/MIT18_05S14_Reading10b.pdf)), 
   or think of it in terms of KL divergence([Kullback-Leibler divergence](https://nowak.ece.wisc.edu/ece830/ece830_spring15_lecture7.pdf)). 

   If you ignore the complexity of the LSTM layer for a second, you are left with just the linear layer. Then it 
   becomes obvious that this is essentially a *multiclass logistic regression* problem, where we aim to find a tag probability between 
   0 and 1 for each of the words. Cross entropy loss is well established for logistic regression models. You can either 
   derive it via MLE ([maximum likeihood estimation](https://ocw.mit.edu/courses/mathematics/18-05-introduction-to-probability-and-statistics-spring-2014/readings/MIT18_05S14_Reading10b.pdf)), by maximizing the joint predicted likelihood function. Alternatively, 
   you can think of it as minimizing the [KL divergence](https://nowak.ece.wisc.edu/ece830/ece830_spring15_lecture7.pdf), which measures the difference between the predicted probability 
   distribution and the actual probability distribution. The connections of cross entropy loss to both MLE and KL divergence 
   is well illustrated in [this blog post by Lei Mao](https://leimao.github.io/blog/Cross-Entropy-KL-Divergence-MLE/) . 


### Quick house-keeping notes: 
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
   
### What's next 
In the next tutorial, I will show the how's and why's of training an LSTM tagger in mini-batches.    

### Further reading 
1. [Numerical instability of softmax](https://medium.com/swlh/are-you-messing-with-me-softmax-84397b19f399)
2. [Numerical trick for softmax](https://www.xarg.org/2016/06/the-log-sum-exp-trick-in-machine-learning/)   
3. [Maximum likehood estimate](https://ocw.mit.edu/courses/mathematics/18-05-introduction-to-probability-and-statistics-spring-2014/readings/MIT18_05S14_Reading10b.pdf) 
4. [KL Divergence](https://nowak.ece.wisc.edu/ece830/ece830_spring15_lecture7.pdf)
5. [Relationship between cross entropy, KL divergence and MLE](https://leimao.github.io/blog/Cross-Entropy-KL-Divergence-MLE) 
6. [Relationship between negative log loss and softmax](https://ljvmiranda921.github.io/notebook/2017/08/13/softmax-and-the-negative-log-likelihood/)


    
