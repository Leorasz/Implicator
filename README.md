Implicator  

This is an intermediate testing phase of a research project I'm working on. The goal is to be able to perform "forensic analysis" where given limited facts you can imply more information from it.  

Original Data  

The dataset used was originally Wikipedia and then that was altered to only have the intro sections of individuals. All of these were then fed through an LLM to break each mini-biography down into individual facts which were then further cleaned and parsed. All of the facts are then embedded and each individual is stored as the indices of their facts.  

Original Model  

The "model" consists of two neural networks: an updater::R^situation\_dimension->R^fact\_dimension->R^situation\_dimension which given the current situation vector (starting from the zero vector) creates an update to it given the fact. Then there's the poller::R^situation\_dimension->R^fact\_dimension->[0,1] which "polls" the situation for the fact and gives the "score" that the fact belongs to the situation. As previously mentioned, each individual is made up of facts. In training, for each fact the situation is updated with it, but then is polled against every other fact in the individual, including the ones not yet given. This "forced guessing" technique is the key to making implications and not letting the model stick to only what it's been given.

The embedding generation and generation of other useful files is embedding\_organizer.py, the implicator itself and training is in implicator.py, and to test it use implicator\_tester.py

LLM Version  

The next version leverages the existing knowledge of LLMs about what data is to be expected of individuals. Instead of have situation vectors and fact vectors, the situation and the fact being polled are in one string. Here's some examples of input data (from data/pos2.json):
```json
"Known information: They manage events. They write. They are a media marketing communications consultant. They produce. They are a rhythmic gymnastics national team member.\nIs this true: They are a rhythmic gymnastics national team member\nProbability:",
"Known information: They play soprano saxophone. They are a singer.\nIs this true: They play alto saxophone\nProbability:",
```
As you can see, it's both assuming a positive match for facts that are directly stated as well as ones that are not.

The implementation off with a hand-coded transformer to load Llama-3.1-8B-Instruct and adds LoRA to it as well as a brand-new classification head, whose output dimension is 1 to give the polling probability. This was then trained on two GPUs using a custom implementation of parallel training as well. Many training runs were done to adjust various things but after getting the code where it is now and training for probably around 10 hours on two 3090s (about a third of the data gone through) this is the result:

```
Give your fact: They are a professional football player
Give your fact: They played for the Chicago Bear
Give your fact: /p
Switching to polling mode
Give your fact: They played in the National Football League
The output was 0.73046875
Give your fact: They are male
The output was 0.7265625
Give your fact: They are female
The output was 0.23828125
Give your fact: They are a mathematician
The output was 0.06005859375
Give your fact: /r
Restarting the individual
Give your fact: Their name is Jack Abbott
Give your fact: /p
Switching to polling mode
Give your fact: They are male
The output was 0.6953125
Give your fact: They are female
The output was 0.056640625
Give your fact: They are British
The output was 0.193359375
Give your fact: They are American
The output was 0.72265625
Give your fact: They are Japanese
The output was 0.05419921875
```
