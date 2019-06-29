I develop a character level LSTM that predicts next character based on the current character. Its based on Andrej Karpathy's great [blog](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) post on RNNs. I use the LSTM module provided by Pytorch. I use Cross Entropy loss and Adam optimizer. Additionally, I use gradient clipping to avoid exploding gradients. 

The model is trained on text from a novel Anna Karenina (also included in the data folder). The text looks like this:
```
Happy families are all alike; every unhappy family is unhappy in its own
way.

Everything was in confusion in the Oblonskys' house. The wife had
discovered that the husband was carrying on an intrigue with a French
girl, who had been a governess in their family, and she had announced to
her husband that she could not go on living in the same house with him.
```

We train the model for 20 epochs. For predicting we do top-k sampling, i.e. limit results to k values and then pick from them based on their probabilities. We prime our model on two small texts (so as to avoid random output), and get these results. 
For prime `Anna`:
```
Anna And Levin, he shook her
first soul of him. She was saying him. She stopped to spond. Affect an office the same time were the same, had as
the country, there was so laty as to her side of them. The sounts of
the
plies songent to the dressing--he could not speak of her arts the
peasant, who came to
the
point that seemed shakleds, tries
to her attempt, to
himself the sorts with his side of the memoriate carriage.

"I have nothing to be married.
There
were a feeling for you and his high single chair, the port and
happiness to
be told her, and so then how is he
has been difficult to ask her about it.
And with your work," had answered something fashionately and
stepping in the
child of her little sour of when
he was
not said, she could not help matter of the same,
began to go the plint into
too. Her husband was not the sould of
the strick and served trans at sick of the moment to
the carest and his conviction that it, secretaying an than so much that was not in a fine takes and with the peop
```

And for prime `and Levin said`:
```
And Levin said, not the children and serfuce and the
fast of the minds in the
past, and he was, said he went
on.

That he would not help spenting. His sense to her face alone there was now an induciness of his selver, when he did not spend of her, and his choice was choice. His face had seen is,
was tried to see her, he felt that that his face he headd to her, she had not the bride, but she saw that
his broks where he was too to herself; that the
pictires had at
a seemed seeing to
stand horrible in whom he sat standing in his foremen in spite
of that.

"You can't let him both on the call of some sort."

"Alexey, too," he said, looking at the
propressivity, and went into her face, and
smiling
his back of the self conversation.

"Ahe it's better?"

"Why so?..."

"Yes, yes," had seemed surmer his son, though the church and his conversation had bound his hands, and a few difficult the manse of a conversation thought he was as soon of
time to trink. He
was so threster when he was asked, taking his father the princess to the prince. "Wait one of the prince with you. The
same stood," said Levin. "Thank,' have something more all
this."

"Yes.... All your sense that the mistake made it, to think out of the sight of the complaces of marriage, and a sort of
painfull of me to say to
him.

"Whom I am too. I don't know him?"

"I would have alled to do
this man, should never be many faces about your house."

"Well, I don't know,
I
said you are first into the carriage?"

"Well, was anyone, with him," said Levin, "was the
same time on the certain thought of her, the second service, the folce was a chunce that the conditions took and too, but he called them. It was a long--and then as is something they means without the portraits."

"What is it, anyone! There were a familiar sours, and here, to tell them in
any money farenize, when you are
for the pirticulor of the silence that have been, then a pity and all so as to do an evening
of tears and marrilge. There are now."

"Yes, I can't go to
have the
```