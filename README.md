# Sentiment-Analysis-Twitter-Feeds
Aim: Analyze the sentiment for a tweet
----
BaseLine: Created Bag of words, and calculated the probability based on the occurrence of words in tweets to determine a sentiment.

Bag of words was determined as follows:
1) We took synonyms of four types of emotions (angry, sad, happy, fear) we were analyzing a tweet on. 
2) To increase the size of the bag we downloaded tweets based on the words we got in the first step. Thereafter we applied lexical feature like tokenization, stop word removal, Lesk and synonym features to increase the bag of words by 50%.

Improvement Strategies on BaseLine:
1) Lexical: Applied tokenization, stop word removal, used regular expression to remove URL link, created abbreviation file to replace internet jargon to that word, for example: lyk:Like or lv:love. Removed repeated character , for example: havvve to have. Improved accuracy: 10%
2) Syntactic: Applied chunking and part of speech tagging selected verb, adjective, noun and adverb. Improved accuracy: 25%
3) Semantic: Applied Lesk and synonymy feature: Improved accuracy: 25%

After Applying all the NLP feature one by one we got a feature vector. To test this feature vector we created learner model using Navie Bayes algorithm, for this we have manually filtered 200 plus tweets related to each sentiment and train this model. Then we used our feature vector and model to determine sentiment of tweet.Improved accuracy: 20%

Language and Tool: Python and NLTK, Pycharm
