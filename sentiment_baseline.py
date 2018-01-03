from nltk.corpus import stopwords
from tweet_processing import testing_tweets_genartion
from tweet_processing import tweets_master_dic_generation

anger_dict = ['acrimony', 'animosity', 'annoyance', 'antagonism', 'displeasure', 'enmity', 'exasperation', 'fury',
              'hatred', 'impatience', 'indignation', 'ire', 'irritation', 'outrage', 'passion', 'rage', 'resentment',
              'temper', 'violence', 'chagrin', 'choler', 'conniption', 'dander', 'disapprobation', 'distemper', 'gall',
              'huff', 'infuriation', 'irascibility', 'irritability', 'miff', 'peevishness', 'petulance', 'pique',
              'rankling', 'soreness', 'stew', 'storm', 'tantrum', 'tiff', 'umbrage', 'vexation', 'blow up', 'cat fit',
              'hissy', 'ill humor', 'ill temper', 'mad', 'slow burn'
              ]

sad_dict = ['unhappy', 'sorrowful', 'dejected', 'depressed', 'downcast', 'miserable', 'down', 'despondent',
            'despairing', 'disconsolate', 'desolate', 'wretched', 'glum', 'gloomy', 'doleful', 'dismal', 'melancholy',
            'mournful', 'woebegone', 'forlorn', 'crestfallen', 'heartbroken', 'inconsolable',
            'informalblue', 'down in the mouth', 'down in the dumps']

fear_dict = ['terror', 'fright', 'fearfulness', 'horror', 'alarm', 'panic', 'agitation', 'trepidation', 'dread',
             'consternation',
             'dismay', 'distress', 'anxiety', 'worry', 'angst', 'unease', 'uneasiness', 'apprehension',
             'apprehensiveness', 'nervousness',
             'nerves', 'perturbation', 'foreboding', 'informalthe creeps', 'the shivers', 'the willies',
             'the heebie-jeebies', 'jitteriness',
             'twitchiness', 'phobia', 'aversion', 'antipathy', 'dread', 'bugbear', 'nightmare', 'horror',
             'terror', 'anxiety', 'neurosis']


happy_dict = ['cheerful', 'cheery', 'merry', 'joyful', 'jovial', 'jolly', 'jocular', 'gleeful', 'carefree', 'delight',
              'smile',
              'grin',
              'in good spirits', 'in a good mood', 'lighthearted', 'pleased', 'contented', 'satisfied', 'gratified',
              'buoyant',
              'radiant', 'sunny',
              'blithe', 'joyous', 'thrilled', 'elated', 'exhilarated', 'ecstatic', 'blissful', 'euphoric', 'overjoyed',
              'exultant', 'rapturous',
              'in seventh heaven', 'on cloud nine', 'walking on air', 'jubilant', 'over the moon',
              'on top of the world', 'tickled pink',
              'on a high', 'bland', 'blissful', 'calm', 'capricious', 'cheerful', 'confident', 'content', 'convinced',
              'dazed', 'delighted',
              'delightful', 'ecstatic', 'elated', 'enchanted', 'epicurean', 'excessive', 'fain', 'fanciful',
              'formidable', 'funny', 'glad',
              'glorious', 'gratified', 'hilarious', 'hopeful', 'humorous', 'joyful', 'jubilant', 'overwhelmed',
              'sanguine', 'sensuous', 'solemn',
              'splendid', 'sprightly', 'spruce', 'sybaritic', 'thrilled', 'voluptuous', 'wry', 'zestful', 'impish',
              'playful', 'prankish', 'roguish',
              'whimsical', 'fastidious' 'heedful']

string_punctuation = "!#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
stop = set(stopwords.words('english'))
tweets = []


def clear_punctuation(s):
    clear_string = ""
    for symbol in s:
        if symbol not in string_punctuation:
            clear_string += symbol
        else:
            clear_string += " "
    return clear_string


def clear_stopwords(s):
    clear_string = []
    for symbol in s:
        if symbol not in stop:
            clear_string.append(symbol)
    return clear_string


def enter_tweet_fbase_line(tweet_data):
    result = {'Anger': 0, 'Sad': 0, 'Depress': 0, 'Happy': 0}
    required_tweet = clear_stopwords(tweet_data.split())
    for words in required_tweet:
        if words in anger_dict:
            result['Anger'] += 1
        if words in anger_dict:
            result['Sad'] += 1
        if words in anger_dict:
            result['Depress'] += 1
        if words in anger_dict:
            result['Happy'] += 1
    # print("Now using Naive Approach Sentiment of tweet is::")
    # print(max(result, key=result.get))
    return max(result, key=result.get)


def accuracy_comparison(training_dict1, testing_dict):
    positive_count = 0
    negative_count = 0
    total_count = 0
    for key_te, value_te in testing_dict.items():
        if key_te in training_dict1.keys():
            value_testing = value_te
            value_training = training_dict1[key_te]
            total_count += 1
            if value_testing == value_training:
                positive_count += 1
            else:
                negative_count += 1
    accuracy = (positive_count/total_count)*100
    negative_accuracy = (negative_count/total_count)*100
    print("positive accuracy")
    print(accuracy)
    print("negative accuracy")
    print(negative_accuracy)
    return accuracy


def data_read():
    test_data = testing_tweets_genartion()
    for tweet in test_data:
        tweets.append(tweet)

# tweet = "!!!!!I#but#acrimony#india#reason#modi#people#culture@"
baseline_output = {}


def run_base_line():
    data_read()
    base_count = 0
    for tweet_data in tweets:
        words = clear_punctuation(tweet_data)
        baseline_output[tweets[base_count]] = enter_tweet_fbase_line(words)
        base_count += 1


def check_base_line_accuracy():
    master_dict = tweets_master_dic_generation()
    run_base_line()
    accuracy_comparison(master_dict, baseline_output)

check_base_line_accuracy()
