import os, json
from tweepy.streaming import json
angry_tweets_dict = []
sad_tweets_dict = []
fear_tweets_dict = []
happy_tweets_dict = []

def training_tweets_generation():
    training_tweets = []
    for filename in os.listdir("tweets"):
        data_json = open(filename, mode='r').read()  # reads in the JSON file into Python as a string
        data_python = json.loads(data_json)
        for line in data_python:
            training_tweets.append(line.get('text').encode('unicode_escape').decode('utf-8'))

    return set(training_tweets)

def tweets_master_dic_generation():
    master_dict = {}
    for filename in os.listdir("tweets"):
        # print(filename)
        data_json = open(filename, mode='r').read()  # reads in the JSON file into Python as a string
        data_python = json.loads(data_json)
        if "angry" in filename:
            # print("inside anger")
            for line in data_python:
                angry_tweets_dict.append(line.get('text').encode('unicode_escape').decode('utf-8'))
                master_dict[line.get('text').encode('unicode_escape').decode('utf-8')] = "Anger"
        if "happy" in filename:
            # print("inside happy")
            for line in data_python:
                happy_tweets_dict.append(line.get('text').encode('unicode_escape').decode('utf-8'))
                master_dict[line.get('text').encode('unicode_escape').decode('utf-8')] = "Happy"
        if "sad" in filename:
            # print("inside sad")
            for line in data_python:
                sad_tweets_dict.append(line.get('text').encode('unicode_escape').decode('utf-8'))
                master_dict[line.get('text').encode('unicode_escape').decode('utf-8')] = "Sad"
        if "fear" in filename:
            # print("insdie fear")
            for line in data_python:
                fear_tweets_dict.append(line.get('text').encode('unicode_escape').decode('utf-8'))
                master_dict[line.get('text').encode('unicode_escape').decode('utf-8')] = "fear"

    return master_dict


def testing_tweets_genartion():
    anger_count = []
    happy_count = []
    sad_count = []
    fear_count = []
    testing_tweets = []
    all_tweets = tweets_master_dic_generation()
    for key, value in all_tweets.items():
            if value == 'angry':
                if len(anger_count) < 5:
                    anger_count.append(key)
                    set(anger_count)
            elif value == 'sad':
                if len(sad_count) < 5:
                    sad_count.append(key)
                    set(sad_count)
            elif value == 'fear':
                if len(fear_count) < 5:
                    fear_count.append(key)
                    set(fear_count)
            elif value == 'happy':
                if len(happy_count) < 5:
                    happy_count.append(key)
                    set(happy_count)
    # print(len(anger_count))
    # print(len(sad_count))
    # print(len(fear_count))
    # print(len(happy_count))
    testing_tweets.extend(anger_count+sad_count+fear_count+happy_count)
    return set(testing_tweets)


def angry_dic_return():
    tweets_master_dic_generation()
    return set(angry_tweets_dict)


def happy_dic_return():
    tweets_master_dic_generation()
    return set(happy_tweets_dict)


def sad_dic_return():
    tweets_master_dic_generation()
    return set(sad_tweets_dict)


def fear_dic_return():
    tweets_master_dic_generation()
    return set(fear_tweets_dict)

# tweets_master_dic_generation()

# print(len(set(angry_dic_return())))
# print(len(set(happy_dic_return())))
# print(len(set(sad_dic_return())))
# print(len(set(fear_dic_return())))

# print(len(testing_tweets_genartion()))
# print(len(training_tweets_generation()))
# print(len(tweets_master_dic_generation()))
