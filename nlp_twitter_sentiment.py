from nltk.tokenize import word_tokenize
from nltk.stem.porter import *
import re
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk import RegexpParser
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from nltk.tree import Tree
# from tweet_processing import training_tweets_generation
from tweet_processing import tweets_master_dic_generation
from tweet_processing import testing_tweets_genartion
from tweet_processing import angry_dic_return
from tweet_processing import happy_dic_return
from tweet_processing import fear_dic_return
from tweet_processing import sad_dic_return
from sentiment_baseline import check_base_line_accuracy
import copy


anger_dict_final = []
sad_dict_final = []
happy_dict_final = []
fear_dict_final = []
check_lexical_accuracy = []

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

# { < NNP > +}
# { < NN > < NN >}  # chunk two consecutive nouns
# {<DT>?<JJ>*<NN>}
tweets = []
stop = set(stopwords.words('english'))
stemmer = PorterStemmer()
grammar = """ NP:
                      {<NNS><VBP>}  # combine nns and vbp
                      {<V.*> <TO> <V.*>}  #combine V to V.*
                      {<N.*>(4,)}    # combine nouns
               VB:    {<VBD>*<JJ>*<NNS>}  #combine verb adverb and noun
          """
cp = RegexpParser(grammar)

string_punctuation = "!#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""
regex_str = [
    emoticons_str,
    r'<[^>]+>',  # HTML tags
    r'(?:@[\w_]+)',  # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',  # URLs

    r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
    r'(?:[\w_]+)',  # other words
    r'(?:\S)'  # anything else
]


# linkregex = re.compile(b'<a\s*href=[\'|"](.*?)[\'"].*?>')

tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
# emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)


def add_words_into_bag(first_parm):
    add_result = []
    for word in first_parm:
        hyper_result = get_hypernyms(word)
        add_result.append(word)
        for words in set(hyper_result):
            add_result.append(words)
    return add_result


def convert_tweet_tolist(se_parm):
    tweets_list = []
    for tweet in se_parm:
        after_lexical = apply_lexical(tweet)
        # handling all the abbreviation in our tweet by replacing it with actual words
        handle_abbrev = replace_abbrev_with_words(after_lexical)
        after_pos_tagging = pos_tag(handle_abbrev)
        for words in after_pos_tagging:
            if words[1] == 'VB' or 'VBD' or 'JJ':
                tweets_list.append(words[0])
    return tweets_list


def add_words_into_bag_training_data_anger(data1):
    final_list = []
    result_list = copy.copy(data1)
    anger_list = angry_dic_return()
    anger_list_after = convert_tweet_tolist(anger_list)
    final_list.extend(result_list + anger_list_after)
    return final_list


def add_words_into_bag_training_data_sad(data1):
    final_list = []
    result_list = copy.copy(data1)
    sad_list = sad_dic_return()
    sad_list_after = convert_tweet_tolist(sad_list)
    final_list.extend(result_list + sad_list_after)
    return final_list


def add_words_into_bag_training_data_happy(data1):
    final_list = []
    result_list = copy.copy(data1)
    happy_list = happy_dic_return()
    happy_list_after = convert_tweet_tolist(happy_list)
    final_list.extend(result_list + happy_list_after)
    return final_list


def add_words_into_bag_training_data_fear(data1):
    final_list = []
    result_list = copy.copy(data1)
    fear_list = fear_dic_return()
    fear_list_after = convert_tweet_tolist(fear_list)
    final_list.extend(result_list + fear_list_after)
    return final_list


def create_final_bag_of_words():
    # Adding words into Bags using hypernym
    anger_dict1 = add_words_into_bag(anger_dict)
    anger_dict_final1 = add_words_into_bag_training_data_anger(anger_dict1)
    for words in anger_dict_final1:
        anger_dict_final.append(words)

    sad_dict1 = add_words_into_bag(sad_dict)
    sad_dict_final1 = add_words_into_bag_training_data_sad(sad_dict1)
    for words in sad_dict_final1:
        sad_dict_final.append(words)

    happy_dict1 = add_words_into_bag(happy_dict)
    happy_dict_final1 = add_words_into_bag_training_data_happy(happy_dict1)
    for words in happy_dict_final1:
        happy_dict_final.append(words)

    fear_dict1 = add_words_into_bag(fear_dict)
    fear_dict_final1 = add_words_into_bag_training_data_fear(fear_dict1)
    for words in fear_dict_final1:
        fear_dict_final.append(words)


def find_sentiment(feature_vector):
    result = {'Anger': 0, 'Sad': 0, 'fear': 0, 'Happy': 0}
    # create_final_bag_of_words()
    # print(len(anger_dict_final))
    # print(len(fear_dict_final))
    # print(len(sad_dict_final))
    # print(len(happy_dict_final))
    for words, value in feature_vector.items():
        if words in sad_dict_final:
            result['Sad'] = result.get('Sad') + float(value/len(sad_dict_final))
        if words in happy_dict_final:
            result['Happy'] = result.get('Happy') + float(value/len(happy_dict_final))
        if words in anger_dict_final:
            result['Anger'] = result.get('Anger') + float(value/len(anger_dict_final))
        if words in fear_dict_final:
            result['fear'] = result.get('fear') + float(value/len(fear_dict_final))

    # print("Getting Result for tweet")
    # print(max(result.values()))
    # print(max(result, key=result.get))
    return max(result, key=result.get)


def find_sentiment_for_lexical(feature_vector):
    result = {'Anger': 0, 'Sad': 0, 'fear': 0, 'Happy': 0}
    for words in feature_vector:
        if words in anger_dict_final:
            result['Anger'] = result.get('Anger') + 1
        if words in fear_dict_final:
            result['fear'] = result.get('fear') + 1
        if words in sad_dict_final:
            result['Sad'] = result.get('Sad') + 1
        if words in happy_dict_final:
            result['Happy'] = result.get('Happy') + 1
    # print("Getting Result for tweet after lexical")
    # print(max(result.values()))
    # print(max(result, key=result.get))
    return max(result, key=result.get)


def get_hypernyms(word):
    hypernym_list = []
    if len(wn.synsets(word)) > 1:
        word_synsets = wn.synsets(word)[0]
        hypernym = word_synsets.hypernym_paths()[0]
        for element in hypernym:
            for chunk in element.lemma_names():
                if chunk not in hypernym_list:
                    hypernym_list.append(chunk)
    return hypernym_list


def add_abbrev_dict():
    clean = open('abbreviations.txt').read().replace('\n', ',').lower()
    abb_dict = dict(item.split(":") for item in clean.split(","))
    return abb_dict


def data_read():
    test_data = testing_tweets_genartion()
    for tweet in test_data:
        tweets.append(tweet)

# start replaceTwoOrMore
def replace_two_or_more(s):
    # look for 2 or more repetitions of character and replace with the character itself
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)


def clear_stopwords(s):
    clear_string = []
    for symbol in s:
        if symbol not in stop:
            clear_string.append(symbol)
    return clear_string


def clear_punctuation(s):
    clear_string = ""
    s = re.sub(r"http\S+", "", s)
    for symbol in s:
        if symbol not in string_punctuation:
            clear_string += symbol
        else:
            clear_string += " "
    return clear_string


# will apply lexical feature lemmatization
# replacing two more more repeating words
# removal of URL link
# removal of stop words,
# removal of special character like punctuation
# check if the word stats with an alphabet
# stemming,
# Return Feature Vector


def apply_lexical(tweet):
    feature_vector = []
    # performing tokenization using NLTK
    lemma = word_tokenize((clear_punctuation(tweet)))
    lemma = map(lambda x: x.lower(), lemma)
    lemma = clear_stopwords(lemma)
    # will need to put condition whether we need to apply stemming or not check performance
    # lemma = [stemmer.stem(plural) for plural in lemma]
    for words in lemma:
        # checking and replacing repeating words
        words = replace_two_or_more(words)
        # checking and removing words starts with numbers
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", words)
        if val is None:
            continue
        else:
            feature_vector.append(words.lower())
    return feature_vector


def replace_abbrev_with_words(wordlist):
    abbrev_dict = add_abbrev_dict()
    result_list = []
    for index in range(len(wordlist)):
        if wordlist[index] in abbrev_dict:
            wordlist[index] = abbrev_dict[wordlist[index]]
            for word in word_tokenize(wordlist[index]):
                result_list.append(word)
        else:
            result_list.append(wordlist[index])
    # wordlist = [w.replace(w, abbrev_dict[w]) for w in wordlist if w in abbrev_dict]
    return result_list


def process_data_using_chunking():

    feature_list_using_chunking = []

    for tweet in tweets:
        # print(tweet)
        res_dict = {'noun': 0, 'verb': 0, 'adjective': 0, 'adverb': 0}
        noun_list = []
        lesk_result_list = []
        after_lexical = apply_lexical(tweet)
        # handling all the abbreviation in our tweet by replacing it with actual words
        handle_abbrev = replace_abbrev_with_words(after_lexical)
        after_pos_tagging = pos_tag(handle_abbrev)
        chunking_result = cp.parse(after_pos_tagging)
        # print(chunking_result)
        for sub_tree in chunking_result:
            if type(sub_tree) is Tree:
                if sub_tree.label() == 'NP' or sub_tree.label() == 'VB':
                    for w, t in sub_tree:
                        if 'NP' or 'VB' in t:
                            noun_list.append(w)
        for words in noun_list:
            res_dict['verb'] = len(noun_list)
            if words in res_dict:
                res_dict[words] = res_dict.get(words, 0) + 1
            else:
                res_dict[words] = 1
            m = lesk(after_lexical, words)
            if len(word_tokenize(str(m))) > 1:
                lesk_result = word_tokenize(str(m))[2].split(".")[0].replace('_', ' ')[1:]
                lesk_result_list.append(lesk_result)
                if lesk_result not in res_dict:
                    res_dict[lesk_result] = 1
            hypernyms_words_list1 = get_hypernyms(words)
            if len(hypernyms_words_list1) > 1:
                for element in hypernyms_words_list1:
                    if element not in res_dict:
                        res_dict[element] = 1
        for lesk_words in lesk_result_list:
            # Now getting hypernyms of the word
            hypernyms_words_list2 = get_hypernyms(lesk_words)
            if len(hypernyms_words_list2) > 1:
                for element in hypernyms_words_list2:
                    if element not in res_dict:
                        res_dict[element] = 1
        feature_list_using_chunking.append(res_dict)

    return feature_list_using_chunking


def process_data_using_pos_tagging_for_user(user_list):
    feature_list = []
    lesk_result_list = []
    for tweet in user_list:
        res_dict1 = {'noun': 0, 'verb': 0, 'adjective': 0, 'adverb': 0}
        after_lexical = apply_lexical(tweet)
        # handling all the abbreviation in our tweet by replacing it with actual words
        handle_abbrev = replace_abbrev_with_words(after_lexical)
        after_pos_tagging = pos_tag(handle_abbrev)
        # print(after_pos_tagging)
        # check accuracy after applying lexical feature
        check_lexical_accuracy.append(handle_abbrev)
        for words in after_pos_tagging:
            if words[1] == 'NN' or 'NNS' or 'NNP':
                res_dict1['noun'] += 1
                if words[0] in res_dict1:
                    res_dict1[words[0]] = res_dict1.get(words[0], 0) + 1
                else:
                    res_dict1[words[0]] = 1
                m = lesk(after_lexical, words[0])
                if len(word_tokenize(str(m))) > 1:
                    lesk_result = word_tokenize(str(m))[2].split(".")[0].replace('_', ' ')[1:]
                    lesk_result_list.append(lesk_result)
                    if lesk_result not in res_dict1:
                        res_dict1[lesk_result] = 1
                # Now getting hypernyms of the word
                hypernyms_words_list = get_hypernyms(words[0])
                if len(hypernyms_words_list) > 1:
                    for element in hypernyms_words_list:
                        if element not in res_dict1:
                            res_dict1[element] = 1
            elif words[1] == 'VB' or 'VBD':
                res_dict1['verb'] += 1
                if words[0] in res_dict1:
                    res_dict1[words[0]] = res_dict1.get(words[0], 0) + 1
                else:
                    res_dict1[words[0]] = 1
                # print("Result of Lesk Algorithm for word", words[0])
                m = lesk(after_lexical, words[0])
                if len(word_tokenize(str(m))) > 1:
                    lesk_result = word_tokenize(str(m))[2].split(".")[0].replace('_', ' ')[1:]
                    lesk_result_list.append(lesk_result)
                    if lesk_result not in res_dict1:
                        res_dict1[lesk_result] = 1
                # Now getting hypernyms of the word
                hypernyms_words_list = get_hypernyms(words[0])
                if len(hypernyms_words_list) > 1:
                    for element in hypernyms_words_list:
                        if element not in res_dict1:
                            res_dict1[element] = 1
            elif words[1] == 'JJ':
                res_dict1['adjective'] += 1
                if words[0] in dict:
                    res_dict1[words[0]] = res_dict1.get(words[0], 0) + 1
                else:
                    res_dict1[words[0]] = 1
                m = lesk(after_lexical, words[0])
                if len(word_tokenize(str(m))) > 1:
                    lesk_result = word_tokenize(str(m))[2].split(".")[0].replace('_', ' ')[1:]
                    lesk_result_list.append(lesk_result)
                    if lesk_result not in res_dict1:
                        res_dict1[lesk_result] = 1
                # Now getting hypernyms of the word
                hypernyms_words_list = get_hypernyms(words[0])
                if len(hypernyms_words_list) > 1:
                    for element in hypernyms_words_list:
                        if element not in res_dict1:
                            res_dict1[element] = 1
            elif words[1] == 'RB':
                res_dict1['adverb'] += 1
                if words[0] in res_dict1:
                    res_dict1[words[0]] = res_dict1.get(words[0], 0) + 1
                else:
                    res_dict1[words[0]] = 1
                m = lesk(after_lexical, words[0])
                if len(word_tokenize(str(m))) > 1:
                    lesk_result = word_tokenize(str(m))[2].split(".")[0].replace('_', ' ')[1:]
                    lesk_result_list.append(lesk_result)
                    if lesk_result not in res_dict1:
                        res_dict1[lesk_result] = 1
                # Now getting hypernyms of the word
                hypernyms_words_list = get_hypernyms(words[0])
                if len(hypernyms_words_list) > 1:
                    for element in hypernyms_words_list:
                        if element not in res_dict1:
                            res_dict1[element] = 1

        for lesk_words in lesk_result_list:
            # Now getting hypernyms of the word
            hypernyms_words_list2 = get_hypernyms(lesk_words)
            if len(hypernyms_words_list2) > 1:
                for element in hypernyms_words_list2:
                    if element not in res_dict1:
                        res_dict1[element] = 1
        feature_list.append(res_dict1)
    return feature_list


def process_data_using_pos_tagging():
    feature_list = []
    lesk_result_list = []
    for tweet in tweets:
        res_dict1 = {'noun': 0, 'verb': 0, 'adjective': 0, 'adverb': 0}
        after_lexical = apply_lexical(tweet)
        # handling all the abbreviation in our tweet by replacing it with actual words
        handle_abbrev = replace_abbrev_with_words(after_lexical)
        after_pos_tagging = pos_tag(handle_abbrev)
        # print(after_pos_tagging)
        # check accuracy after applying lexical feature
        check_lexical_accuracy.append(handle_abbrev)
        for words in after_pos_tagging:
            if words[1] == 'NN' or 'NNS' or 'NNP':
                res_dict1['noun'] += 1
                if words[0] in res_dict1:
                    res_dict1[words[0]] = res_dict1.get(words[0], 0) + 1
                else:
                    res_dict1[words[0]] = 1
                m = lesk(after_lexical, words[0])
                if len(word_tokenize(str(m))) > 1:
                    lesk_result = word_tokenize(str(m))[2].split(".")[0].replace('_', ' ')[1:]
                    lesk_result_list.append(lesk_result)
                    if lesk_result not in res_dict1:
                        res_dict1[lesk_result] = 1
                # Now getting hypernyms of the word
                hypernyms_words_list = get_hypernyms(words[0])
                if len(hypernyms_words_list) > 1:
                    for element in hypernyms_words_list:
                        if element not in res_dict1:
                            res_dict1[element] = 1
            elif words[1] == 'VB' or 'VBD':
                res_dict1['verb'] += 1
                if words[0] in res_dict1:
                    res_dict1[words[0]] = res_dict1.get(words[0], 0) + 1
                else:
                    res_dict1[words[0]] = 1
                # print("Result of Lesk Algorithm for word", words[0])
                m = lesk(after_lexical, words[0])
                if len(word_tokenize(str(m))) > 1:
                    lesk_result = word_tokenize(str(m))[2].split(".")[0].replace('_', ' ')[1:]
                    lesk_result_list.append(lesk_result)
                    if lesk_result not in res_dict1:
                        res_dict1[lesk_result] = 1
                # Now getting hypernyms of the word
                hypernyms_words_list = get_hypernyms(words[0])
                if len(hypernyms_words_list) > 1:
                    for element in hypernyms_words_list:
                        if element not in res_dict1:
                            res_dict1[element] = 1
            elif words[1] == 'JJ':
                res_dict1['adjective'] += 1
                if words[0] in dict:
                    res_dict1[words[0]] = res_dict1.get(words[0], 0) + 1
                else:
                    res_dict1[words[0]] = 1
                m = lesk(after_lexical, words[0])
                if len(word_tokenize(str(m))) > 1:
                    lesk_result = word_tokenize(str(m))[2].split(".")[0].replace('_', ' ')[1:]
                    lesk_result_list.append(lesk_result)
                    if lesk_result not in res_dict1:
                        res_dict1[lesk_result] = 1
                # Now getting hypernyms of the word
                hypernyms_words_list = get_hypernyms(words[0])
                if len(hypernyms_words_list) > 1:
                    for element in hypernyms_words_list:
                        if element not in res_dict1:
                            res_dict1[element] = 1
            elif words[1] == 'RB':
                res_dict1['adverb'] += 1
                if words[0] in res_dict1:
                    res_dict1[words[0]] = res_dict1.get(words[0], 0) + 1
                else:
                    res_dict1[words[0]] = 1
                m = lesk(after_lexical, words[0])
                if len(word_tokenize(str(m))) > 1:
                    lesk_result = word_tokenize(str(m))[2].split(".")[0].replace('_', ' ')[1:]
                    lesk_result_list.append(lesk_result)
                    if lesk_result not in res_dict1:
                        res_dict1[lesk_result] = 1
                # Now getting hypernyms of the word
                hypernyms_words_list = get_hypernyms(words[0])
                if len(hypernyms_words_list) > 1:
                    for element in hypernyms_words_list:
                        if element not in res_dict1:
                            res_dict1[element] = 1

        for lesk_words in lesk_result_list:
            # Now getting hypernyms of the word
            hypernyms_words_list2 = get_hypernyms(lesk_words)
            if len(hypernyms_words_list2) > 1:
                for element in hypernyms_words_list2:
                    if element not in res_dict1:
                        res_dict1[element] = 1
        feature_list.append(res_dict1)
    return feature_list


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
    print(negative_accuracy)
    print("negative accuracy")
    print(accuracy)
    return accuracy

training_dict = tweets_master_dic_generation()
data_read()
create_final_bag_of_words()
featureList = process_data_using_pos_tagging()
feature_list_chunking = process_data_using_chunking()

TestOutput = {}
# test = ['love','fear', 'deep', 'happy']
# find_sentiment(test)

# baseline
print("****************************")
print("Accuracy using baseline using Bag of words")
check_base_line_accuracy()
print("*****************************")

lexical_output = {}
lexical_count = 0
for item in check_lexical_accuracy:
    # print(tweets[lexical_count])
    # print(item)
    lexical_output[tweets[lexical_count]] = find_sentiment_for_lexical(item)
    lexical_count += 1
print("****************************")
print("Accuracy after Lexical Analysis")
print("****************************")
accuracy_comparison(training_dict, lexical_output)
# final_run
count = 0
for i in featureList:
    # print("*********Tweet**********")
    # print(tweets[count])
    # print("Feature Vector build after applying all the features")
    # print(i)
    # print("*******************")
    # print("Applying all six feature sentiment for this tweet will be")
    # find_sentiment(i)
    TestOutput[tweets[count]] = find_sentiment(i)
    count += 1
print("****************************")
print("We are getting an accuracy After applying All the six feature")
accuracy_comparison(training_dict, TestOutput)
print("****************************")

chunk_count = 0
chunk_output = {}
for item in feature_list_chunking:
    # print(tweets[chunk_count])
    # print(item)
    chunk_output[tweets[chunk_count]] = find_sentiment(item)
    chunk_count += 1

print("****************************")
print("We are getting an accuracy After applying chunking the six feature")
accuracy_comparison(training_dict, chunk_output)
print("****************************")

required_user_list = []
tweet1 = " @hello I am very happy to see you #event #birthday"
tweet2 = " @hello I am very sad to see you #event #birthday"
required_user_list.append(tweet1)
required_user_list.append(tweet2)
user_output = {}

user_training_dict = {}

user_training_dict[tweet1] = "Happy"
user_training_dict[tweet2] = "Sad"

print("Enter your input tweet")
tweet3 = input()
print("Do you want to see accuracy if yes then press 1 or else 2 for sentiment")
option = input()
if option == "1":
    print("Enter your sentiment which you think it should be :Possible sentiment (Anger, Sad, Happy, fear)")
    sentiment = input()
    user_training_dict[tweet3] = sentiment
elif option == "2":
    print("Your sentiment for tweet is:")
else:
    print("not a valid input")

user_count = 0
required_user_list.append(tweet3)

user_feature_vector = process_data_using_pos_tagging_for_user(required_user_list)

# print(user_feature_vector)
print("Accuracy after user tweets sentiment")
for user_data in user_feature_vector:
    # print("User entered Tweet")
    # print(required_user_list[user_count])
    user_output[required_user_list[user_count]] = find_sentiment(user_data)
    user_count += 1

accuracy_comparison(user_training_dict, user_output)
