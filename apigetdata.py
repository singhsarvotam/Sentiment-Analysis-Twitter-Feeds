import time
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import io
ckey = 'vNu3DgMteZJewvx9RpT9U7XKJ'
consumer_secret = 'yh4bJIEfzWB2fd7y3Zolatg9X7rEL08fLBW049vkvzisltqwg8'
access_token_key = '86056153-R8j8RPgMZSKFrfBdWGJIgBgDCyMPdvfSY2TebxJSU'
access_token_secret = 'vI0az6wdtCh0Lur3Pzn1mSM1a5mVpSLZrE9C0Y72bctgp'

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


class listener(StreamListener):

    def __init__(self, start_time, time_limit=60):
        self.time = start_time
        self.limit = time_limit
        self.tweet_data = []

    def on_data(self, data):
        saveFile = io.open('sad_tweets.json', 'a', encoding='utf-8')
        while (time.time() - self.time) < self.limit:
            try:
                self.tweet_data.append(data)
                return True
            except BaseException:
                print('failed ondata')
                time.sleep(5)
                pass
        saveFile = io.open('sad_tweets.json', 'w', encoding='utf-8')
        saveFile.write(u'[\n')
        saveFile.write(','.join(self.tweet_data))
        saveFile.write(u'\n]')
        saveFile.close()
        exit()

    def on_error(self, status):
        print(status)

    def on_disconnect(self, notice):
        print('bye')


# Beginning of the specific code
start_time = time.time()  # grabs the system time
keyword_list = sad_dict  # track list
auth = OAuthHandler(ckey, consumer_secret)  # OAuth object
auth.set_access_token(access_token_key, access_token_secret)


twitterStream = Stream(auth, listener(start_time, time_limit=20))  # initialize Stream object with a time out limit
twitterStream.filter(track=keyword_list, languages=['en'])  # call the filter method to run the Stream Listener
