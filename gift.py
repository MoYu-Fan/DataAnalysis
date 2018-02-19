# -*-coding:utf-8 -*-
# coding:utf-8
# coding:gb2312
# __author__= 'admin'

import time
from wordcloud import WordCloud
import matplotlib.pyplot as plt


class Gift():

    def __init__(self):
        pass

    def gift_antigravity(self):
        print '请在命令行输入  import antigravity   这句话。'

    def gift_love(self):
        words = raw_input('请输入你最想说的话：')
        for word in words.split():
            print '\n'.join([''.join([(word[(x - y) % len(word)] if ((x * 0.05) ** 2 + (y * 0.1) ** 2 - 1) ** 3 - (
                    x * 0.05) ** 2 * (y * 0.1) ** 3 <= 0 else ' ') for x in range(-30, 30)]) for y in
                             range(12, -12, -1)])
            time.sleep(1.4)

    def words_for_wiki(self):
        filename = 'wiki.txt'
        with open(filename) as f:
            mytext = f.read()
            wordcloud = WordCloud().generate(mytext)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')