import nltk
nltk.download('punkt')

from nltk import word_tokenize,sent_tokenize
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import pickle
import pymongo
from bson.json_util import dumps

host_name = "localhost"
port = "27017"

conn_str = {
    "local" : f"mongodb://{host_name}:{port}/"
}

client = pymongo.MongoClient(conn_str["local"])

db_name = "netflix_data_final"
db = client[db_name]

# PART 3 - run bot to answer questions

with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)

except:
    words = []
    labels = []
    docs_x = []
    docs_y = []
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))
    labels = sorted(labels)

    training = []
    output = []
    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")

try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)


def chat():
    with open("intents.json") as file:
        data = json.load(file)

    print("Start talking with the bot! Type 'help' to see the types of questions you can ask.")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        result = model.predict([bag_of_words(inp, words)])[0]
        result_index = np.argmax(result)
        tag = labels[result_index]

        if result[result_index] > 0.7:
            for tg in data["intents"]:

                if tag == "question1":
                    for movie in db.kaggle_movies_data.find(filter={'RELEASE_YEAR': 2017}).sort("SCORE", -1).limit(3):
                        title = movie["TITLE"]
                        theGENRE = movie["MAIN_GENRE"]
                        score = movie["SCORE"]
                    answer = "The genre of the top best scoring movie in 2017 was " + theGENRE

                if tag == "question2":
                    for show in db.kaggle_shows_data.find().sort("NUMBER_OF_SEASONS", -1).limit(1):
                        title = show["TITLE"]
                        seasons = show["NUMBER_OF_SEASONS"]
                    answer = "'" + title + "'" + " with " + str(seasons) + " seasons"

                if tag == "question3":
                    for show in db.kaggle_shows_data.find().sort("score", 1).limit(1):
                        title = show["TITLE"]
                        thescore = show["SCORE"]
                    answer = "'" + title + "'" + " with a score of " + str(thescore)

                if tag == "question4":
                    for show in db.kaggle_shows_data.find().sort("DURATION", 1).limit(-1):
                        title = show["TITLE"]
                        time = show["DURATION"]
                    answer = "'" + title + "'" + " with an episode duration of " + str(time) + " minutes"

                if tag == "question5":
                    answer = []
                    thedata = db.kaggle_movies_data.find(filter={'RELEASE_YEAR': 2010}, sort=list({'SCORE': -1}.items()),
                                                      limit=5)

                    dump_data = dumps(list(thedata))
                    my_list = json.loads(dump_data)

                    t1 = str(my_list[0]["TITLE"])
                    s1 = str(my_list[0]["SCORE"])
                    m1 = t1 + " (" + s1 + ")"
                    answer.append(m1)

                    t2 = str(my_list[1]["TITLE"])
                    s2 = str(my_list[1]["SCORE"])
                    m2 = t2 + " (" + s2 + ")"
                    answer.append(m2)

                    t3 = str(my_list[2]["TITLE"])
                    s3 = str(my_list[2]["SCORE"])
                    m3 = t3 + " (" + s3 + ")"
                    answer.append(m3)

                    t4 = str(my_list[3]["TITLE"])
                    s4 = str(my_list[3]["SCORE"])
                    m4 = t4 + " (" + s4 + ")"
                    answer.append(m4)

                    t5 = str(my_list[4]["TITLE"])
                    s5 = str(my_list[4]["SCORE"])
                    m5 = t5 + " (" + s5 + ")"
                    answer.append(m5)

                if tag == "question6":
                    com = db.kaggle_movies_data.aggregate([{"$group": {'_id': "$MAIN_GENRE", 'count': {'$sum': 1}}}])
                    sum_com = list(com)
                    answer = str([obj for obj in sum_com if obj['_id'] == 'comedy'][0]['count']) + " comedy movies"

                if tag == "question7":
                    for movie in db.kaggle_movies_data.find().sort("SCORE", 1).limit(-1):
                        title = movie["TITLE"]
                        thescore = movie["SCORE"]
                        theyear = movie["RELEASE_YEAR"]
                    answer = title + " (" + str(thescore) + ") released in " + str(theyear)

                if tag == "question8":
                    list1 = []
                    for movie in db.kaggle_raw_data.find(filter={'type': 'MOVIE', 'release_year': 2020, 'age_certification': 'PG'}):
                        list1.append(movie)
                    answer = str(len(list1)) + " were rated 'PG' in 2020"

                if tag == "question9":
                    list2 = []
                    for movie in db.kaggle_raw_data.find(filter={'type': 'MOVIE', 'age_certification': 'R'}):
                        list2.append(movie)
                    answer = str(len(list2)) + " movies were rated 'R'"

                if tag == "question10":
                    answer = []
                    for movie in db.kaggle_raw_data.find(
                            filter={'type': 'MOVIE', 'age_certification': 'R', 'release_year': 2021}).limit(10):
                        answer.append(movie["title"])

            print(answer)



        elif inp.lower() == "help":

            print("1) What is the genre of the top best scoring movie in 2017?")  # from movies data
            print("2) Which show had the most number of seasons?")  # from shows data
            print("3) Which show scored the highest?")  # from shows data
            print("4) What is the duration of the shortest episode for a show?")  # from shows data
            print("5) What are the top 5 highest scored movies in 2010?")  # from movies data
            print("6) How many comedy movies are there?")  # from movies data
            print("7) What is the lowest scoring movie?")  # from movies data
            print("8) How many movies were rated 'PG' in 2020?")  # from raw titles data
            print("9) How many movies were rated 'R'?")  # from raw titles data
            print("10) What are 10 rated 'R' movies from 2021?")  # from raw titles data

        else:
            print("I didn't get that. Can you explain or try again.")


chat()
