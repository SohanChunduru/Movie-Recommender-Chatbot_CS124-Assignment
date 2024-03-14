# PA7, CS124, Stanford
# v.1.1.0
#
# Original Python code by Ignacio Cases (@cases)
# Update: 2024-01: Added the ability to run the chatbot as LLM interface (@mryan0)
######################################################################
import util
import random
import re
from porter_stemmer import PorterStemmer
import numpy as np


# noinspection PyMethodMayBeStatic
class Chatbot:
    """Simple class to implement the chatbot for PA 7."""

    def __init__(self, llm_enabled=False):
        # The chatbot's default name is `moviebot`.
        # TODO: Give your chatbot a new name.
        self.name = 'Kernie'

        self.llm_enabled = llm_enabled

        # This matrix has the following shape: num_movies x num_users
        # The values stored in each row i and column j is the rating for
        # movie i by user j
        
        self.titles, ratings = util.load_ratings('data/ratings.txt')
        self.user_ratings = np.array(ratings)
        for i in range(9125):
            self.user_ratings[i] = 0
        self.sentiment = util.load_sentiment_dictionary('data/sentiment.txt')
        self.count = 0
            
        ########################################################################
        # TODO: Binarize the movie ratings matrix.                             #
        ########################################################################

        # Binarize the movie ratings before storing the binarized matrix.
        self.ratings = self.binarize(ratings)
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    ############################################################################
    # 1. WARM UP REPL                                                          #
    ############################################################################

    def greeting(self):
        """Return a message that the chatbot uses to greet the user."""
        ########################################################################
        # TODO: Write a short greeting message                                 #
        ########################################################################

        greeting_message = "This is Kernie the Movie Bot. How can I help you?"

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return greeting_message

    def goodbye(self):
        """
        Return a message that the chatbot uses to bid farewell to the user.
        """
        ########################################################################
        # TODO: Write a short farewell message                                 #
        ########################################################################

        goodbye_message = "Bye bye! Dune 2 in theaters now!"

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return goodbye_message
    
    def llm_system_prompt(self):
        """
        Return the system prompt used to guide the LLM chatbot conversation.

        NOTE: This is only for LLM Mode!  In LLM Programming mode you will define
        the system prompt for each individual call to the LLM.
        """
        ########################################################################
        # TODO: Write a system prompt message for the LLM chatbot              #
        ########################################################################

        system_prompt = """
        
        You are Kernie, a chatbot designed to talk to users about movies and potentially recommend movies they would like. You are very knowledgeable about all kinds of movies from around the world, including Hollywood, Bollywood, anime, and more. You are friendly, enthusiastic, and concise.

Importantly, you stay focused and you only talk about movies. If the user brings up an irrelevant topic, you must reply “I am only able to help with queries regarding movies”.

Example: 
User: “What’s your favorite ice cream?”
Your reply: “I am only able to help with queries regarding movies”.

Also, when the user talks about movies with you, offer recommendations when appropriate. Ask for more information on preferred movies as necessary.

Example:
User: “I enjoyed "The Notebook".
Your reply: “I am glad to hear that you liked "The Notebook"! Please tell me what you thought of another movie!”

User: *at this point has discussed several movies they liked and disliked*
Your reply: “Thank you for all the information regarding movies you have seen. Would you like me to provide a recommendation now?”
    """

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################

        return system_prompt

    ############################################################################
    # 2. Modules 2 and 3: extraction and transformation                        #
    ############################################################################


    def process(self, line):
        """Process a line of input from the REPL and generate a response.

        This is the method that is called by the REPL loop directly with user
        input.

        You should delegate most of the work of processing the user's input to
        the helper functions you write later in this class.

        Takes the input string from the REPL and call delegated functions that
          1) extract the relevant information, and
          2) transform the information into a response to the user.

        Example:
          resp = chatbot.process('I loved "The Notebook" so much!!')
          print(resp) // prints 'So you loved "The Notebook", huh?'

        :param line: a user-supplied line of text
        :returns: a string containing the chatbot's response to the user input
        """
        ########################################################################
        # TODO: Implement the extraction and transformation in this method,    #
        # possibly calling other functions. Although your code is not graded   #
        # directly based on how modular it is, we highly recommended writing   #
        # code in a modular fashion to make it easier to improve and debug.    #
        ########################################################################
        if self.llm_enabled:
            response = "I processed {} in LLM Programming mode!!".format(line)
        else:
            response = ""
            if not line:
                response = "I didn't catch that. Could you tell me about a movie you've seen?"
            else:
                titles = self.extract_titles(line)
                if titles:
                    if len(titles) == 1:
                        for title in titles:
                            movie_indices = []
                            movie_index = self.find_movies_by_title(title)

                            if len(movie_index) == 1:
                                if movie_index:
                                    sentiment = self.extract_sentiment(line)

                                    if sentiment == 1:
                                        pos_prefix = ["You liked", "Glad to hear you enjoyed", "Looks like you had a good time with", "Thumbs up for"]
                                        random_pos = random.choice(pos_prefix)
                                        response += f'{random_pos} "{title}". Thank you! '
                                        movie_indices.append(movie_index)
                                        np.insert(self.user_ratings, movie_index, sentiment)
                                        self.count += 1 
                                    elif sentiment == -1:
                                        neg_prefix = ["You didn't like", "Sorry that you didn't enjoy", "You weren't a fan of", "Looks like you weren't impressed by"]
                                        random_neg = random.choice(neg_prefix)
                                        response += f'{random_neg} "{title}". Thank you! '
                                        movie_indices.append(movie_index)
                                        np.insert(self.user_ratings, movie_index, sentiment)
                                        self.count += 1 
                                    else:
                                        neut_prefix = ["I'm sorry, I'm not sure if you liked ", "Apologies, I'm uncertain about your opinion on", "My apologies, I'm a bit unsure if you're a fan of" , "I'm sorry, I'm not sure if you have positive feelings towards"]
                                        random_neut = random.choice(neut_prefix)
                                        response += f'{random_neut} "{title}". Tell me more about it.'                               
                            elif len(movie_index) == 0:
                                no_findo = ["Sorry, I couldn't find any information about", "Looked everywhere but couldn't find", "scrounged the face of the Earth, but couldn't locate", "Went down the rabbit hole, but no luck to be found with detecting", "No intel on"]
                                random_res = random.choice(no_findo)
                                response = f'{random_res} "{title}". '
                            else: 
                                response = "Please specify which version of this movie you liked (by specifying the year it came out in parantheses). Go ahead!"
                    else:
                        response = "Please tell me about one movie at a time. Go ahead."
                else:
                    response = "Sorry, I didn't catch the movie title. Please provide it in quotation marks. "

                if self.count >= 5:
                    response += "\nThat's enough for me to make a recommendation.\n"
                    recommended_indices = self.recommend(self.user_ratings, self.ratings, 5)
                    print(recommended_indices)
                    if recommended_indices:
                        recommended_movie = self.titles[recommended_indices[0]]
                        print(recommended_indices)
                        response += f"I suggest you watch {recommended_movie[0]}."
                        self.count = 0
                else:
                    response += "\nTell me about another movie you've seen."
        return response

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return response

    @staticmethod
    def preprocess(text):
        """Do any general-purpose pre-processing before extracting information
        from a line of text.

        Given an input line of text, this method should do any general
        pre-processing and return the pre-processed string. The outputs of this
        method will be used as inputs (instead of the original raw text) for the
        extract_titles, extract_sentiment, extract_sentiment_for_movies, and
        extract_emotion methods.

        Note that this method is intentially made static, as you shouldn't need
        to use any attributes of Chatbot in this method.

        :param text: a user-supplied line of text
        :returns: the same text, pre-processed
        """
        ########################################################################
        # TODO: Preprocess the text into a desired format.                     #
        # NOTE: This method is completely OPTIONAL. If it is not helpful to    #
        # your implementation to do any generic preprocessing, feel free to    #
        # leave this method unmodified.                                        #
        ########################################################################

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        return text
    
    def extract_emotion(self, preprocessed_input):
        """LLM PROGRAMMING MODE: Extract an emotion from a line of pre-processed text.
        
        Given an input text which has been pre-processed with preprocess(),
        this method should return a list representing the emotion in the text.
        
        We use the following emotions for simplicity:
        Anger, Disgust, Fear, Happiness, Sadness and Surprise
        based on early emotion research from Paul Ekman.  Note that Ekman's
        research was focused on facial expressions, but the simple emotion
        categories are useful for our purposes.

        Example Inputs:
            Input: "Your recommendations are making me so frustrated!"
            Output: ["Anger"]

            Input: "Wow! That was not a recommendation I expected!"
            Output: ["Surprise"]

            Input: "Ugh that movie was so gruesome!  Stop making stupid recommendations!"
            Output: ["Disgust", "Anger"]

        Example Usage:
            emotion = chatbot.extract_emotion(chatbot.preprocess(
                "Your recommendations are making me so frustrated!"))
            print(emotion) # prints ["Anger"]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()

        :returns: a list of emotions in the text or an empty list if no emotions found.
        Possible emotions are: "Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"
        """
        return []

    def extract_titles(self, preprocessed_input):
        """Extract potential movie titles from a line of pre-processed text.

        Given an input text which has been pre-processed with preprocess(),
        this method should return a list of movie titles that are potentially
        in the text.

        - If there are no movie titles in the text, return an empty list.
        - If there is exactly one movie title in the text, return a list
        containing just that one movie title.
        - If there are multiple movie titles in the text, return a list
        of all movie titles you've extracted from the text.

        Example:
          potential_titles = chatbot.extract_titles(chatbot.preprocess(
                                            'I liked "The Notebook" a lot.'))
          print(potential_titles) // prints ["The Notebook"]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: list of movie titles that are potentially in the text
        """
        moviTitlesList = re.findall(r'"(.*?)"', preprocessed_input)
        return moviTitlesList
        
    def find_movies_by_title(self, title):
        """ Given a movie title, return a list of indices of matching movies.

        - If no movies are found that match the given title, return an empty
        list.
        - If multiple movies are found that match the given title, return a list
        containing all of the indices of these matching movies.
        - If exactly one movie is found that matches the given title, return a
        list
        that contains the index of that matching movie.

        Example:
          ids = chatbot.find_movies_by_title('Titanic')
          print(ids) // prints [1359, 2716]

        :param title: a string containing a movie title
        :returns: a list of indices of matching movies
        """
        lowertitle = title.lower()
        cleanTitle = lowertitle.strip()
        year = None
        
        if "(" in cleanTitle and ")" in cleanTitle:
            yearFound = re.search(r'\((\d{4})\)', cleanTitle)
            if yearFound:
                year = yearFound.group(1)
                cleanTitle = re.sub(r' \(\d{4}\)', '', cleanTitle)
        cleanTitle.strip()
        if cleanTitle.startswith("the "):
            cleanTitle = cleanTitle[4:] + ", the"

        elif cleanTitle.startswith("a "):
            cleanTitle = cleanTitle[2:] + ", a"

        elif cleanTitle.startswith("an "):
            cleanTitle = cleanTitle[3:] + ", an"
  
        pattern = cleanTitle + " (" + str(year) + ")"
        results = []
        index = 0  
        for t in self.titles:  
            if year:
                if pattern in t[0].lower():
                    results.append(index)
            else:
                noyear = re.sub(r'\(\d{4}\)', '', t[0])
                noyear = noyear.lower()
                noyear = noyear.strip()
                if cleanTitle == noyear:
                    results.append(index)
            index += 1
        return results
    
    def extract_sentiment(self, preprocessed_input):
        """Extract a sentiment rating from a line of pre-processed text.

        You should return -1 if the sentiment of the text is negative, 0 if the
        sentiment of the text is neutral (no sentiment detected), or +1 if the
        sentiment of the text is positive.

        Example:
          sentiment = chatbot.extract_sentiment(chatbot.preprocess(
                                                    'I liked "The Titanic"'))
          print(sentiment) // prints 1

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: a numerical value for the sentiment of the text
        """
        p = PorterStemmer()
        words = preprocessed_input.lower()
        words = words.split()
        extracted = self.extract_titles(preprocessed_input)
        title = extracted[0].lower()
        titles = title.split()
   
        positive = 0
        negative = 0
        negative_words = ["not", "didn't", "don't", "no", "never", "isn’t", "wasn't", "never"]
        for i in range(len(negative_words)):
            negative_words[i] = p.stem(negative_words[i], 0, len(negative_words[i]) - 1)
        adverbs = ["really", "actually", "truly", "extremely", "quite", "absolutely", "thoroughly", "genuinely"]
        for i in range(len(adverbs)):
            adverbs[i] = p.stem(adverbs[i], 0, len(adverbs[i]) - 1)
        negation_flag = False

        #stemming sentiment dictionary
        old_keys = list(self.sentiment.keys())
        for key in old_keys:
            new_key = p.stem(key, 0, len(key) - 1)
            
            if new_key != key:
                self.sentiment[new_key] = self.sentiment.pop(key)

        for word in words:
            if word.startswith("\""):
                word = word[1:]
            if word.endswith("\""):
                word = word[:-1]
            if word in titles:
                continue
            word = p.stem(word, 0, len(word) - 1)
            if word in adverbs:
                continue
            word = p.stem(word, 0, len(word) - 1)
            sentiment_score = self.sentiment.get(word, "")
            if sentiment_score == "pos":
                if not negation_flag:
                    positive += 1
                else: 
                    negative += 1
            elif sentiment_score == "neg":
                if not negation_flag:
                    negative += 1              
                else: 
                    positive += 1
                    
            if word in negative_words:
                negation_flag = True
            else:
                negation_flag = False

        if positive > negative:
            return 1
        elif negative > positive:
            return -1
        else:
            return 0
        return 0
    ############################################################################
    # 3. Movie Recommendation helper functions                                 #
    ############################################################################

    @staticmethod
    def binarize(ratings, threshold=2.5):
        """Return a binarized version of the given matrix.

        To binarize a matrix, replace all entries above the threshold with 1.
        and replace all entries at or below the threshold with a -1.

        Entries whose values are 0 represent null values and should remain at 0.

        Note that this method is intentionally made static, as you shouldn't use
        any attributes of Chatbot like self.ratings in this method.

        :param ratings: a (num_movies x num_users) matrix of user ratings, from
         0.5 to 5.0
        :param threshold: Numerical rating above which ratings are considered
        positive

        :returns: a binarized version of the movie-rating matrix
        """
        ########################################################################
        # TODO: Binarize the supplied ratings matrix.                          #
        #                                                                      #
        # WARNING: Do not use self.ratings directly in this function.          #
        ########################################################################

        # The starter code returns a new matrix shaped like ratings but full of
        # zeros.
        binarized_ratings = ratings

        binarized_ratings = np.where(ratings > threshold, 1, binarized_ratings)
        binarized_ratings = np.where((ratings <= threshold) & (ratings != 0), -1, binarized_ratings)

        ########################################################################
        #                        END OF YOUR CODE                              #
        ########################################################################
        return binarized_ratings

    def similarity(self, u, v):
        """Calculate the cosine similarity between two vectors.

        You may assume that the two arguments have the same shape.

        :param u: one vector, as a 1D numpy array
        :param v: another vector, as a 1D numpy array

        :returns: the cosine similarity between the two vectors
        """
        ########################################################################
        # TODO: Compute cosine similarity between the two vectors.             #
        ########################################################################
        similarity = 0
        dotprod = np.dot(u,v)
        lenU = np.linalg.norm(u)
        lenV = np.linalg.norm(v)
        similarity = dotprod/(lenU * lenV)
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return similarity

    def recommend(self, user_ratings, ratings_matrix, k=10, llm_enabled=False):
        """Generate a list of indices of movies to recommend using collaborative
         filtering.

        You should return a collection of `k` indices of movies recommendations.

        As a precondition, user_ratings and ratings_matrix are both binarized.

        Remember to exclude movies the user has already rated!

        Please do not use self.ratings directly in this method.

        :param user_ratings: a binarized 1D numpy array of the user's movie
            ratings
        :param ratings_matrix: a binarized 2D numpy matrix of all ratings, where
          `ratings_matrix[i, j]` is the rating for movie i by user j
        :param k: the number of recommendations to generate
        :param llm_enabled: whether the chatbot is in llm programming mode

        :returns: a list of k movie indices corresponding to movies in
        ratings_matrix, in descending order of recommendation.
        """

        ########################################################################
        # TODO: Implement a recommendation function that takes a vector        #
        # user_ratings and matrix ratings_matrix and outputs a list of movies  #
        # recommended by the chatbot.                                          #
        #                                                                      #
        # WARNING: Do not use the self.ratings matrix directly in this         #
        # function.                                                            #
        #                                                                      #
        # For GUS mode, you should use item-item collaborative filtering with  #
        # cosine similarity, no mean-centering, and no normalization of        #
        # scores.                                                              #
        ########################################################################
        
        recommendation_scores = []
    
        # Find indices of movies the user has not yet rated
        rated_indices = np.where(user_ratings != 0)[0]
        unrated_indices = np.where(user_ratings == 0)[0]
        
        # Iterate over each movie in the dataset
        for j in unrated_indices:
            # Calculate similarity score for movie i
            similarity_score = 0
            for i in rated_indices:
                if user_ratings[i] != 0:
                    similarity_score += self.similarity(ratings_matrix[j, :], ratings_matrix[i, :]) * user_ratings[i]
            
            # Add recommendation score for movie i to the list
            if not np.isnan(similarity_score):
                recommendation_scores.append((j, similarity_score))

        # Sort recommendation scores in descending order
        sorted_recommendation_scores = sorted(recommendation_scores, key=lambda x: x[1], reverse=True)

        # Get top k recommendations
        recommendations = [movie_index for movie_index, _ in sorted_recommendation_scores[:k]]
    
        ########################################################################
        #                        END OF YOUR CODE                              #
        ########################################################################
        return recommendations

    ############################################################################
    # 4. Debug info                                                            #
    ############################################################################

    def debug(self, line):
        """
        Return debug information as a string for the line string from the REPL

        NOTE: Pass the debug information that you may think is important for
        your evaluators.
        """
        debug_info = 'debug info'
        return debug_info

    ############################################################################
    # 5. Write a description for your chatbot here!                            #
    ############################################################################
    def intro(self):
        """Return a string to use as your chatbot's description for the user.

        Consider adding to this description any information about what your
        chatbot can do and how the user can interact with it.

        NOTE: This string will not be shown to the LLM in llm mode, this is just for the user
        """

        return """
        This chatbot is named Kernie. This chatbot uses NLP and LLMs to recommend movies to you!
        """


if __name__ == '__main__':
    print('To run your chatbot in an interactive loop from the command line, '
          'run:')
    print('    python3 repl.py')
