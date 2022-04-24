import nltk
import sys
import os
import string
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """

    
    
   
    
    dict_of_file_content = dict()

    # get list of all file names in the directory
    all_files_in_directory = os.listdir(directory)

    #iterate through all the file names
    for fname in all_files_in_directory:
        #only process text files
        txt_extension_location = fname.find(".txt")
        if len(fname) == txt_extension_location + 4: 
            #construct an operating specific path to the file
            full_filename_path = os.path.join(directory, fname)
            #open the file
            try:
                f = open(full_filename_path, mode="rt",encoding='utf-8')

                #read the whole file content into a dictionary and using the fname as the key
                dict_of_file_content[fname] = f.read()
                #close the file and loop back up and process the next file
            finally:
                f.close()
            

    return dict_of_file_content    

#determine if a word is among the group of english stopword
def check_valid_word(word):
    stopwords = nltk.corpus.stopwords.words("english")
    stopwords = set(stopwords)

    for stopword in stopwords:
        if word  in stopwords:
            return False
        else:
            return True

def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    
    #function to remove punctuation from any word
    def punctuation_filter(word_in):
        word = word_in 
        punctuation = string.punctuation
        processed_word = word
        for character in punctuation:
            processed_word = processed_word.replace(character, "")    
        return processed_word
    

    final_processed_document =""
    processed_document = document
    processed_document = processed_document.split(" ")

    
    # remove stopwords         # do not process any English stop words
    for word in processed_document:
        processed_word = punctuation_filter(word)
        if not check_valid_word(processed_word):
            continue

        words = "".join(processed_word)
        final_processed_document += words.lower() + " "

    tokenized_words = nltk.word_tokenize(final_processed_document)
    return tokenized_words

    

def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    
    idfs = dict()
    dictionary_of_documents_containg_set_of_their_words = dict()
    set_of_all_documents_words = set()

    set_of_all_documents_words = set()
    for document_name in documents:
        for word in documents[document_name]:
            # create a set of all unique words in all files
            set_of_all_documents_words.add(word)

            #initialize all possible idfs to '0'
            idfs[word] = 0


    for document_name in documents:       
        # creat dictionary of each document pointing to a set of all it's own words
        dictionary_of_documents_containg_set_of_their_words[document_name]=set(documents[document_name])

    num_of_docs_containing_word = dict()
    for word in set_of_all_documents_words:
        num_of_docs_containing_word[word] = 0
        for document_name in dictionary_of_documents_containg_set_of_their_words:
            if word in dictionary_of_documents_containg_set_of_their_words[document_name]:
                num_of_docs_containing_word[word] += 1

        for word in num_of_docs_containing_word.keys():
            if num_of_docs_containing_word[word] > 0:
                idfs[word] = math.log(len(documents)/num_of_docs_containing_word[word])
    
    return idfs

def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    tfs = dict()
    tfsdict=dict()
    tfidfs  = dict()
    document_query_score = dict()

    ######################## determine the the tf's for each word in each document ###################################
    #get each document in the files Dictionary
    for document_name in files:
        #creat a new set for each document
        specific_words_in_document = set()
        for word in files[document_name]:
            # add every word in the current document to a set
            specific_words_in_document.add(word)

        #find word frequencies for each document
        for word in specific_words_in_document:
            # create an array of all locations in the documment that matches a specific word
            matching_words_indexes=[document_word for document_word in range(len(files[document_name])) if files[document_name][document_word]==word]
            matching_word_count= len(matching_words_indexes)
            tfsdict.update({word:matching_word_count})
            document_plus_word_key = document_name + word  
            tfs[document_plus_word_key] = matching_word_count

          
            

    ######################## Compute the tdifs for all words in all ducuments ###################################
    #now that we found the tf's for each word in each document calculate the TF-IDFs
    for document_name in files:
        specific_words_in_specific_document = set()
        for word in files[document_name]:
            specific_words_in_specific_document.add(word)

        document_query_score[document_name] = 0 # this is being initialize for each documentname and will be used in the next section
        # this is a dictionary to hold all the corpuses file's tdifs
        tfidfs[document_name] = dict() 
        for word in specific_words_in_specific_document:
            document_plus_word_key = document_name + word 
            tfidfs[document_plus_word_key] = (tfs[document_plus_word_key] * idfs[word])



    ######################## now figure out the query word score for each document   ########################
    # variable to add up the total tdidf of each document for each word in the query  
  

    # check each document to see if the each query word is in it. 
    for document_name in files:
        for query_word in query:
            #create a key consiting of wh documentname and the current query word
            document_plus_query_word_key = document_name + query_word
            if document_plus_query_word_key in tfidfs.keys():
                #add up the tdif totals for each query_word for eah document 
                document_query_score[document_name] += tfidfs[document_plus_query_word_key]

    sort_document_query_score = sorted(document_query_score.items(), key=lambda data: data[1], reverse=True)
   
    results = []
    record_counter = -1
    for i in sort_document_query_score:
        record_counter +=1
        if record_counter >= n:
            break
        results.append(i[0])
 
        
    return results

    
top_sentences_results =[]
def rank_sentences_by_query_term_density(sentence_query_term_dict):
    if sentence_query_term_dict =="getSentences":
        return(top_sentences_results)

    if len(sentence_query_term_dict) == 1:
        for final_sentence,notneedvalue in sentence_query_term_dict.items():
            top_sentences_results.append(final_sentence)
    else:    
            # sort the sentences by the highest query term density to the lowest
        sort_sentenced = dict(sorted(sentence_query_term_dict.items(), key=lambda item: item[1], reverse=True)) 
        for final_sentence,notneedvalue in sort_sentenced.items():
            top_sentences_results.append(final_sentence)

    return True


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """

    sentence_candidates=[]
    sentence_score = dict() # NOTE!!!  This is also the "sentence term count"
    sentence_idfs = dict()
    query_term_density=dict()
    

    #Find all sentences that has one more query words in it.
    for key in sentences.keys():
        #initilize dictionary to all zeros. This will be used in a for lok later down.
        sentence_score[key] = 0

        last_key_added = ""
        for query_word in query:
            
            #identify which sentences that contains any querywords and create a list of them  
            if key.lower().find(query_word.lower()) > -1: 
                if last_key_added != key:
                    sentence_candidates.append(key)
                    last_key_added = key


    #count the number of times the keywords show up in each sentence_candidates
    for speific_candidate_sentence in sentence_candidates:
        query_words_in_particular_sentences_group = [] #For a list of tuples of all query-words and their counts in each sentence
        #last_query_word_word_match_count="" #this is to make sure we do not add more that 1 (query_word,word_match_count)to the "query_words_in_particular_sentences_group"
        matched_query_words_list =[]
        for query_word in query:
            query_search_word = query_word.lower()+" "

            word_match_count = speific_candidate_sentence.lower().count(query_search_word.lower())
            if word_match_count > 0:
                # this dictionary is used to track how many times all query words are in the sentence
                sentence_score[speific_candidate_sentence] += speific_candidate_sentence.lower().count(query_word.lower())
                matched_query_words_list.append(query_word)
                query_words_in_particular_sentences_group.append((query_word,word_match_count))





        
        #create the dictionary 'query_term_density[speific_candidate_sentence]' that holds the sentence term density for any sentence whith a matching word(s)
        #this dictionary will be used near the end of this code.
        if sentence_score[speific_candidate_sentence] > 0: 
            query_term_density[speific_candidate_sentence] = (sentence_score[speific_candidate_sentence] / len(speific_candidate_sentence.split(" ")))

        
           
        
        #figure out the idfs for each sentence
        sentence_idfs_sum = 0
        if sentence_score[speific_candidate_sentence] > 0:
            for matched_word in matched_query_words_list:
                if check_valid_word(matched_word.lower()): 
                    try:
                        sentence_idfs_sum += idfs[matched_word]  
                    except:
                        sentence_idfs_sum += idfs[matched_word.capitalize()]
                    sentence_idfs[speific_candidate_sentence] = sentence_idfs_sum
                

    # sort the sentences by the highest idfs to the lowest
    sort_sentence_idfs= dict(sorted(sentence_idfs.items(), key=lambda item: item[1], reverse=True)) 

    # make a ist of sorted hight to low idfs and their sentences
    final_idfs_And_final_sentences_list_of_tuples =[]
    for final_sentences,final_idfs in sort_sentence_idfs.items():
        final_idfs_And_final_sentences_list_of_tuples.append((final_idfs, final_sentences))


    #now iterate the the final_idfs_And_final_sentences_list_of_tuples and determine if there are entries with the same idf values and create a final ranked list of sentences(final_ranked_sentences)
    #if there are entries with the same idf values then determine their final rank by considering which has the highest Query term density
    query_term_density_of_final_ranked_sentences = dict()


    for i in range(len(final_idfs_And_final_sentences_list_of_tuples)):
        #CREATE A NEW DICTIONARY HERE WITH KEY being query-term-density and value the final_ranked_sentences

        #new dictionary with the key being the final ranked sentences and value being query-term-density.  This will be later sorted by query-term-density and used to break the ties of sentence idf ranking
        query_term_density_of_final_ranked_sentences[final_idfs_And_final_sentences_list_of_tuples[i][1]] = query_term_density[final_idfs_And_final_sentences_list_of_tuples[i][1]]
        if i == len(final_idfs_And_final_sentences_list_of_tuples) - 1:
            rank_sentences_by_query_term_density(query_term_density_of_final_ranked_sentences)

        elif final_idfs_And_final_sentences_list_of_tuples[i][0] != final_idfs_And_final_sentences_list_of_tuples[i+1][0]:
            rank_sentences_by_query_term_density(query_term_density_of_final_ranked_sentences)
            query_term_density_of_final_ranked_sentences = dict()

    candidate_sentences = rank_sentences_by_query_term_density("getSentences") # "getSentences" triggers the function "rank_sentences_by_query_term_density" to return the final results
    result_sentences = []
    for i in  range(len(candidate_sentences)):
        if i < n:
            result_sentences.append(candidate_sentences[i])

    return(result_sentences)



if __name__ == "__main__":
    main()
