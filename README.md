ISAGEe-English: Integrated Short Answer Grader for e-learning environment (For English)
                                Version 1.0
=======================

License : http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later

======================= CONTENTS

1. Introduction
2. The Folder Structure
   2.a. The Moodle Plugin ISAGEe-English /V1/
   2.b. External Grader code on Distant server

3. How to deploy ?
4. Feedback & Contact
5. Acknowledgments
=======================

1. Introduction
The grader is integrated into the online quiz system deployed on Moodle. 
The Question Engine, the core module of the quiz system, is extended with our new Question Type ISAGeenglish. 
The grader architecture highlights two modules:
• The ISAGe Question Type Plugin which extends the Question Engine of the quiz system to our new question type. 
  The plugin controls the communication flow with the other modules of the quiz system and inherits the question behavior, question bank, quiz reports, … from the e-learning system. 
• The external grader that is deployed on the Cloud. It implements the trained model to predict grades.

Actually, the Grader is used on our Moodle University Plateform in formative and summative tasks using open short answers.

======================= 
2. Folder Structure (Plugin Moodle + Grader code on distant server)

   2.a.ISAGe Question Type Plugin-English (PHP classes zipped to install on Moodle: ISAGEeenglish.zip)

   2.b.ISAGe External Grader-English (codes and data requirements).  

===== 
2.a. The Moodle Plugin V1 (ISAGEeenglish)

This is our new Moodle Plugin using Moodle question type template. We extend the Moodle question engine to ISAGEenglish to grade English short answers scoring using NLP computation and supervised learning.

=========================

2.b. Grader code on Distant server /

- The grader function is deployed on a distant server (Pythonanywhere).
- The code in this folder is deployed via a web application on Pythonanywhere using the framework Flask.
  A PHP cURL script is used to connect the plugin to the Flask API which deploys the grader on PythonanyWhere.

- We use Python version : Python 3.7 under PyCharm Community Edition 2020.3.2 x64

- The grader can be used and tested in a desktop version (Whithout the plugin as well) :

#############" Instructions to run the code #################

- The folder "Requirements" includes files data(the trained model, semantic space, Word Embeddings, words-dictionary,stop words, Min-Max dictionary(TFlog Wheights))

- Copy the content of the folder "requierement English" in the link bolow in the "requirements" folder of the plugin:
https://drive.google.com/drive/folders/1KPZCvXvQ6VeK54fhkOpNxkUC3PQlPTXp?usp=sharing

- Introduce (question, student answer, reference answer and difficulty) to the getScore function as in the example in the main function.

- The code is written to test several pairs of reference answers and student answers. 
  That's why we used "Arrays" for the questions and answers each time. 
  If the code is to be used to test a single pair then put the question, the reference answer and the student answer each in an "Array". 
  All functions that take the parameters named "AllQuestionCorpus" or "ResponsesCorpus" must be in the folowing syntax: AllQuestionCorpus=[[response1Question1,response2Question1,...],[response1Question2,response2Questio2,...],....] ModelAnswers=[Model1,Model2 , ... ]

======================= 
3. How to deploy the the grader into the e-learning environment?

- The Web gateway to the clients is provided by the API framework which permits hosting the grader on the Cloud.
- We use the Flask Framework to do it.
  The Control and View component is handled by HTTP requests from the client-side (the e-learning system), to the server (Hosting Cloud).
  The Integrated Development Environment (EDI) is used as “PaaS” (Platform as a Service) where the ISAGe runs as “a service” separately. 
  The two modules(code grader(in Python) and the plugin(developped in PHP) communicate through a cURL interface. cURL is a command-line tool for getting or sending data using URL syntax. cURL supports HTTPS to transmit students’ answers and reference answers to the external grader; then it asynchronously waits for grades from the grader and returns them to the LMS quiz system.

- The Plugin must be installed on a Moodle plateform to be used: (Install the ISAGEeenglish.zip) 
 (Dashboard / Site administration / Plugins / Install plugins).

- The external grader code and data requirements must be uploaded on the server (here pythonanywhere)

- A PHP cURL script is used to connect the plugin to the Flask API which deploys the grader on PythonanyWhere(already done in the plugin). 
 (Please see the "question.php" of the plugin in the section commented: // Call the grader ). 
========================
4. Feedback & Contact:
5. Acknowledgments
