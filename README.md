# FightTheVirus 
-- frontend: HTML, CSS, JavaScript, BootStrap   
-- backend: Flask   
-- machine learning: Python, NumPy, Pandas, Scikit-learn  
-- model: logistic regression  

# Inspiration
Coronavirus, our new enemy, has continued to infect us and threaten our lives. According to Baidu data, 1.06 people in China got infected by the Coronavirus every minute (Feb. 1. 2020), and this number has continued to rise. In Canada, there had been 4 cases confirmed which disturb the lives of many civilians. If there is no action to prevent it, humanity is at stake. Now, it is time for us to stand up against it and use the power of machine learning.

# What it does
FightTheVirus is available on both web and IOS platform. The main mission is to utilize machine learning to predict the probability of a person having Coronavirus based on attributes such as gender, age, fever, cough, headache, breathing, travel history in Wuhan and China, and sputum. We trained the data of 150 confirmed and suspected cases of patients and by the model of Logistic Regression, we are able to compute the probability. Also, on our app, we have tips section that included all the information that people need to know about Coronavirus. Lastly, we integrated machine learning again to display projected data -- the number of confirmed cases, suspected cases, deaths and recoveries to remind everyone of our situation in order to prepare for it.

# How we built it
We built the app both on the website and mobile. For the web app, we implemented the Python machine learning library scikit-learn for the training of the dataset. The model is Logistic Regression which uses the logistic function to model a binary dependent variable and output the probability from 0 to 1 in the form of a sigmoid/softmax function. Numpy and Pandas were also used in preprocessing data. To integrate the machine learning backend with our web front end, we used flask to connect everything together. The front end website was designed with HTML, CSS, Bootstrap and JavaScript which blended well with flask and Python. For the mobile app, we made an IOS prototype that has all of the functions on the website.

# Challenges we ran into
Since our team is only made of two members and we are both relatively new to machine learning, it was somewhat difficult for us to start. Without any backend experience, we learned flask in a short period amount of time and successfully implemented it in the fusion of frontend and backend. The machine learning part was also tricky to start but with tons of failures, we still manage to finish our model in processing patient attributes and outputting probabilities.

# What's next for FightTheVirus
We were somehow pitied by the limited amount of Coronavirus data online in order for a better machine learning training result. If we have more time, we would request data from official databases to further optimize the prediction. Our app was also a prototype and we can develop the fully operative IOS + Android App in the future.

# Acknowledgement
The basic model of FightTheVirus project is based on "Flask and Data Science Workshop" by Nordstrom.
