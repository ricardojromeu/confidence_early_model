# confidence_early_model
Here I put the code I wrote in Python in order to fit data of confidence ratings in a probabilistic reasoning task from healthy controls and schizophrenia patients.

These data were originally obtained from *Gerit Pfuhl's Open Science Framework* page found here: https://osf.io/uca5e/
If you're interested in the design of the experiment, or how they analyzed their data, I highly encourage you to check out the preprint they posted to their OSF page, it's a great read. 

To get a quick summary of the models and the results, see the YouTube video I submitted to the Australian Mathematical Psychology Conference for February 2021:
https://www.youtube.com/watch?v=gjFi_4MGGxY&t=8s

# Basics of the task

Participants were told that there were two bags of beads, one with 80 blue and 20 white beads, and one with 80 white beads and 20 blue beads. The participant would be shown 20 beads drawn from one of these two bags, and their task was to give confidence ratings on how likely they believed the experimenter was pulling from the white bag. This is an example of a probabilistic reasoning task, since it involves people having to make decisions based on probabilistic evidence. 



# The Model

The YouTube video explains the model in a bit more depth, but the basic idea is to treat belief updating as if it were a kind of learning from data; the hypothesis is that belief updating is similar to learning your own subjective assessments of the likelihood of each idea under consideration, where these subjective assessments are responsive to sequences of data presented to you. This is captured in the model by applying a sequence of linear operators on vectors of evidence. 

I fit the model by trying to minimize the sum-of-squared error (SSE) between the model's predictions and the actual data. This was an okay-approach as a starting point, but early on it became apparent that this method would run into difficulties. In particular, there is essentially no treatment of measurement error in the model as it stands, and fitting with SSE also does not incorporate measurement error. This means that, if the participant gave wild fluctuations in their confidence ratings, which itself might be due to some kind of response error on the participant's part, fitting the model with SSE would treat these data as without error, and this could sometimes lead either to faulty parameter estimates or faulty predictions. 

Nonetheless, I wanted to see in a simple way whether this modeling approach was viable or not; if it couldn't even have a modicum of success in the early stages, then there would be no point in continuing with more difficult estimation techniques (like hierarchical Bayesian, for instance). 

I not only fit the model to some of the data, but beforehand I had split off a portion of the data to use as test data, to see whether the model could make out-of-sample predictions in accordance with the data. 

Finally, I had to make a separate file to fit the "Bayes-like" version of the model, since I ran into the issue of numbers blowing up too fast. This is because successively applying a linear operator will make the components of the vector grow quite rapidly, and this inevitably led to fitting problems. I figured out a simpler way of implementing the Bayes version of the model that would avoid this blow up, and that method is implemented in the fit_bayes_pfuhl.py file. 

# What I learned

I had some practice in performing a cross validation with the model and these data. I also learned how to implement a minimizing algorithm in Python, which I had to do to minimize the SSE when given the data. There were also some issues with some participants in the dataset, so I had to do some data cleaning beforehand. I also found the original way the data were organized to be cumbersome for the fitting I wanted to do, so I wrote a program to output a pandas dataframe that was more conducive to the analyses I was aiming to do. (This rearranging of data can be found in the file 'Clean_BT_data.py'.) I also learned how to deal with a combination of matrices and vectors in Python using numpy. 

I ran into a problem in the data wrangling process that I'm not so sure I solved in the best way. Each participant had their bead sequences randomized, so for each participant, I needed the bead sequence and sequence of confidence ratings in order to fit the model. I found this difficult to have all in one dataframe unless each datapoint were a 2-tuple holding the bead on that trial and the confidence rating given on that trial. This worked okay until I saved the data to a .txt file. Reloading these data had me run into problems, because pandas would interpret the tuple now as a string. Having to constantly switch between data types was a bit of an annoyance, and some of the code at the top of the fitting files are dedicated in some way of avoiding having to reload the data. Looking back, since the confidence ratings were numbers between 0 and 1, I could have coded the data in a simpler way: let's say the confidence rating on trial 2 was 0.56, and the bead was coded as 1 for "blue bead". Then code the data as 1.56. If instead the bead code was 0 for "white bead," then keep the data coded as 0.56. In other words, I think it would have been simpler to use the one's place of the data to code the bead, instead of saving tuples in a dataframe. 


# Main Files

To see the fitting procedure in action, I suggest paying particular attention to final_fit_pfuhl_data.py or fit_bayes_pfuhl.py. The text files give the estimated parameters (those which end with "\_param") or the data for that group ("\_data").  

And, if you found the data interesting, I highly encourage you to visit Gerit Pfuhl's OSF page :) 
