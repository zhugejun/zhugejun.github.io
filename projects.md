---
layout: page
title: Projects 
---

Here is a list of projects I have worked on.

## Popular Time of Miami Fitness Center

This project was from client - Employee Benefits & Wellness Department at Miami University. The client wanted to generate the popular time within some time interval for the fitness center after uploading the raw data. 

We built a shiny app which is hosted on shiny server of Statistics Department. It can be viewed in [here](http://dataviz.miamioh.edu/EHWB) or by running `runGitHub("sta660", "zhugejun")` in `R`. The source code can be found [here](https://github.com/zhugejun/sta660/tree/master).

Here is a screen shot of what the shiny looks like:
![EHWB](/images/EWHB_shiny_app.png)



## Median Income vs. Female Percentage

I was interested in finding if there existed any relationship between median income and female percentage in different major categories. I chose the data from Kaggle Datasets - [US Census](https://www.kaggle.com/census/2013-american-community-survey) and used this [script](https://www.kaggle.com/zhugds/d/census/2013-american-community-survey/extract-data) to extract the desired dataset. 

After that, `ggplot2` and `Inkscape` were used to create the [plot](/projects/plot+reorder.pdf). The plot style was inspired by the [post](http://blog.yhat.com/posts/replicating-five-thirty-eight-in-r.html) on [yhat](www.yhat.com). The source code can be found [here](https://gist.github.com/zhugejun/88bb4d354f9b7c7319f3).

![median income vs female percentage](/images/median.png)

As can be seen that different categories had different relationships. But in general, as female percentage increased, median income decreased. However, this didn't imply that there existed any casual relationship between median income and female percentage. 