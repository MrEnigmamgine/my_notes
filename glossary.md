[Stats Gist List: An Irreverent Statistician’s Guide to Jargon | by Cassie Kozyrkov | Towards Data Science](https://towardsdatascience.com/stats-gist-list-an-irreverent-statisticians-guide-to-jargon-be8173df090d)

> [**Alternative Hypothesis**](http://bit.ly/quaesita_damnedlies)
> 
> A description of all possible states of the world where you would not want to be taking your [**default action**](http://bit.ly/quaesita_damnedlies). (Example [here](http://bit.ly/quaesita_hypexample).)
> 
> [**Analytics**](http://bit.ly/quaesita_battle)
> 
> Analytics is a subdiscipline of [**data science**](http://bit.ly/quaesita_datasci) that is often confused with **statistics**.
> 
> > Analytics is all about finding good questions, while statistics is all about finding good answers.
> 
> The key difference is that analytics is concerned primarily with what’s _in_ your [**data**](http://bit.ly/quaesita_hist) while **statistics** is concerned with what’s _beyond_ your [**data**](http://bit.ly/quaesita_hist). Learn more [here](http://bit.ly/quaesita_versus).
> 
> [**Artificial Intelligence**](http://bit.ly/quaesita_ai) **(AI)**
> 
> A term that used to mean [something else](http://bit.ly/quaesita_aim), but these days it’s often used as a casual synonym for [machine learning](http://bit.ly/quaesita_simplest) (ML).
> 
> > “If it’s written in Python, it’s probably machine learning. If it’s written in PowerPoint, it’s probably AI.” — Mat Velloso
> 
> For the difference between ML/AI and **statistics**, see the entry on **machine learning.**
> 
> [**Assumption**](http://bit.ly/quaesita_saddest)
> 
> Assumptions are ugly band-aids we put over the parts where information is missing. If we knew [_all_ the facts](http://bit.ly/quaesita_popwrong) (and we _knew_ that our facts were actually true facts), we wouldn’t need assumptions (or statisticians). Unfortunately, when we have partial information — when we want to know about a whole **population** but we only observe data from a **sample** — the only way we can make the leap beyond our paltry information is to make assumptions.
> 
> > STATISTICAL INFERENCE = DATA + ASSUMPTIONS
> 
> Yup, **statistics** isn’t magic, it’s the art of making assumptions veeeeeery carefully (and mathematically) to make conclusions beyond our [**data**](http://bit.ly/quaesita_hist).
> 
> **Bar Chart**
> 
> A way to visualize counts. Like **distributions**, you can think of **bar charts** (used for **categorical data**) and **histograms** (used for **continuous data**) in terms of popularity contests. Or tip jars. That works too.
> 
> ![](https://miro.medium.com/max/700/0*CEvGe0w_mdXIeyfy.png)
> 
> [**Bayesian Statistics**](http://bit.ly/quaesita_freqvsbayes)
> 
> A statistical school of thought deals with mathematical models of belief. You start with a mathematical description of what you believe and then (via [**Bayes’ rule**](http://bit.ly/mfml_part4)) discover what you reasonably ought to believe after adding some [**data**](http://bit.ly/quaesita_hist). The results are highly personal because they’re about _reasonably_ updating subjective models of belief — different starting beliefs (called “**priors**”) should give different results (called “**posteriors**”). In Bayesian **statistics**, [**parameters**](http://bit.ly/quaesita_vocab) have [**probabilities**](http://bit.ly/quaesita_bday) attached to them, which is heinous sacrilege as far as your typical ([**frequentist**](http://bit.ly/quaesita_fvsb)) STAT101 class is concerned. The question of whether or not a **parameter** should have a [**probability distribution**](http://bit.ly/quaesita_distribution) what the [Bayesian vs Frequentist controversy](http://bit.ly/quaesita_fvsb) is all about.
> 
> [**Bayes’ Rule**](http://bit.ly/mfml_part4)
> 
> A formula that helps you to go from **probability** of checking Twitter when your code is compiling to the **probability** that your code is compiling when you are checking Twitter. [Bayes’ Rule](https://youtu.be/hfH8Dh8V1RU) is the mathematical underpinning of [Bayesian statistics](http://bit.ly/quaesita_fvsb).
> 
> [**Bias**](http://bit.ly/quaesita_biasdef)
> 
> Statistical bias occurs when results are consistently off the mark. But that’s not the only definition of bias, so dive deeper here: [**selection bias**](http://bit.ly/quaesita_bias), [algorithmic bias](http://bit.ly/quaesita_aibias), and [other kinds of bias](http://bit.ly/quaesita_biasdefn).
> 
> ![](https://miro.medium.com/max/700/0*FtEnCWbqTfUT_4ZX.png)
> 
> [**Binary Data**](http://bit.ly/quaesita_datatypes)
> 
> **Data** that can take values in two categories, e.g. (Yes, No). When you’re dealing with more than two categories, it’s called [**multiclass data**](http://bit.ly/quaesita_datatypes).
> 
> [**Binomial Distribution**](http://bit.ly/quaesita_distributions)
> 
> The **distribution** that describes the **probability** of a particular number of successes out of a bunch of attempts. Found in the context of [binary data](http://bit.ly/quaesita_datatypes).
> 
> **Captured Data**
> 
> Captured data are intentionally created for a specific analytical purpose, while **exhaust data** are byproducts of digital/online activity.
> 
> [**Causation**](http://bit.ly/quaesita_corr)
> 
> The propensity of one variable to _cause_ another variable to change.
> 
> [**Categorical Data**](http://bit.ly/quaesita_datatypes)
> 
> **Data** that takes category values, e.g. Orange, Apple, Mango. [**Binary data**](http://bit.ly/quaesita_datatypes) is a type of categorical data.
> 
> [**CDF**](http://bit.ly/quaesita_distributions)
> 
> See **cumulative distribution function**.
> 
> [**Central Limit Theorem**](http://bit.ly/quaesita_lemur) **(CLT)**
> 
> The CLT is a handy rule that says that if we compute averages or sums from lots of data, those sums/averages will be **normally distributed**. Learn more [here](http://bit.ly/quaesita_distributions).
> 
> ![](https://miro.medium.com/max/700/1*b68VgQZNoyPeph_Ly_WIMw.png)
> 
> The Central Limit Theorem (CLT) says that if you’re working with lots of data, you can safely assume that the distribution of the sample average is normal (bell-shaped), just like the curve behind me.
> 
> **Chi-squared Test**
> 
> Something you may want to learn more about if you work a lot with [**categorical data**](http://bit.ly/quaesita_datatypes). It’s the classic quick check to see if two categorical **variables** are independent, for example to ask a question like, _“Is the distribution of favorite musical genres the same across all college majors?”_ (Or, more morbidly, many STAT101 professors introduce it as, _“Did surviving the Titanic depend on how fancy your ticket was?”_ Because we statisticians are a grim bunch.)
> 
> **Classical Statistics**
> 
> Synonym for **Frequentist Statistics.**
> 
> **Cluster** [**Sampling**](http://bit.ly/quaesita_bias)
> 
> An approach to collecting **data** that involves deciding on sections/clusters (e.g. schools), randomly selecting several clusters from the collection, then collecting **observations** on all of the units (e.g. students) in those clusters.
> 
> [**Combinatorics**](http://bit.ly/quaesita_bday)
> 
> The branch of mathematics for counting things, like the number of ways you can seat your wedding guests without offending anyone (0). [Here](http://bit.ly/quaesita_bday)’s my primer on the subject, which will help you understand, among other things, why your combination lock is actually a permutation lock.
> 
> **Confidence Interval**
> 
> A concept from **frequentist statistics** with a tricky definition. There’s no way around the fact that it’s tricky, so be careful when you interpret confidence intervals (and don’t confuse them with **credible intervals**). A 95% confidence interval means: _“Were this procedure to be repeated on infinite samples, the calculated confidence interval (which would differ for each sample) would encompass the true population_ **_parameter_** _95% of the time.”_
> 
> [**Continuous Data**](http://bit.ly/quaesita_datatypes)
> 
> **Data** that is obtained by measuring, not counting. Examples: 176.5 cm (my height), 12% (free space on my phone), 3.141592… (pi), -40.00 (where Celsius meets Fahrenheit), etc.
> 
> **Convenience Sample**
> 
> A **sample** that is nonrandom but the **observations** were convenient to make, e.g. when you put a booth in the airport terminal and ask people walking by to take a survey about air travel… what could possibly go wrong?
> 
> [**Correlation**](http://bit.ly/quaesita_correlation)
> 
> The propensity of two **variables** to look like they’re moving together. Learn more [here](http://bit.ly/quaesita_correlation).
> 
> ![](https://miro.medium.com/max/305/0*57CSS6kGyZelqTtb)
> 
> Two variables X and Y are correlated if they seem to be moving together in some way. (Plot by author.)
> 
> ![](https://miro.medium.com/max/700/0*TOmKz0gJAnF3iZH8.png)
> 
> On the left, height and (left-to-right) distance are positively correlated. When one goes up, so does the other. On the right, height and distance are negatively correlated. Image by author.
> 
> [**Count Data**](http://bit.ly/quaesita_datatypes)
> 
> [Data](http://bit.ly/quaesita_hist) that can only take non-negative integer values. Obtained by counting things.
> 
> **Credible Interval**
> 
> The Bayesian cousin of the **confidence interval**. It has the easy interpretation you _wish_ a **confidence interval** had. A 95% credible interval is interpreted as _“I believe that the_ **_parameter_** _lives between here and here with 95%_ **_probability_**_.”_
> 
> [**Cumulative Distribution Function (CDF)**](http://bit.ly/quaesita_distributions)
> 
> A mathematical formula describing the **probability** of observing a particular value of a **random variable**. See [**distribution**](http://bit.ly/quaesita_distributions).
> 
> [**Data**](http://bit.ly/quaesita_hist)
> 
> Stuff someone recorded in electronic form. Or, for the slightly more reverent explanation, read [this](http://bit.ly/quaesita_hist).
> 
> [**Data-Mining**](http://bit.ly/quaesita_datasci)
> 
> A synonym for **analytics**. The act of finding patterns in your [**data**](http://bit.ly/quaesita_hist) in order to form **hypotheses** or generate ideas.
> 
> ![](https://miro.medium.com/max/700/0*qz0LPshc1IM6I1q0.png)
> 
> [**Data Science**](http://bit.ly/quaesita_datascim)
> 
> Data science is the discipline of making **data** useful. Its three subdisciplines are called **statistics**, **machine learning**, and **analytics**. To learn more about the differences between these three areas and how they fit into data science, [read this](http://bit.ly/quaesita_datascim).
> 
> ![](https://miro.medium.com/max/700/0*1UBIDB_uHxZQcCN_.jpeg)
> 
> [**Data Point**](http://bit.ly/quaesita_slkid)
> 
> Synonym for **observation** and instance.
> 
> [**Dataset**](http://bit.ly/quaesita_vocab)
> 
> Synonym for **sample** (collection of **data**).
> 
> [**Data Type**](http://bit.ly/quaesita_datatypes)
> 
> Exactly what it sounds like — convenient descriptions of various kinds of data you’d encounter in the wild. Many STAT101 classes kick off their first lesson with data types, so if you’re keen to recreate that experience, head over to [this article](http://bit.ly/quaesita_datatypes) for a guided tour.
> 
> [**Default Action**](http://bit.ly/quaesita_damnedlies)
> 
> A physical action/decision that you [commit to](http://bit.ly/quaesita_inspired) doing if you don’t gather any (more) [evidence](http://bit.ly/quaesita_hist). This is a frequently-overlooked yet super-important concept; you can’t get started with **classical statistics** without it! (Example [here](http://bit.ly/quaesita_hypexample).)
> 
> **Dependent Variable**
> 
> The **variable** (usually Y in our models) we [want to predict](http://bit.ly/mfml_006) using some other ones (usually Xs in our **models**, which are our **independent variables**).
> 
> [**Discrete Data**](http://bit.ly/quaesita_datatypes)
> 
> **Data** that is obtained by counting, not measuring. Examples: 1 short story, 6 words, 2 baby shoes, 0 times worn, etc.
> 
> [**Distribution**](http://bit.ly/quaesita_distributions)
> 
> Think of this as a “[**histogram**](http://bit.ly/quaesita_hist)” of your [**population**](http://bit.ly/quaesita_popwrong) [**data**](http://bit.ly/quaesita_hist). The concept is abstract, since we usually can’t observe the [**population**](http://bit.ly/quaesita_population).
> 
> ![](https://miro.medium.com/max/680/0*3ev-_KNJfM7Zn-Yj.png)
> 
> A distribution gives you popularity contest results for your whole [population](http://bit.ly/quaesita_popwrong). It’s basically the population histogram. Learn more [here](http://bit.ly/quaesita_hist). (Plot by author.)
> 
> **Dummy Variable**
> 
> Synonym for **indicator variable**.
> 
> **Empirical Rule**
> 
> If your data are [**normally distributed**](http://bit.ly/quaesita_distributions), (68%)-(95%)-(virtually all) will be found within 1–2–3 [**standard deviations** of the **mean**](http://bit.ly/quaesita_lemur).
> 
> **Exhaust Data**
> 
> **Captured data** are intentionally created for a specific analytical purpose, while exhaust data are byproducts of digital/online activity. **Exhaust data** usually come about when websites store activity logs for purposes — such as debugging or data hoarding — other than specific analyses.
> 
> **Error**
> 
> The difference between what we observed in our [**data**](http://bit.ly/quaesita_hist) and what we predicted with our [**model**](http://bit.ly/quaesita_emperor). Another term for error is “residual.” In [**simple linear regression**](http://bit.ly/mfml_006), we assume the errors are [**normally distributed**](http://bit.ly/quaesita_distributions). In a method called GLM, we’re allowed to have more creative assumptions about the errors.
> 
> [**Estimate, Estimator, and Estimand**](http://bit.ly/quaesita_vocab)
> 
> An _estimate_ is just a fancy word for **_best guess_** about the true value of a **parameter** (the _estimand_). It’s the value your guess takes, while an _estimator_ is the formula you use for arriving at that number.
> 
> [**Expected Value E(X)**](http://bit.ly/quaesita_distributions)
> 
> A ([**probability**](http://bit.ly/quaesita_bday)\-weighted) average spoken about in the context of **distributions**. Synonyms include: expectation, expected value, mean.
> 
> [**Experiment**](http://bit.ly/quaesita_experiment)
> 
> A [scientific](http://bit.ly/quaesita_scientists) procedure undertaken to test a [**hypothesis**](http://bit.ly/quaesita_damnedlies) involving **causal** relationships. Core characteristics are randomization into groups and manipulation of those groups by the experimenter (different groups are assigned different “treatments”). Experiments allow you to make causal statements about how two things are related (i.e. in order to be able to say that a medication _causes_ an improvement in disease progression, you need to run a well-designed randomized experiment). Until you have clear, measurable, quantitative null and alternative hypothesis statements, as well as a plan for how you will do different things to different parts of the universe at random, what you are about to do is _not_ an experiment.
> 
> ![](https://miro.medium.com/max/700/1*RKLkRYbXGn-w9xa6cV8DXw.png)
> 
> Image modified from [Wiki Commons](https://upload.wikimedia.org/wikipedia/commons/a/a0/Grumpy_Cat_%2814349262900%29.jpg).
> 
> **Exploratory Data Analysis (EDA)**
> 
> The act of using some [fraction of your **data**](http://bit.ly/quaesita_sydd) for the purpose of generating ideas, forming [**hypotheses**](http://bit.ly/quaesita_damnedlies), and discovering potentially useful inputs for [**machine learning**](http://bit.ly/quaesita_simplestm). All of these inspiring nuggets must be [tested on separate data](http://bit.ly/quaesita_sydd) before they can be taken seriously (otherwise you’re [cheating](http://bit.ly/quaesita_charlatan)).
> 
> **Frequentist Statistics**
> 
> The kind of approach you tend to see in STAT 101, based on the long-run frequencies you’d see if procedures were to be repeated infinitely many times. Unlike [**Bayesian statistics**](http://bit.ly/quaesita_fvsb), you’ll never see the words “belief” or “**prior**” when using these methods. In frequentist statistics, **parameters** never have [**probabilities**](http://bit.ly/quaesita_bday) attached to them — [this video](http://bit.ly/quaesita_ytbayescoin) will help you make sense of this with a coin toss and personality test.
> 
> **Gauss-Markov Assumptions**
> 
> [Technical **assumptions**](https://en.wikipedia.org/wiki/Gauss%E2%80%93Markov_theorem) you must make in order to use standard [**linear regression**](http://bit.ly/mfml_006). They translate roughly as, _“Assume it’s_ **_normal_** _and well-behaved.”_ (Now you get the joke on my shirt below.) See **normal distribution** and **scedasticity** to learn more.
> 
> ![](https://miro.medium.com/max/700/1*2zVQzPkis9YS9hO5AuSDgg.png)
> 
> _“Assume it’s_ **_normal_** _and well-behaved.”_
> 
> **Generalized Linear Model (GLM)**
> 
> Generalized linear models (GLMs) extend [**regression**](http://bit.ly/mfml_006) to situations where the distribution of the [**errors**](http://bit.ly/mfml_006) is not [**normal**](http://bit.ly/quaesita_distributions). You may want to learn more about this if you are trying to predict a **categorical response** (e.g. click/no click).
> 
> [**Histogram**](http://bit.ly/quaesita_hist)
> 
> A plot describing the frequency with which things occur in your [**data**](http://bit.ly/quaesita_hist). The [**categories**](http://bit.ly/quaesita_datatypes) (or intervals of values) are on the horizontal axis and the height of the bars gives the relative number of times a particular category has occurred in your data. See also: **bar chart** and [**distribution**](http://bit.ly/quaesita_distributions).
> 
> ![](https://miro.medium.com/max/471/0*_PjmHdqt709ucrNO.png)
> 
> See the full story of where this little histogram comes from [here](http://bit.ly/quaesita_hist). (Plot by author.)
> 
> [**Hypothesis**](http://bit.ly/quaesita_damnedlies)
> 
> A description of how reality might work. H0 stands for **null hypothesis** (all the worlds in which you’d want to take your [**default action**](http://bit.ly/quaesita_damnedlies)), H1 stands for **alternative hypothesis** (all the worlds in which you wouldn’t). (Example [here](http://bit.ly/quaesita_hypexample).)
> 
> [**Hypothesis Testing**](http://bit.ly/quaesita_fisher)
> 
> The game of trying to see if your **data** convinces you that your [**null hypothesis**](http://bit.ly/quaesita_damnedlies) is ridiculous and thus that you should stop doing your [**default action**](http://bit.ly/quaesita_damnedlies). (Example [here](http://bit.ly/quaesita_fisher).)
> 
> **Independent Variable**
> 
> The **variable** (usually X in our [**models**](http://bit.ly/quaesita_emperor)) we want to [use to predict](http://bit.ly/mfml_006) another one (usually Y in our **models**).
> 
> **Indicator Variable**
> 
> A **variable** that takes the value 1 if a condition is met, 0 otherwise. For example, I might record your pet ownership as Cat=1 if you’re owned by a cat and Cat=0 if no cat has claimed you yet. Devs and **ML** folk call the use of indicator variables **one-hot encoding**_._
> 
> ![](https://miro.medium.com/max/700/1*aaXlgxedeqiR1HkNSPjUvA.png)
> 
> My cat politely informing me that I’m not finished reading that other page yet.
> 
> You’re using **_primary data_** if you (or the team you’re part of) collected [observations](http://bit.ly/quaesita_vocab) directly from the real world. In other words, you had control over how those measurements were recorded and stored.
> 
> [**Inherited data**](http://bit.ly/quaesita_provenance)
> 
> Inherited (secondary) data are those you obtain from someone else. The opposite is **primary data**. Here’s [my guide](http://bit.ly/quaesita_notyours) to working with inherited data.
> 
> ![](https://miro.medium.com/max/700/0*SUzqZZWq8bDE0uIQ.jpeg)
> 
> Image by the author.
> 
> [**Kurtosis**](http://bit.ly/quaesita_lemur)
> 
> Kurtosis is a way to describe the chubbiness of a **distribution’s** [tail](http://bit.ly/quaesita_lemur).
> 
> [**Logistic Regression**](http://bit.ly/mfml_part4)
> 
> Often a good **model** to use when the **response variable** (Y) is [**binary**](http://bit.ly/quaesita_datatypes).
> 
> [**Long-Tailed Distribution**](http://bit.ly/quaesita_lemur)
> 
> An [asymmetric (skewed) **distribution**](http://bit.ly/quaesita_lemur) that features extremely large or extremely small values which are relatively infrequent.
> 
> ![](https://miro.medium.com/max/700/0*v545gUympf6e4xS0.png)
> 
> Image by author.
> 
> [**Loss function**](http://bit.ly/quaesita_lossvsmetric)
> 
> A **loss function** is the formula your **machine learning** algorithm tries to minimize during the optimization / model fitting step.
> 
> [**Machine Learning**](http://bit.ly/quaesita_simplest) **(ML)**
> 
> A discipline that’s related to **statistics**, but has a different focus: automation. **Statistics** cares about rigor, inference, and coming to the right conclusion, whereas **machine learning** cares about performance and turning patterns in **data** into recipes that get the job done. Statistics is extremely important in the testing step of an applied machine learning project, since that’s when it’s time to find out whether the prototype actually works.
> 
> [**Mean**](http://bit.ly/quaesita_lemur)
> 
> An [average](http://bit.ly/quaesita_lemur). Used in the context of talking about [**samples**](http://bit.ly/quaesita_vocab) (“sample mean”) or [**populations**](http://bit.ly/quaesita_vocab) (“population mean”). This is one of the important [**moments**](http://bit.ly/quaesita_lemur) (descriptors of the shape of a [**distribution**](http://bit.ly/quaesita_distributions)), which is why you see it talked about so often.
> 
> [**Mean Squared Error**](http://bit.ly/quaesita_babymse)
> 
> The mean squared error (MSE) is one of _many_ [**metrics**](http://bit.ly/quaesita_opera) you could use to measure your [**model**](http://bit.ly/quaesita_emperorm)**’s** performance. If you take a [**machine learning**](http://bit.ly/quaesita_simplest) class, chances are you’ll come across it very early in the syllabus — it’s usually baby’s first [**loss function**](http://bit.ly/quaesita_emperorm) for [**continuous**](http://bit.ly/quaesita_datatypes) data.
> 
> [**Median**](http://bit.ly/quaesita_lemur)
> 
> The middle thing. Arrange your data from smallest to largest and grab the one in the middle — that’s the median. The median is robust to **outliers** while the [**mean**](http://bit.ly/quaesita_lemur) is not.
> 
> [**Mode**](http://bit.ly/quaesita_lemur)
> 
> Mode is pronounced _“the most common value.”_ The mode corresponds to the spot where a [**distribution/histogram**](http://bit.ly/quaesita_distributions) has its peak. When you hear that a distribution is _multimodal_, it means there’s more than one peak. When a distribution is _symmetric and unimodal_, like the pretty little **bell-shaped curve**, the mode also happens to be the **mean**. If you want to be technically correct, you’d stop saying _“the average Joe”_ when you actually mean _“the modal Joe.”_
> 
> **Model**
> 
> Depending on context, either a fancy word for recipe or a description of how a system might work. For example, here’s a straight line model: Salary _\= intercept + slope\*_ Years\_Of\_Experience _\+ error_
> 
> [**Moment**](http://bit.ly/quaesita_lemur)
> 
> Refers to certain numerical summaries of the shape of a [distribution](http://bit.ly/quaesita_distributions). The [average](http://bit.ly/quaesita_lemur) of your data points is called the first [moment](http://bit.ly/quaesita_lemur), the average of squares of your data points is the second moment and so on.
> 
> [**Multiclass Data**](http://bit.ly/quaesita_datatypes)
> 
> **Categorical data** that can take values in more than two categories, e.g. (Cat, Dog, Parrot, Goldfish, [Anteater](https://www.openculture.com/2015/05/salvador-dali-takes-his-anteater-for-a-stroll-in-paris-1969.html)). When you’re dealing with only two categories, it’s called [**binary data**](http://bit.ly/quaesita_datatypes).
> 
> **Multiple Comparisons**
> 
> The act of testing many **hypotheses.** If you don’t make any special corrections to your methods (**multiple testing correction**) and you claim to be making statistical conclusions, you’re going on a fishing expedition in your data: you will find “something interesting” by random chance even though the results will not be real.
> 
> If you’re not doing **statistics** but instead running an **exploratory data analysis** (EDA), you’re in the clear because you know you’re not supposed to take those findings seriously. But if you’re claiming to draw statistical conclusions from this process, what you’re not doing is not statistics — it’s a pantomime that violently misses the point. Just. Don’t.
> 
> Always remember to [split your **data**](http://bit.ly/quaesita_sydd) so that you’re able to run a lean and controlled test in an unmolested dataset after you’ve fished around in a _different one_ for inspiration.
> 
> **Multiple Testing Correction**
> 
> That said, if you wish to test multiple **hypotheses** the valid statistical way, you can… but there’s a price.\* You must make adjustments (start by reading up on _Bonferroni correction_ (the simplest, strictest, and most data-expensive option) and then progress to its cousins), otherwise the results that you think are “**statistically-significant**” will turn out to be embarrassingly fake.
> 
> \*Oh, and that price is pretty steep — you’ll need much more **data** to get the same quality of result. Don’t test multiple **hypotheses** unless you’ve got a really good reason for it.
> 
> [**Multiple Regression**](http://bit.ly/mfml_007)
> 
> Like **simple linear regression**, except now you’re allowed to use more than one **predictor**, e.g. using both Experience and Education to predict Salary.
> 
> **Multivariate Data**
> 
> When your measurements are too complicated for a single number, e.g. the set of measurements a tailor needs in order to create a good custom suit for you. Data that’s not multivariate is called **univariate** (fits in a single number, e.g. your height).
> 
> **Multivariate Regression**
> 
> Like **multiple regression**, except now you’re predicting a **response** that’s **multivariate** — your Y is a vector now, not a scalar.
> 
> **Nonparametric Statistics**
> 
> Methods which do not require you to make [**assumptions**](http://bit.ly/quaesita_saddest) about which [**distribution**](http://bit.ly/quaesita_distributions) your data come from.
> 
> [**Nonresponse Bias**](http://bit.ly/quaesita_bias)
> 
> Happens when your targeted participants omit their responses. For example, you set up a booth to ask people how their day is going and many of the people who are having a rubbish day scowl at you instead of taking your survey. As a result, the **data** you record are too cheerful thanks to nonresponse **bias**.
> 
> [**Normal Distribution**](http://bit.ly/quaesita_distributions)
> 
> The symmetric, bell-shaped distribution found often in nature and wherever we see sums/averages. See [**Central Limit Theorem**](http://bit.ly/quaesita_lemur).
> 
> ![](https://miro.medium.com/max/700/1*jLeyCIiOTeau0Xt40DWMTg.png)
> 
> Normal distribution, also called bell-shaped or Gaussian distribution. (Plot by author.)
> 
> [**Null Hypothesis**](http://bit.ly/quaesita_damnedlies)
> 
> All possible worlds in which you’re happy to take your [default action](http://bit.ly/quaesita_damnedlies). (Example [here](http://bit.ly/quaesita_hypexample).)
> 
> [**Observation**](http://bit.ly/quaesita_vocab)
> 
> A single item in the [**sample**](http://bit.ly/quaesita_vocab).
> 
> ![](https://miro.medium.com/max/602/0*t-FMQjc1qnL-S7Zv.png)
> 
> The measurement we took from this blue-labeled tree is an observation. Image by author.
> 
> **One-Hot Encoding**
> 
> The use of **indicator variables**.
> 
> [**Outliers**](http://bit.ly/quaesita_ytoutliers)
> 
> Unusual data points or data points which are unlikely to have been generated by the process responsible for the bulk of the data. What should you do with them? It depends…
> 
> [**p-value**](http://bit.ly/quaesita_puppies)
> 
> The probability of obtaining a [**sample**](http://bit.ly/quaesita_vocab) at least as extreme as the one we just observed when assuming the [**null hypothesis**](http://bit.ly/quaesita_damnedlies) is actually true. That’s a mouthful, so I made the video below to help you wrap your head around it.
> 
> To calculate a p-value, you need to know what [the **CDF** looks like under the **null hypothesis**](http://bit.ly/quaesita_p2). The smaller your p-value, the more ridiculous your **null hypothesis** looks.
> 
> [**Parameter**](http://bit.ly/quaesita_vocab)
> 
> A summary measure of a [**population**](http://bit.ly/quaesita_vocab).
> 
> [**PDF**](http://bit.ly/quaesita_distributions)
> 
> See [**probability density function**](http://bit.ly/quaesita_distributions).
> 
> **Poisson Regression**
> 
> Often a good model to use when the **response variable** (Y) can only be a nonnegative integer. See also: **count data, response, GLM**.
> 
> [**Population**](http://bit.ly/quaesita_popwrong)
> 
> The collection of all items we are interested in.
> 
> **Posterior**
> 
> The belief you end up having when you add data to your **prior** (starting belief). If you see this word, you’re in [**Bayesian statistics**](http://bit.ly/quaesita_fvsb) land.
> 
> [**Power**](http://bit.ly/quaesita_statistics)
> 
> Power is the probability of rejecting the [**null hypothesis**](http://bit.ly/quaesita_damnedlies) if it is false (i.e. of changing your mind if that’s the right thing to do). Along with **significance level**, it determines the quality of your **hypothesis test**.
> 
> **Prediction Interval**
> 
> An interval giving a plausible range for the next value we might observe. A common statistics gotcha is that people think this is what the **confidence interval** does — nope, that’s the prediction interval: it’s wider than the **confidence interval**, meaning that you should be a lot less sure about where the next [**observation**](http://bit.ly/quaesita_vocab) will land than where the [**population parameter**](http://bit.ly/quaesita_vocab) will land. Which makes sense intuitively too— I’d be a lot more surprised to discover that the average height of all males in a city is 6'4'' than to discover that the height of some random next dude I see in the grocery store is 6'4''. One of these would make a headline, the other would make a shrug.
> 
> **Predictor**
> 
> Another word for an independent (X) **variable**. Predictors are observed [**data**](http://bit.ly/quaesita_hist).
> 
> **Primary Data**
> 
> You’re using **primary data** if you (or the team you’re part of) collected [**observations**](http://bit.ly/quaesita_vocab) directly from the real world. In other words, you had control over how those measurements were recorded and stored. If you didn’t, we call that secondary **(inherited) data**.
> 
> **Prior**
> 
> A starting belief written as a [**distribution**](http://bit.ly/quaesita_distribution). If you see this word, you’re in [**Bayesian statistics**](http://bit.ly/quaesita_fvsb) land.
> 
> [**Probability**](http://bit.ly/quaesita_bday)
> 
> _P(X=4)_ would be read in English as _“The probability that my die lands with the 4 facing up.”_ If I’ve got a fair six-sided die, _P(X=4)_\=1/6. But… but… but… what is probability and where does that 1/6 come from? Glad you asked! I’ve covered some probability basics for you [here](http://bit.ly/quaesita_bday), with **combinatorics** thrown in as a bonus.
> 
> [**Probability Density Function (PDF)**](http://bit.ly/quaesita_distributions)
> 
> A mathematical formula describing the relative **probability** of observing a particular value of a [**random variable**](http://bit.ly/quaesita_lemur). If the **random variable** is not [**continuous**](http://bit.ly/quaesita_datatypes), technically this should be called a [probability mass function](http://bit.ly/quaesita_lemur) and it should not exist at values the **random variable** can’t take.
> 
> **Q-Q Plot**
> 
> A visual testing tool for checking [**distribution**](http://bit.ly/quaesita_distributions) [**assumptions**](http://bit.ly/quaesita_saddest), which compares the **data** you got with the **data** you’d tend to get from the **distribution** you’re interested in. Also, it often makes you cry, but that’s [not why it’s named that](https://en.wikipedia.org/wiki/Quantile).
> 
> **R-Squared**
> 
> Proportion of the variability in Y explained by X. In **simple linear regression**, this is the square of the [**correlation**](http://bit.ly/quaesita_corr) between X and Y. Here’s a fun trick: chances are you’re pretty bad at intuiting [**correlations**](http://bit.ly/quaesita_corr), but you’re pretty good at intuiting R-squared as a performance grade… so you can use this like a magic trick in front of your friends. Try it out on [guessthecorrelation.com](http://guessthecorrelation.com/) — instead of guessing the **correlation** itself, guess R-squared by assigning a “percentage grade” (like a teacher grading a student) to how well you’d say X “captures” Y, then take the square root. (Don’t forget to insert a minus if the cloud of points is moving down and to the right.)
> 
> [**Random Variable**](http://bit.ly/quaesita_distributions)
> 
> A random variable (R.V.) is a mathematical function that turns reality into numbers. Think of it as a rule to decide what number you should record in your [dataset](http://bit.ly/quaesita_hist) after a real-world event happens.
> 
> [**Random Variate**](http://bit.ly/quaesita_distributions)
> 
> Many students confuse random variables with random variates. If you’re a casual reader, skip this, but enthusiasts take note: random variates are outcome **values** like {1, 2, 3, 4, 5, 6} while random variables are **functions** that map reality onto numbers. Little _x_ versus big _X_ in your textbook’s formulas.
> 
> [**Raw Data**](http://bit.ly/quaesita_provenance)
> 
> A **dataset** that’s exactly in the form it was collected in — no cleaning or **transformation** has been done to it.
> 
> [**Regression Analysis**](http://bit.ly/mfml_006)
> 
> **Statistical** methods involving fitting [linear **models**](http://bit.ly/mfml_part4) to [**data**](http://bit.ly/quaesita_hist). Usually, the goal is [prediction](http://bit.ly/quaesita_parrot) or [**hypothesis testing**](http://bit.ly/quaesita_damnedlies) about [**correlations**](http://bit.ly/quaesita_corr)/relationships. See [**simple linear regression**](http://bit.ly/mfml_006)**.**
> 
> [**Representative Sample**](http://bit.ly/quaesita_bias)
> 
> A [**sample**](http://bit.ly/quaesita_vocab) which accurately reflects the characteristics of the [population](http://bit.ly/quaesita_vocab).
> 
> **Residual**
> 
> Synonym for **error**.
> 
> **Response**
> 
> Another word for the **dependent (Y) variable**.
> 
> [**Response Bias**](http://bit.ly/quaesita_bias)
> 
> Happens when your targeted participants lie in their responses. To enjoy some instant response **bias**, put on an ugly hat and ask your coworkers whether you look good in it.
> 
> [**Sample**](http://bit.ly/quaesita_vocab)
> 
> A subgroup of the [**population** of interest](http://bit.ly/quaesita_popwrong).
> 
> [**Sample**](http://bit.ly/quaesita_vocab) **size**
> 
> The number of **observations** (**data points**) in your **sample**.
> 
> [**Sampling**](http://bit.ly/quaesita_bias)
> 
> The act of drawing [**observations**](http://bit.ly/quaesita_vocab) from a [**population**](http://bit.ly/quaesita_popwrong).
> 
> **Sampling Frame**
> 
> The list of all items from which we can draw our **sample** [**observations**](http://bit.ly/quaesita_vocab).
> 
> **Scedasticity** (sometimes also spelled **skedasticity**)
> 
> The ugliest possible word we could have picked for a concept that asks, “Is the **distribution** of **errors** the same everywhere?”
> 
> ![](https://miro.medium.com/max/700/1*MWaUvYyiEoq1a_5uBJZsBg.png)
> 
> “”Homoscedastic errors are all alike but every heteroscedastic model [is unhappy in its own way.](https://en.wikipedia.org/wiki/Anna_Karenina_principle)” Homoscedasticity refers to equal variances everywhere along the line, while heteroscedasticity refers to unequal variance.
> 
> You’ll see this word in the context of **linear regression** 101, where we’re asking one of the diagnostic questions about whether the **Gauss-Markov assumptions** are satisfied (if they are, we can proceed with **simple linear regression**, and if they aren’t, \*sad trombone\*). If the scatter of **errors** around the line looks like a sausage (same width of scatter everywhere), you can say, “Whew, the errors are **homoscedastic**.” If the scatter looks more a fan or an orchestra of trumpets or a python that has swallowed an antelope, we declare the **errors** to be **heteroscedastic** and turn to **GLMs** instead of **simple linear regression** or find a clever way to **transform** your **data**.
> 
> [**Selection Bias**](http://bit.ly/quaesita_bias)
> 
> Happens when your **sampling** method prefers some participants to others. For other definitions of **bias**, see [this list](http://bit.ly/quaesita_biasdef).
> 
> **Significance Level**
> 
> The largest [**probability**](http://bit.ly/quaesita_bday) of [**Type I error**](http://bit.ly/quaesita_statistics) you’re willing to tolerate. Along with **power** and **sample** **size,** this is a massively important knob you use to control the quality of your test.
> 
> [**Simple Linear Regression**](http://bit.ly/mfml_006)
> 
> **Regression analysis** with just one **response variable** and one **predictor**, meaning that you’re just fitting a straight line through your data.
> 
> [**Simple Random Sampling**](http://bit.ly/quaesita_bias)
> 
> A completely random draw. In this **sampling** scheme, drawing any [permutation](http://bit.ly/quaesita_bday) of items from the **population** is equally likely.
> 
> **Simpson’s Paradox**
> 
> An aggregation paradox where disaggregated **data** looked at separately for each group points towards conclusions diametrically opposed to what the aggregated **data** would show.
> 
> ![](https://miro.medium.com/max/700/1*zG3nmLzW6oO8iYw5hlphjw.gif)
> 
> In this plot, the aggregated data go up and to the right (line with a positive slope) while the disaggregated data each produce negative slopes that go down and to the left.
> 
> [**Skewness**](http://bit.ly/quaesita_lemur)
> 
> Exactly what it sounds like: a measure of the asymmetry of a **distribution**. For a handy mnemonic to help you remember which is which on positive and negative skew, read [this](http://bit.ly/quaesita_lemur).
> 
> [**Standard Deviation**](http://bit.ly/quaesita_lemur)
> 
> A measure of dispersion. Tells you how far your **data** points are from their **mean**. For more info, see [this article](http://bit.ly/quaesita_lemur). This term is used in the context of talking about [**samples**](http://bit.ly/quaesita_vocab) (“**sample** **standard deviation**”) or [**populations**](http://bit.ly/quaesita_vocab) (“**population standard deviation**”). **Standard deviation** is one of the important descriptors of the shape of a [**distribution**](http://bit.ly/quaesita_distributions), which is why you see it talked about so often. I cover it in more detail [here](http://bit.ly/quaesita_lemur).
> 
> ![](https://miro.medium.com/max/700/0*12dsmlpSa_hAqBac.jpg)
> 
> Think of things that call themselves “deviations” as measuring the distance between diners and the central line of a long Hogwarts-style banquet table. Image: [Pixabay](https://pixabay.com/photos/dining-room-banquet-oxford-5114247/).
> 
> [**Statistic**](http://bit.ly/quaesita_vocab)
> 
> A summary measure computed from a **sample**. In other words, any way of mushing up your **data**. The way we use the terms these days, [**analytics**](http://bit.ly/quaesita_analysts) is the discipline that’s about calculating **statistics**, but [**statistics**](http://bit.ly/quaesita_statistics) is all about going _beyond_ those [**data**](http://bit.ly/quaesita_hist) mushups — an [Icarus-like leap](http://bit.ly/quaesita_popwrong) into the unknown (expect a big splat if you’re not careful). Learn more [here](http://bit.ly/quaesita_datasci) about the subdisciplines of [**data science**](http://bit.ly/quaesita_datasci).
> 
> **Statistically Significant**
> 
> Doesn’t mean what just happened is “significant” in the eyes of the universe. This is a technical term that simply means that a [**null hypothesis** was rejected](http://bit.ly/quaesita_fisher).
> 
> [**Statistics**](http://bit.ly/quaesita_statistics)
> 
> The science of changing your mind under uncertainty. For an 8min intro to the discipline, read [this](http://bit.ly/quaesita_statistics).
> 
> **Stratified Sampling**
> 
> Dividing your **population** into categories, e.g. statisticians and non-statisticians, and then taking **simple random sample** of a desired size from each category (e.g. 100 statisticians and 100 non-statisticians at random).
> 
> [**Structured Data**](http://bit.ly/quaesita_provenance)
> 
> Structured data are neatly formatted for analysis. Most of the **datasets** you’d work with in a [**statistics**](http://bit.ly/quaesita_statistics) class are structured, whereas in the wild, there’s plenty of **unstructured data** (**data** that needs \*you\* to put structure on it).
> 
> [**Unstructured Data**](http://bit.ly/quaesita_provenance)
> 
> If we’re pedantic about it, there’s no such thing as unstructured data (since by being stored, they’re necessarily forced to have some kind of structure), but let me be generous. Here’s what the definition intends to convey: **structured data** are neatly formatted for analysis, while unstructured data are not in the format you want and they force you to put your own structure onto them. (Think of images, emails, music, videos, text comments left by [trolls](http://bit.ly/trollololo).)
> 
> > That thing you call unstructured data is just data that needs \*you\* to put structure on it.
> 
> **Systematic Sampling**
> 
> A systematic selection process, e.g. constructing your **sample** out of all of the IDs which are divisible by 99.
> 
> **Time Series**
> 
> A time-indexed **dataset** where sequential order matters. For example, if we’re recording your hours of sleep today and my hours of sleep today, we can write those in any order — it doesn’t matter if we write yours first or mine first. But if we’re recording your sleep over a few days, we’d be losing something if we shuffled those.
> 
> **Transformation**
> 
> Taking a **variable** ([column of your dataset](http://bit.ly/quaesita_slkid)) and applying a function (e.g. the logarithm) to all the values in that column to make a new **variable**.
> 
> [**Type I Error**](http://bit.ly/quaesita_statistics)
> 
> Falsely [rejecting](http://bit.ly/quaesita_fisher) the [**null hypothesis**](http://bit.ly/quaesita_fisher). (Equivalent to “convicting an innocent person.”) Related to the concept of [false positive](https://www.youtube.com/watch?v=nvRSzEqPQNo&list=PLRKtJ4IpxJpDxl0NTvNYQWKCYzHNuy2xG&index=45), but it’s not quite the same thing (in the same way that a **prediction interval** is different from a **confidence interval**).
> 
> [**Type II Error**](http://bit.ly/quaesita_statistics)
> 
> Incorrectly failing to [reject](http://bit.ly/quaesita_fisher) the **null hypothesis**. (Equivalent to “failing to convict a guilty person.”) The probability of this is equal to one minus Power. Related to [false negative](https://www.youtube.com/watch?v=nvRSzEqPQNo&list=PLRKtJ4IpxJpDxl0NTvNYQWKCYzHNuy2xG&index=45), but not quite the same thing (in the same way that a **prediction interval** is different from a **confidence interval**).
> 
> [**Type III Error**](http://bit.ly/quaesita_statistics)
> 
> Correctly rejecting the wrong **null hypothesis**. In other words, using all the right math to solve the _wrong_ problem!
> 
> **Undercoverage Bias**
> 
> Happens when you can’t reach your whole [**population**](http://bit.ly/quaesita_vocab) (e.g. when your survey can only be seen by people who have computers, but you want to make a statement about the **population** of all human adults).
> 
> **Uniform Distribution**
> 
> The **probability distribution** that is shaped like a brick: all outcomes have equal probability associated with them. Someone asked me to include a photo of myself playing this distribution the way I did the **normal distribution** above, and this is the best thing I could come up with:
> 
> ![](https://miro.medium.com/max/700/1*11N6HWWZJ_1YGnXme6mNSQ.png)
> 
> Uniform distribution Cassie says hi.
> 
> **Univariate Data**
> 
> When your measurements fit in a single number, e.g. your height. If it’s not univariate, it’s **multivariate,** e.g. the set of measurements a tailor needs in order to create a good custom suit for you.
> 
> [**Variance Var(X)**](http://bit.ly/quaesita_distributions)
> 
> This is the square of the [**standard deviation**](http://bit.ly/quaesita_distributions). Used in the context of talking about [**samples**](http://bit.ly/quaesita_vocab) (“sample variance”) or [**populations**](http://bit.ly/quaesita_vocab) (“population variance”). This is one of the important [**moments**](http://bit.ly/quaesita_distributions) (descriptors of the shape of a [**distribution**](http://bit.ly/quaesita_distributions)), which is why you see it talked about so often.
> 
> **Variable**
> 
> Casual usage: a column of your **dataset** if your **dataset** is formatted the [polite way](http://bit.ly/quaesita_slkid). Formal usage: see **random variable**.
> 
> **Volunteer Sample**
> 
> A [**sample**](http://bit.ly/quaesita_vocab) where the respondents are different from the population because they opted-in (e.g. if we want to measure willingness to lend a hand and we ask people to help out by taking our survey, then we are collecting a volunteer sample and we’d expect that the respondents are more willing to help than the nonrespondents). A recipe for **nonresponse bias**.
> 
> **Z-Score**
> 
> Allows you to compare quantities measured in different scales, for example
> 
> _‘Among long jumpers, Melanie’s best jump is twice as “exceptional” as Yufeng’s marathon time is among marathoners.’_
> 
> > z-score = (thing - thing’s [mean](http://bit.ly/quaesita_lemur)) / (thing’s [standard deviation](http://bit.ly/quaesita_lemur))
> 
> # Thanks for reading! Liked the author?
> 
> Let’s be friends! You can find me on [Twitter](https://twitter.com/quaesita), [YouTube](https://www.youtube.com/channel/UCbOX--VOebPe-MMRkatFRxw), [Substack](http://decision.substack.com), and [LinkedIn](https://www.linkedin.com/in/kozyrkov/). Interested in having me speak at your event? Use [this form](http://bit.ly/makecassietalk) to get in touch.
> 
> 1.9K