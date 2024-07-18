---
title: "A.I... where to start?"
categories:
  - blog
tags:
  - AI
  - ML
  - Data Science
---

<!-- {%- include mathjax.html -%} -->

> This post is work in progress...
>
> After reading this post you will not be an A.I. expert, this post just covers some basic concepts.
>
> Want to contribute to this post? Create a Pull Request [here](https://github.com/gianmarco-aversano/gianmarco-aversano.github.io).

Recently, I got a message from a friend asking me where one could get acquainted with the following concepts:

- Characteristics and sources of Big Data
- Programming languages (R, Python, Matlab, Stata, SAS, SQL, etc.)
- Analysis and design of systems for Data Mining, Business Intelligence, Big Data, Data
  Warehouse, Data Lake and Data Mesh
- Fundamentals of cryptography and blockchain
- Fundamentals of Natural Language Processing: [bag-of-words](https://ataspinar.com/2016/01/21/sentiment-analysis-with-bag-of-words/#:~:text=In%20this%20bag%2Dof%2Dwords,will%20be%20classified%20as%20positive), word embedding
  (Word2vec, GloVe), sentiment analysis
- Text Mining, Large Language Models and Generative Al
- Cloud computing
- GDPR, Differential privacy (I already made a post about it on [this blog](https://gianmarco-aversano.github.io/blog/differential-privacy/))
- A.I. ethics: fairness, transparency, security, privacy, safety, explainability, accountability, human oversight, trustworthiness, and long-term impacts

And I realized that I do not have any handy fancy resource to share with them, so I'll just write it myself.

This post's target audience is people that do not work in or study any form of A.I., yet wish to grasp the basic concepts.

I may share links of or mention more detailed stuff, either to outsource sections of this post or refer the reader to more in-depth documentation.

## Big Data

Big Data refers to datasets that are so large and complex that traditional data-processing software cannot manage them. It encompasses a vast array of data types and sources.

Let's see some examples:

- Platforms like Facebook, Twitter, LinkedIn, and Instagram generate vast amounts of data from user interactions, posts, comments, likes, and shares.
- (Internet of Things, IoT) Devices such as sensors, smart appliances, and wearable technology continuously produce data regarding their operations, usage patterns, and environmental conditions.
- Medical records, clinical trials, wearable health devices, and medical imaging generate extensive data related to patient care, diagnostics, and treatment.
- Digital media, including news articles, videos, audio recordings, and online publications, contribute significantly to Big Data.

Not only Big Data is too, well, big to be hosted on our laptops, it may even be too big to sit in one place, and some company's databases or data sources may be shattered across the globe.

Big Data are also often associated some characteristics, referred to as the 5 V's of Big Data: velocity, volume, value, variety and veracity.

### Velocity

Velocity refers to how quickly data is generated and how fast it moves. This is an important aspect for organizations that need their data to flow quickly, so it's available at the right times.

Velocity applies to the speed at which this information arrives -- for example, how many social media posts per day are ingested -- as well as the speed at which it needs to be digested and analyzed -- often quickly and sometimes in near real time.

In some cases, however, it might be better to have a limited set of collected data than to collect more data than an organization can handle -- because this can lead to slower data velocities.

### Volume

Volume refers to the amount of data. If the volume of data is large enough, it can be considered Big Data. However, what's considered to be Big Data is relative and will change depending on the available computing power that's on the market.

For example, a company that operates hundreds of stores across several states generates millions of transactions per day. This qualifies as Big Data, and the average number of total transactions per day across stores represents its volume.

### Value

Value refers to the benefits that Big Data can provide, and it relates directly to what organizations can do with that collected data. Being able to pull value from Big Data is a requirement.

Organizations can use Big Data tools to gather and analyze the data, but how they derive value from that data should be unique to them. Tools like [Apache Hadoop](https://hadoop.apache.org) can help organizations store, clean and rapidly process this massive amount of data.

### Variety

Variety refers to the diversity of data types.

- Tabular data: tables, where with named column containing data related to one entity, and indexed rows containings one data point for all entities (columns).
- Images: usually objects made of pixels of shape $C \times H \times W$ (number of channels, height and width), where $C$ is the number of channels (usually 3 as most images are RGB images, standing for red-green-blue), $H$ is the height and $W$ is the width (yes, pictures are cubes). Each pixel is just a number between $0$ and $255$. For each point in the figure, we have $3$ values per pixel, corresponding to RGB. The higher $W$ and $H$ are, the higher is the picture's definition. Examples [here](https://pytorch.org/vision/main/datasets.html).
- Textual data: refers to any (natural language or not) text. The content of this post is text data. Some text datasets. For more examples, see [here](https://ics.uci.edu/~smyth/courses/cs175/text_data_sets.html) and [here](https://github.com/pytorch/text).
- Graph data are data where nodes can be connected to other nodes via edges. Examples are molecules, where nodes are atoms and edges are the chemical bonds, or social media data, where a node can be a user, connected via edges to other users or other posts. Nodes can have attributes (e.g. color, age, location, etc.), and so do edges (type of chemical bond, strength, etc.). For examples, see [here](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html).
- Timeseries data are usually tabular data where the ordering of the rows is important. Usually, they have a _time_ column, containing the timestamp at which that particular row was created. Examples of such data are stock prices, weather data, etc. Also see [here](https://hackernoon.com/10-best-datasets-for-time-series-analysis).
- Multi-modal data: an example of this is an image with a description of what is in it, but also two paired images showing the same person from different angles (this is techically multi-view but bear with me), or a graph where nodes are videos, or captioned videos, etc. See [here](https://github.com/xiaobai1217/Awesome-Video-Datasets).

There are more.

Data can be unstructured, semi-structured or structured. Unstructured data is data that's unorganized and comes in different files or formats. Typically, unstructured data isn't a good fit for a mainstream relational database because it doesn't fit into conventional data models. Semi-structured data is data that hasn't been organized into a specialized repository but has associated information, such as metadata. This makes it easier to process than unstructured data. Structured data, meanwhile, is data that has been organized into a formatted repository. This means the data is made more addressable for effective data processing and analysis.

Raw data also qualifies as a data type. While raw data can fall into other categories -- structured, semi-structured or unstructured -- it's considered raw if it has received no processing at all. Most often, raw applies to data imported from other organizations or submitted or entered by users. Social media data often falls into this category.

### Veracity

Veracity refers to the quality, accuracy, integrity and credibility of data. Gathered data could have missing pieces, might be inaccurate or might not be able to provide valuable insight. Veracity, overall, refers to the level of trust there is in the collected data.

Data can sometimes become messy and difficult to use. A large amount of data can cause more confusion than insights if it's incomplete. For example, in the medical field, if data about what drugs a patient is taking is incomplete, the patient's life could be endangered.

Data with compromised veracity could, for instance, be lacking in proper data lineage -- that is, a verifiable trace of its origins and movement.

## Programming languages

Python and SQL. Yes, R is "popular" and I myslef used MATLAB (a lot) during my Ph.D., but I will focus on Python and SQL. Simply because they are even more popular and demanded.

### SQL

SQL (Structured Query Language) is a standardized programming language specifically designed for managing and manipulating relational databases. It is widely used for querying, updating, and managing data in a relational database management system (RDBMS).

SQL is a powerful, versatile, and widely used language for managing relational databases. Its standardized nature ensures consistency and compatibility across different database systems, making it an essential tool for data management, manipulation, and analysis.

SQL is a declarative language, meaning users specify what they want to do with the data rather than how to do it. The underlying database system handles the execution details. SQL _queries_ are usually executed against a database that carries out the calculations.

Example:

```SQL
SELECT * FROM employees WHERE department = 'Sales';
```

This line of code asks the database to select all rows from the `employees` table, where the column `department` is equal to `Sales`.

As you can see, with SQL you can (quickly) get useful information from a database. Of course, more complicated commands also exist. You can create tables, insert/delete rows, etc.

A very important concept here is a database **schema**. A schema defines what columns exist, their names, their data type, any constraints on it (e.g. a column may have numbers only between 0 and 1). SQL also allows you to manage a database's schema.

For example:

```sql
CREATE TABLE employees (
    id INT PRIMARY KEY,
    name VARCHAR(100),
    department VARCHAR(50)
);
```

In the `employees` table, `id`, `name`, and `department` are columns.

For more, you can visit websites such as [this](https://www.w3schools.com/sql/).

### Python

Python is a high-level, interpreted programming language known for its readability, simplicity, and versatility. It was created by Guido van Rossum and first released in 1991. Python's design philosophy emphasizes code readability and ease of use, making it an excellent choice for both beginners and experienced developers. It supports multiple programming paradigms, including procedural, object-oriented, and functional programming.

Python is an interpreted language, meaning code is executed line by line, which facilitates debugging and development.
Variable types are determined at runtime, allowing more flexibility in writing code.

```python
x = 10   # x is an integer
x = 1.0 # x is a float
x = "Hello"  # x is now a string
```

You can also specify type, but just for readabilty:

```python
x: int  = 10   # x is an integer
x: float = 1.0 # x is a float
x: str = "Hello"  # x is now a string
```

Today, Python is the most popular programming language for Data Science, Machine Learning and Deep Learning.
Many Python frameworks exist for almost anything, so of course they also exist for Machine Learning: [Scikit-Learn](https://scikit-learn.org/stable/), [Keras](https://keras.io/), [Lightning](https://lightning.ai/docs/pytorch/stable/) (listed in order of complexity).

Here is a code snippet to train an [Image GPT](https://cdn.openai.com/papers/Generative_Pretraining_from_Pixels_V2.pdf) model on the [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset.

```python
import pytorch_lightning as pl
from pl_bolts.datamodules import FashionMNISTDataModule
from pl_bolts.models.vision import ImageGPT
model = ImageGPT(datamodule=FashionMNISTDataModule(PATH))
pl.Trainer(accelerator="auto").fit(model)
```

To learn more about Python, I recommend you get started with a serious bootcamp or online tutorial. This post cannot cover all the ground for you. But feel free to reach out.

## Analysis and design of systems for Data Mining, Business Intelligence, Big Data, Data Warehouse, Data Lake and Data Mesh

Each of these systems serves a different purpose within the realm of data management and analytics, even if they often intersect and interact.

Data Mining is usually referred to the process of discovering patterns, correlations, and insights from large datasets. Example of data mining applications are: market analysis, [fraud detection](https://www.fraud.com/post/fraud-detection#:~:text=Fraud%20detection%20is%20the%20process%20of%20identifying%20and%20mitigating%20fraudulent,suspicious%20activities%20indicative%20of%20fraud.), [customer segmentation](https://www.forbes.com/advisor/business/customer-segmentation/), predictive maintenance.

So first of all you need a data source, like a database, data warehouses, data lakes, etc. Once you have that, you may want to use Python ([Numpy](https://numpy.org/), [Pandas](https://pandas.pydata.org/)) or SQL (or more complex stuff such [Spark](https://spark.apache.org/)).

Some processes that you'll do with these tools are: data processing, i.e. clean and transform data (handling missing values, normalization), or even train a Machine Learning model (classification, clustering, regression, association rule learning, anomaly detection). This depends on what you want to achieved.

Data mining is a critical component of business intelligence (BI). Once all the valuable information has been extracted from the data, businesses turn it into actionable knowledge – in other words, business intelligence.

This knowledge helps organizations make data-driven decisions not only to improve operations, increase revenue, and drive growth, but also to reduce risks and detect fraud, errors, and inconsistencies that can potentially lead to profit loss and reputation damage. Different industries use data mining in different contexts, but the goal is the same: to better understand customers and the business.

Examples:

- Service providers, such as telecom and utility companies, use data mining to predict ‘churn’, the terms they use for when a customer leaves their company to get their phone/gas/broadband from another provider. They collate billing information, customer services interactions, website visits, and other metrics to give each customer a probability score, then target offers and incentives to customers whom they perceive to be at a higher risk of churning. For example, if a customer has a history of calling customer service with complaints, the service provider can offer them a discount or other incentives to keep them from leaving. This not only helps businesses retain customers but also helps them save on customer acquisition costs.
- Some of the most well-known data mining applications are in e-commerce. E-commerce companies use data mining to analyze customer behavior and create personalized, real-time recommendations. By analyzing customer purchase history, e-commerce companies can recommend products that are most relevant to the customer's interests. One of the most famous of these is, of course, Amazon, which uses sophisticated mining techniques to drive their, "People who viewed that product, also liked this" functionality. This not only helps increase customer satisfaction but also helps businesses increase revenue through cross-selling and upselling.

There are many data mining techniques that businesses use to analyze their data. Some of the common ones are:

- **Classification**: Categorizes data into predefined groups based on specific criteria.
- **Clustering**: Groups similar data points together based on their similarity.
- **Regression**: Predicts the value of one variable based on another variable.
- **Association Rules**: Identifies relationships between different variables in large datasets.
- **Sequence Mining**: Identifies patterns and sequences in data that occur frequently.
- **Text Mining**: Extracts relevant information and patterns from unstructured text data.
- **Anomaly Detection**: Identifies unusual patterns or outliers in data that deviate from expected norms.
- **Dimensionality Reduction**: Reduces the number of variables in a dataset while retaining key information.
- **Feature Selection**: Identifies the most important variables or features in a dataset.
- **Neural Networks**: Models complex relationships between variables using a system inspired by the human brain.

Data mining can be an incredibly powerful tool for businesses, but it's not without its challenges. Here are some common data mining pitfalls that businesses may face:

- Data quality is a key challenge in data mining as incomplete, inconsistent, or erroneous data can lead to incorrect conclusions.
- Data privacy and security is a concern as businesses need to follow best practices and comply with regulations to prevent sensitive data from being leaked or hacked.
- Technical expertise is required for data mining, and finding and hiring skilled data scientists and analysts can be a challenge.
- The volume of data being generated can be overwhelming and lead to longer processing times and higher costs.
- Interpretation of data mining results can be challenging as patterns and relationships may not be immediately clear and require further analysis.
- Overfitting can occur if the model used is too complex and based on too few examples, leading to incorrect conclusions.

> [TO BE CONTINUED]
