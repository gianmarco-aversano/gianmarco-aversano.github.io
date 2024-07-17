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

Recently, I got a message from a friend asking me where one could get acquainted with the following concepts:

- Characteristics and sources of Big Data
- Programming languages (R, Python, Matlab, Stata, SAS, SQL, etc.)
- Analysis and design of systems for Data Mining, Business Intelligence, Big Data, Data
  Warehouse, Data Lake e Data Mesh
- Fundamentals of cryptography and blockchain
- Fundamentals of Natural Language Processing: [bag-of-words](https://ataspinar.com/2016/01/21/sentiment-analysis-with-bag-of-words/#:~:text=In%20this%20bag%2Dof%2Dwords,will%20be%20classified%20as%20positive), word embedding
  (Word2vec, GloVe), sentiment analysis
- Text Mining, Large Language Models and Generative Al
- Cloud computing
- GDPR, Differential privacy (I already made a post about it on this blog)
- A.I. ethics: fairness, transparency, security, privacy, safety, explainability, accountability, human oversight, trustworthiness, and long-term impacts

And I realized that (1) I do not have any handy fancy resource to share with them and (2) so I'll just write it myself.

This post's target audience is people that do not work in or study any form of A.I., yet wish to grasp the basic concepts. But I may share links of or mention more detailed stuff.

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

Python

[TO BE CONTINUED]
