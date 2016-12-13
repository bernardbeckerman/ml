ml - predict new venue popularity from data available at opening using a machine-learning model trained on Yelp data

    • Parsed restaurant attributes [ujson] 20MB

    • Trained models on estimator-and-transformer pipelines [sklearn]

        ◦ Features include city, location (latlong) [kneighbors],
  	  category, attributes (e.g., ambience, parking, etc.)
  	  [DictVectorizer, Tfidf, lassoCV, randomforest], and a
  	  stacked model containing all of the above

    • Deployed pickled model [dill]
