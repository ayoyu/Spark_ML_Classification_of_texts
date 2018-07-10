from pyspark import SparkContext
from pyspark.sql import Row
from pyspark.sql import SparkSession
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
sc = SparkContext()
spark = SparkSession.builder.getOrCreate()

def load_data(path):
    rdd = sc.textFile(path).map(lambda line:line.split()).map(lambda word:Row(label=word[0],words=word[1:]))
    return spark.createDataFrame(rdd)

train_data = load_data('/home/ayoub/Desktop/Testcode/openDS/20ng-train-all-terms.txt')
test_data = load_data('/home/ayoub/Desktop/Testcode/openDS/20ng-test-all-terms.txt')

vectorizer = CountVectorizer(inputCol = 'words',outputCol='bag_of_words')
vectorizer_transformer = vectorizer.fit(train_data)
train_bag_of_word = vectorizer_transformer.transform(train_data)
test_bag_of_words = vectorizer_transformer.transform(test_data)

label_indexer = StringIndexer(inputCol = 'label', outputCol = 'label_index')
label_transformer = label_indexer.fit(train_bag_of_word)
train_bag_of_word = label_transformer.transform(train_bag_of_word)
test_bag_of_words = label_transformer.transform(test_bag_of_words)

classifier_naive = NaiveBayes(labelCol = 'label_index',featuresCol = 'bag_of_words',predictionCol ='label_pred')
classifier_transf = classifier_naive.fit(train_bag_of_word)
test_pred = classifier_transf.transform(test_bag_of_words)

evaluator = MulticlassClassificationEvaluator(labelCol = 'label_index',predictionCol ='label_pred',metricName = 'accuracy')
accuracy = evaluator.evaluate(test_pred)
print('NaiveBayes model accuracy_score = {:.2f}'.format(accuracy))
