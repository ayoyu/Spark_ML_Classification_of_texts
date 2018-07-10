from pyspark import SparkContext
from pyspark.sql import Row
from pyspark.sql import SparkSession
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
sc = SparkContext()
spark = SparkSession.builder.getOrCreate()

def load_data(path):
    rdd = sc.textFile(path).map(lambda line:line.split()).map(lambda word:Row(label=word[0],words=word[1:]))
    return spark.createDataFrame(rdd)

train_data = load_data('/home/ayoub/Desktop/Testcode/openDS/20ng-train-all-terms.txt')
test_data = load_data('/home/ayoub/Desktop/Testcode/openDS/20ng-test-all-terms.txt')

vectorizer = CountVectorizer(inputCol = 'words',outputCol='bag_of_words')
label_indexer = StringIndexer(inputCol = 'label', outputCol = 'label_index')
classifier_naive = NaiveBayes(labelCol = 'label_index', featuresCol = 'bag_of_words', predictionCol ='label_pred')

pipeline = Pipeline(stages = [vectorizer, label_indexer, classifier_naive])
pipeline_model = pipeline.fit(train_data)

test_pred = pipeline_model.transform(test_data)
evaluator = MulticlassClassificationEvaluator(labelCol = 'label_index', predictionCol = 'label_pred', metricName = 'accuracy')
accuracy = evaluator.evaluate(test_pred)
print('NaiveBayes model accuracy_score = {:.2f}'.format(accuracy))
