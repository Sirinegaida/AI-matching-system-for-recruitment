import builtins
import pandas as pd
import os

min = builtins.min


# Import Spark components (with fallback)
try:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import *
    from pyspark.sql.types import *
    import pyspark.sql.functions as F
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False

class SparkAnalysis:
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.spark = None
        self.spark_enabled = SPARK_AVAILABLE   
        optimal_cores = 4  # Use max 4 cores       
        if self.spark_enabled:
            try:
                #creéation de dossier où Spark stockera ses logs d'événements
                event_log_dir = "./spark-events" 
                os.makedirs(event_log_dir, exist_ok=True)
                # Constructeur pour créer une session Spark
                # Exécute Spark en mode local, utilisant tous les cœurs disponibles (*)
                #Taille max des résultats retournés au driver (128 MB)
                #spark.ui.enabled : Active l'interface web Spark
                #spark.ui.port : Port de l'interface web (4041)
                #spark.ui.killEnabled : Permet de tuer les jobs depuis l'interface
                #spark.ui.retainedJobs : Nombre de jobs conservés dans l'historique (500)
                #spark.ui.retainedStages : Nombre de stages conservés dans l'historique (500)
                #spark.driver.host : Adresse hostname du driver
                #spark.driver.bindAddress : Adresse IP sur laquelle le driver écoute (0.0.0.0 = toutes les interfaces)
                #spark.driver.port : Port du driver (0 = port automatique)
                #spark.eventLog.enabled : Active la journalisation des événements Spark
                #spark.serializer : Utilise KryoSerializer (plus rapide que Java par défaut)
                #spark.kryo.registrationRequired : N'exige pas l'enregistrement des classes avec Kryo
                #Parralélisme: 2x cœurs (default), 4x cœurs (shuffle)
                #PerformanceKryo serializer, SQL adaptatif, Arrow désactivé
                self.spark = SparkSession.builder \
                    .appName("CVMatchingSystem") \
                    .master("local[*]") \
                    .config("spark.driver.memory", "512m") \
                    .config("spark.executor.memory", "512m") \
                    .config("spark.driver.maxResultSize", "128m") \
                    .config("spark.driver.host",  "localhost") \
                    .config("spark.driver.bindAddress", "0.0.0.0") \
                    .config("spark.driver.port", "0") \
                    .config("spark.ui.enabled", "true") \
                    .config("spark.ui.port", "4041") \
                    .config("spark.ui.killEnabled", "true") \
                    .config("spark.ui.retainedJobs", "500") \
                    .config("spark.ui.retainedStages", "500") \
                    .config("spark.eventLog.enabled", "true") \
                    .config("spark.eventLog.dir", event_log_dir) \
                    .config("spark.eventLog.compress", "true") \
                    .config("spark.sql.adaptive.enabled", "true") \
                    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                    .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "64MB") \
                    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
                    .config("spark.kryo.registrationRequired", "false") \
                    .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
                    .config("spark.sql.execution.arrow.maxRecordsPerBatch", "10000") \
                    .config("spark.default.parallelism", str(optimal_cores * 2)) \
                    .config("spark.sql.shuffle.partitions", str(optimal_cores * 4)) \
                    .config("spark.sql.files.maxPartitionBytes", "128MB") \
                    .config("spark.network.timeout", "300s") \
                    .config("spark.executor.heartbeatInterval", "20s") \
                    .getOrCreate()
                self.spark.sparkContext.setLogLevel("INFO")
                print(f"Spark UI URL: {self.spark.sparkContext.uiWebUrl}")
                
                #self.spark.sparkContext.setLogLevel("WARN")
                print("🚀 Spark Analytics Engine initialized")
                #prendre 10 
                df = self.spark.range(100)
                print("✅ Spark is working!")
                print(f"Count: {df.count()}")
                
                self.spark.stop()
                            
            except Exception as e:
                print(f"⚠️ Spark initialization failed: {e}, using pandas fallback")
                self.spark_enabled = False
    def get_global_statistics(self):
        """
        Fast global statistics using distributed processing
        """
        if self.spark_enabled and self.spark:
            return self._get_spark_statistics()
        else:
            return self._get_pandas_statistics()
    
    def _get_spark_statistics(self):
        """Spark-accelerated statistics calculation"""
        try:
            # Load all tables into Spark DataFrames
            candidates_df = self._load_to_spark("candidates")
            job_offers_df = self._load_to_spark("job_offers") 
            job_matches_df = self._load_to_spark("job_matches")
            clients_df = self._load_to_spark("clients")
            
            # Cache for multiple operations
            candidates_df.cache()
            job_offers_df.cache()
            job_matches_df.cache()
            clients_df.cache()
            
            # Calculate all statistics in parallel
            stats = {
                'nb_candidates': candidates_df.count(),
                'nb_offers': job_offers_df.count(),
                'nb_matches': job_matches_df.count(),
                'nb_clients': clients_df.count()
            }
            
            # Advanced statistics from matches
            if job_matches_df.count() > 0:
                match_stats = job_matches_df.agg(
                    avg("similarity_score").alias("avg_match_score"),
                    count(when(col("similarity_score") > 0.8, 1)).alias("excellent_matches")
                ).collect()[0]
                
                stats.update({
                    'avg_match_score': float(match_stats['avg_match_score']) if match_stats['avg_match_score'] else 0,
                    'excellent_matches': int(match_stats['excellent_matches'])
                })
            else:
                stats.update({'avg_match_score': 0, 'excellent_matches': 0})
            
            # Cleanup
            candidates_df.unpersist()
            job_offers_df.unpersist()
            job_matches_df.unpersist()
            clients_df.unpersist()
            
            return stats
            
        except Exception as e:
            print(f"Spark statistics failed: {e}, falling back to pandas")
            return self._get_pandas_statistics()
    
    def _load_to_spark(self, table_name):
        """Load PostgreSQL table to Spark DataFrame"""
        try:
            # Try direct JDBC first
            jdbc_url = f"jdbc:postgresql://{self.db_manager.db_config.config['host']}:{self.db_manager.db_config.config['port']}/{self.db_manager.db_config.config['dbname']}"
            
            return self.spark.read \
                .format("jdbc") \
                .option("url", jdbc_url) \
                .option("dbtable", table_name) \
                .option("user", self.db_manager.db_config.config['user']) \
                .option("password", self.db_manager.db_config.config['password']) \
                .option("driver", "org.postgresql.Driver") \
                .load()
                
        except Exception as e:
            # Fallback to pandas bridge
            pd_df = pd.DataFrame(self.db_manager.execute_query(f"SELECT * FROM {table_name}"))
            return self.spark.createDataFrame(pd_df)
    
    