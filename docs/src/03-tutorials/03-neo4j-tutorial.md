# Neo4j Integration Tutorial

This tutorial demonstrates how to integrate GraphFrames with Neo4j, a popular graph database. We'll use the same Stack Exchange data from the [Network Motif Tutorial](/03-tutorials/02-motif-tutorial.md), but this time we'll:

1. Load the data into Neo4j as separate node and relationship types
2. Ingest the data from Neo4j into PySpark
3. Create a GraphFrame
4. Calculate PageRank
5. Store the PageRank results back into Neo4j

This workflow shows how to combine the power of Neo4j for graph storage and querying with GraphFrames' distributed graph algorithms.

# Why Integrate GraphFrames with Neo4j?

Neo4j and GraphFrames complement each other well:

- **Neo4j** excels at:
  - Interactive graph queries with Cypher
  - ACID transactions
  - Real-time graph traversals
  - Multiple node and relationship types
  - Rich property graph modeling

- **GraphFrames** excels at:
  - Distributed graph algorithms at scale (PageRank, Connected Components, etc.)
  - Integration with Spark's DataFrame API
  - Machine learning pipelines with MLlib
  - Processing graphs too large to fit in a single machine

By integrating the two, you can store and query your graph in Neo4j while leveraging Spark's computational power for complex analytics.

# Prerequisites

## Neo4j Installation

You'll need a running Neo4j instance. You can:

1. **Use Neo4j Desktop** (recommended for learning): Download from [neo4j.com/download](https://neo4j.com/download/)
2. **Use Neo4j Docker**: `docker run --publish=7474:7474 --publish=7687:7687 neo4j`
3. **Use Neo4j AuraDB**: Neo4j's cloud offering with a free tier

## Stack Exchange Data

First, download the Stack Exchange data using the GraphFrames CLI as described in the [Network Motif Tutorial](/03-tutorials/02-motif-tutorial.md#download-the-stack-exchange-dump-for-statsmeta):

```bash
graphframes stackexchange stats.meta
```

Then build the graph from the XML files:

```bash
spark-submit --packages com.databricks:spark-xml_2.12:0.18.0 --driver-memory 4g --executor-memory 4g python/graphframes/tutorials/stackexchange.py
```

This creates `Nodes.parquet` and `Edges.parquet` files that we'll use in this tutorial.

# Understanding Schema Mapping

GraphFrames currently supports only a single node type and single edge type, while Neo4j supports multiple node labels and relationship types. We need to map between these two models:

## GraphFrames Schema

In GraphFrames, all nodes are in a single DataFrame with a `Type` column:

```python
nodes_df.select("id", "Type").show()
```

```
+--------------------+---------+
|                  id|     Type|
+--------------------+---------+
|01a2b3c4-1234-...|  Question|
|02b3c4d5-2345-...|   Answer |
|03c4d5e6-3456-...|     User |
+--------------------+---------+
```

All edges are in a single DataFrame with a `relationship` column:

```python
edges_df.select("src", "dst", "relationship").show()
```

```
+--------------------+--------------------+------------+
|                 src|                 dst|relationship|
+--------------------+--------------------+------------+
|01a2b3c4-1234-...|02b3c4d5-2345-...|     Answers|
|03c4d5e6-3456-...|01a2b3c4-1234-...|        Asks|
+--------------------+--------------------+------------+
```

## Neo4j Schema

In Neo4j, nodes have specific labels and relationships have specific types:

```cypher
(:Question {id: "...", Title: "...", Score: 10})
(:Answer {id: "...", Body: "...", Score: 5})
(:User {id: "...", DisplayName: "Alice", Reputation: 100})

(:User)-[:ASKS]->(:Question)
(:Answer)-[:ANSWERS]->(:Question)
```

## The Schema Mapper

The @:srcLink(python/graphframes/tutorials/neo4j_integration.py) script includes a `Neo4jSchemaMapper` utility class that handles the conversion between these two schemas:

```python
class Neo4jSchemaMapper:
    """Utility class to map between GraphFrames unified schema and Neo4j node/relationship types."""
    
    # Node type mappings: GraphFrames Type -> Neo4j label
    NODE_TYPE_LABELS = {
        "Question": "Question",
        "Answer": "Answer",
        "User": "User",
        "Vote": "Vote",
        "Badge": "Badge",
        "Tag": "Tag",
        "PostLinks": "PostLink",
    }
    
    # Relationship type mappings
    RELATIONSHIP_TYPES = {
        "Answers": "ANSWERS",
        "Asks": "ASKS",
        "Posts": "POSTS",
        "CastFor": "CAST_FOR",
        "Earns": "EARNS",
        "Tags": "TAGGED_WITH",
        "Links": "LINKS",
        "Duplicates": "DUPLICATES",
    }
```

This class provides methods to:
- Split the unified nodes DataFrame by type and select only relevant properties
- Combine typed Neo4j nodes back into a unified DataFrame
- Handle schema differences automatically

# Running the Tutorial

## Configure Neo4j Connection

Edit the connection settings in @:srcLink(python/graphframes/tutorials/neo4j_integration.py):

```python
# Neo4j connection configuration - customize these for your Neo4j instance
NEO4J_URL = "neo4j://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"  # Change this to your Neo4j password
```

## Install Required Packages

The tutorial requires the Neo4j Spark Connector. Install it with:

```bash
pip install neo4j-spark-connector
```

## Run the Complete Tutorial

Execute the tutorial script with all required packages:

```bash
spark-submit \
  --packages graphframes:graphframes:0.8.4-spark3.5-s_2.12,org.neo4j:neo4j-connector-apache-spark_2.12:5.3.0_for_spark_3 \
  python/graphframes/tutorials/neo4j_integration.py
```

The script will:
1. Load Stack Exchange data from parquet files
2. Write nodes and relationships to Neo4j with proper types
3. Read the data back from Neo4j
4. Create a GraphFrame
5. Calculate PageRank
6. Write PageRank scores back to Neo4j as node properties

# Step-by-Step Walkthrough

Let's walk through each step of the tutorial in detail.

## Step 1: Load Stack Exchange Data

First, we load the pre-processed Stack Exchange data:

```python
import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession
from graphframes import GraphFrame

# Initialize Spark Session
spark: SparkSession = (
    SparkSession.builder
    .appName("GraphFrames Neo4j Integration")
    .config("spark.sql.caseSensitive", True)
    .getOrCreate()
)
spark.sparkContext.setCheckpointDir("/tmp/graphframes-checkpoints/neo4j")

# Load nodes and edges
STACKEXCHANGE_SITE = "stats.meta.stackexchange.com"
BASE_PATH = f"python/graphframes/tutorials/data/{STACKEXCHANGE_SITE}"

nodes_df: DataFrame = spark.read.parquet(f"{BASE_PATH}/Nodes.parquet")
edges_df: DataFrame = spark.read.parquet(f"{BASE_PATH}/Edges.parquet")
```

## Step 2: Write Data to Neo4j

We write nodes to Neo4j, separating them by type:

```python
def write_nodes_to_neo4j(spark: SparkSession, nodes_df: DataFrame) -> None:
    """Write nodes from GraphFrames to Neo4j, separating by node type."""
    
    for node_type in Neo4jSchemaMapper.NODE_TYPE_LABELS.keys():
        neo4j_label = Neo4jSchemaMapper.NODE_TYPE_LABELS[node_type]
        
        # Convert to Neo4j format (filter by type, select relevant properties)
        type_df = Neo4jSchemaMapper.nodes_to_neo4j_format(nodes_df, node_type)
        
        # Write to Neo4j
        (
            type_df.write
            .format("org.neo4j.spark.DataSource")
            .mode("Overwrite")
            .option("url", NEO4J_URL)
            .option("authentication.basic.username", NEO4J_USER)
            .option("authentication.basic.password", NEO4J_PASSWORD)
            .option("labels", f":{neo4j_label}")
            .option("node.keys", "id")
            .save()
        )
```

The `Neo4jSchemaMapper.nodes_to_neo4j_format()` method:
- Filters nodes to a specific type (e.g., only Questions)
- Selects only properties relevant to that type
- Returns a clean DataFrame ready for Neo4j

Similarly, we write relationships:

```python
def write_edges_to_neo4j(edges_df: DataFrame) -> None:
    """Write edges from GraphFrames to Neo4j as relationships."""
    
    for rel_type in Neo4jSchemaMapper.RELATIONSHIP_TYPES.keys():
        neo4j_rel = Neo4jSchemaMapper.RELATIONSHIP_TYPES[rel_type]
        
        # Filter to specific relationship type
        type_df = edges_df.filter(F.col("relationship") == rel_type)
        
        # Write to Neo4j
        (
            type_df.write
            .format("org.neo4j.spark.DataSource")
            .mode("Overwrite")
            .option("url", NEO4J_URL)
            .option("authentication.basic.username", NEO4J_USER)
            .option("authentication.basic.password", NEO4J_PASSWORD)
            .option("relationship", neo4j_rel)
            .option("relationship.save.strategy", "keys")
            .option("relationship.source.labels", ":Node")
            .option("relationship.source.save.mode", "Match")
            .option("relationship.source.node.keys", "src:id")
            .option("relationship.target.labels", ":Node")
            .option("relationship.target.save.mode", "Match")
            .option("relationship.target.node.keys", "dst:id")
            .save()
        )
```

After this step, you can query Neo4j directly! Open Neo4j Browser and try:

```cypher
// Count nodes by type
MATCH (n) RETURN labels(n)[0] as Type, count(*) as Count ORDER BY Count DESC

// View a sample question
MATCH (q:Question) RETURN q LIMIT 1

// Find users who asked questions
MATCH (u:User)-[:ASKS]->(q:Question) 
RETURN u.DisplayName, q.Title 
LIMIT 10
```

## Step 3: Read Data from Neo4j

Now we read the data back from Neo4j into Spark:

```python
def read_nodes_from_neo4j(spark: SparkSession) -> DataFrame:
    """Read nodes from Neo4j and combine into GraphFrames format."""
    
    node_dfs = {}
    for node_type, neo4j_label in Neo4jSchemaMapper.NODE_TYPE_LABELS.items():
        df = (
            spark.read
            .format("org.neo4j.spark.DataSource")
            .option("url", NEO4J_URL)
            .option("authentication.basic.username", NEO4J_USER)
            .option("authentication.basic.password", NEO4J_PASSWORD)
            .option("labels", neo4j_label)
            .load()
        )
        node_dfs[node_type] = df
    
    # Combine into unified GraphFrames format
    return Neo4jSchemaMapper.neo4j_to_graphframes_nodes(spark, node_dfs)
```

The `Neo4jSchemaMapper.neo4j_to_graphframes_nodes()` method:
- Adds a `Type` column to each node DataFrame
- Ensures all DataFrames have the same columns (adding nulls where needed)
- Unions all node types into a single DataFrame

Similarly for relationships:

```python
def read_edges_from_neo4j(spark: SparkSession) -> DataFrame:
    """Read edges from Neo4j and combine into GraphFrames format."""
    
    edge_dfs = []
    for rel_type, neo4j_rel in Neo4jSchemaMapper.RELATIONSHIP_TYPES.items():
        df = (
            spark.read
            .format("org.neo4j.spark.DataSource")
            .option("url", NEO4J_URL)
            .option("authentication.basic.username", NEO4J_USER)
            .option("authentication.basic.password", NEO4J_PASSWORD)
            .option("relationship", neo4j_rel)
            .option("relationship.source.labels", ":Node")
            .option("relationship.target.labels", ":Node")
            .load()
        )
        
        # Add relationship type column and rename for GraphFrames
        df = (
            df.withColumn("relationship", F.lit(rel_type))
            .withColumnRenamed("source.id", "src")
            .withColumnRenamed("target.id", "dst")
            .select("src", "dst", "relationship")
        )
        edge_dfs.append(df)
    
    # Union all edge types
    result_df = edge_dfs[0]
    for df in edge_dfs[1:]:
        result_df = result_df.union(df)
    
    return result_df
```

## Step 4: Create GraphFrame

With our unified DataFrames, we can create a GraphFrame:

```python
# Read from Neo4j
neo4j_nodes_df = read_nodes_from_neo4j(spark)
neo4j_edges_df = read_edges_from_neo4j(spark)

# Optimize for graph algorithms
neo4j_nodes_df = neo4j_nodes_df.repartition(50).checkpoint().cache()
neo4j_edges_df = neo4j_edges_df.repartition(50).checkpoint().cache()

# Create GraphFrame
g = GraphFrame(neo4j_nodes_df, neo4j_edges_df)

print(f"GraphFrame created with {g.vertices.count():,} vertices and {g.edges.count():,} edges")
```

## Step 5: Calculate PageRank

Now we can use GraphFrames' distributed PageRank algorithm:

```python
# Run PageRank
pagerank_result = g.pageRank(resetProbability=0.15, maxIter=10)
nodes_with_pagerank = pagerank_result.vertices

# Show top nodes by PageRank
nodes_with_pagerank.select("id", "Type", "pagerank").orderBy(
    F.col("pagerank").desc()
).show(10)
```

Example output:

```
+--------------------+--------+------------------+
|                  id|    Type|          pagerank|
+--------------------+--------+------------------+
|01a2b3c4-1234-...|Question| 2.567891234567891|
|02b3c4d5-2345-...|    User| 1.987654321098765|
|03c4d5e6-3456-...|Question| 1.765432109876543|
+--------------------+--------+------------------+
```

Let's look at top Questions by PageRank:

```python
nodes_with_pagerank.filter(F.col("Type") == "Question").select(
    "Id", "Title", "pagerank"
).orderBy(F.col("pagerank").desc()).show(5, truncate=50)
```

```
+----+--------------------------------------------------+------------------+
|  Id|                                             Title|          pagerank|
+----+--------------------------------------------------+------------------+
|  42|What is the best way to visualize hierarchical ...|2.5678901234567891|
| 123|How do I interpret p-values in multiple compari...|1.9876543210987654|
| 456|When should I use fixed effects vs random effec...|1.7654321098765432|
+----+--------------------------------------------------+------------------+
```

And top Users:

```python
nodes_with_pagerank.filter(F.col("Type") == "User").select(
    "Id", "DisplayName", "pagerank"
).orderBy(F.col("pagerank").desc()).show(5)
```

```
+----+-------------+------------------+
|  Id|  DisplayName|          pagerank|
+----+-------------+------------------+
| 789|        Alice|1.8765432109876543|
| 234|          Bob|1.6543210987654321|
| 567|      Charlie|1.4321098765432109|
+----+-------------+------------------+
```

## Step 6: Write Results Back to Neo4j

Finally, we write the PageRank scores back to Neo4j as node properties:

```python
def write_pagerank_to_neo4j(nodes_with_pagerank: DataFrame) -> None:
    """Write PageRank results back to Neo4j, updating node properties."""
    
    pagerank_df = nodes_with_pagerank.select("id", "Type", "pagerank")
    
    for node_type in Neo4jSchemaMapper.NODE_TYPE_LABELS.keys():
        neo4j_label = Neo4jSchemaMapper.NODE_TYPE_LABELS[node_type]
        type_df = pagerank_df.filter(F.col("Type") == node_type)
        
        # Write pagerank back to Neo4j
        (
            type_df.write
            .format("org.neo4j.spark.DataSource")
            .mode("Append")
            .option("url", NEO4J_URL)
            .option("authentication.basic.username", NEO4J_USER)
            .option("authentication.basic.password", NEO4J_PASSWORD)
            .option("labels", f":{neo4j_label}")
            .option("node.keys", "id")
            .save()
        )
```

Now in Neo4j Browser, you can query the PageRank scores:

```cypher
// Top questions by PageRank
MATCH (q:Question)
WHERE q.pagerank IS NOT NULL
RETURN q.Title, q.pagerank
ORDER BY q.pagerank DESC
LIMIT 10

// Top users by PageRank
MATCH (u:User)
WHERE u.pagerank IS NOT NULL
RETURN u.DisplayName, u.pagerank
ORDER BY u.pagerank DESC
LIMIT 10

// Questions with high PageRank and their askers
MATCH (u:User)-[:ASKS]->(q:Question)
WHERE q.pagerank > 1.0
RETURN u.DisplayName, q.Title, q.pagerank
ORDER BY q.pagerank DESC
LIMIT 10
```

# Use Cases for Neo4j + GraphFrames

This integration pattern is valuable for many scenarios:

## 1. Real-Time Analytics on Historical Computation

- Store graph structure in Neo4j for real-time queries
- Periodically run batch algorithms (PageRank, Community Detection) in Spark
- Update Neo4j with computed metrics
- Use metrics in real-time recommendations or dashboards

## 2. Exploratory Analysis

- Use Neo4j Browser to visually explore the graph
- Identify interesting subgraphs or patterns
- Export specific subgraphs to Spark for deeper analysis
- Store findings back in Neo4j for team collaboration

## 3. Multi-Algorithm Workflows

- Run multiple Spark algorithms (PageRank, Label Propagation, Connected Components)
- Store results as node properties in Neo4j
- Combine algorithmic results with transactional queries
- Build composite scores or recommendations

## 4. Data Engineering Pipelines

- Ingest data from various sources into Neo4j
- Export to Spark for cleansing, deduplication, entity resolution
- Import cleaned data back to Neo4j
- Maintain data quality while leveraging Spark's scale

# Advanced Topics

## Incremental Updates

Instead of rewriting all data, you can update incrementally:

```python
# Only update nodes that changed
new_pagerank_df = nodes_with_pagerank.join(
    old_pagerank_df, 
    on="id", 
    how="left"
).filter(
    (F.col("new_pagerank") - F.col("old_pagerank")).abs() > 0.001
)
```

## Custom Schema Mappings

Extend `Neo4jSchemaMapper` for your own data:

```python
class CustomSchemaMapper(Neo4jSchemaMapper):
    NODE_TYPE_LABELS = {
        **Neo4jSchemaMapper.NODE_TYPE_LABELS,
        "CustomType": "CustomLabel"
    }
    
    @staticmethod
    def get_node_properties(node_type: str) -> List[str]:
        if node_type == "CustomType":
            return ["id", "name", "custom_property"]
        return Neo4jSchemaMapper.get_node_properties(node_type)
```

## Performance Optimization

For large graphs:

1. **Partitioning**: Use more partitions for repartition operations
2. **Batching**: Write to Neo4j in batches with `.option("batch.size", "10000")`
3. **Indexes**: Create indexes in Neo4j on commonly queried properties
4. **Caching**: Cache intermediate results in Spark
5. **Checkpointing**: Use checkpoints to break lineage chains

## Query Pushdown

The Neo4j Connector supports query pushdown for efficient filtering:

```python
# Filter in Neo4j before loading into Spark
df = (
    spark.read
    .format("org.neo4j.spark.DataSource")
    .option("url", NEO4J_URL)
    .option("authentication.basic.username", NEO4J_USER)
    .option("authentication.basic.password", NEO4J_PASSWORD)
    .option("labels", "Question")
    .option("query", "MATCH (q:Question) WHERE q.Score > 10 RETURN q")
    .load()
)
```

# Troubleshooting

## Connection Issues

If you can't connect to Neo4j:

1. Check that Neo4j is running: `neo4j status` or check Docker container
2. Verify the URL (bolt:// or neo4j://)
3. Check authentication credentials
4. Ensure firewall allows port 7687

## Memory Issues

For large graphs:

1. Increase driver and executor memory: `--driver-memory 8g --executor-memory 8g`
2. Use more partitions when reading from Neo4j
3. Process node types separately if needed
4. Consider filtering data before loading

## Schema Mismatches

If properties don't match:

1. Check that column names match between GraphFrames and Neo4j
2. Use `.printSchema()` to inspect DataFrames
3. Verify the schema mapper includes all needed properties
4. Handle null values appropriately

## Performance Issues

If operations are slow:

1. Create indexes in Neo4j: `CREATE INDEX ON :Question(id)`
2. Use more Spark executors and cores
3. Adjust batch sizes for writes
4. Profile with Spark UI to identify bottlenecks

# Conclusion

In this tutorial, you learned how to:

1. Map between GraphFrames' unified schema and Neo4j's typed schema
2. Write graph data from GraphFrames to Neo4j
3. Read graph data from Neo4j into GraphFrames
4. Calculate PageRank on data from Neo4j
5. Store computed results back in Neo4j

This integration enables powerful workflows combining Neo4j's transactional graph database capabilities with GraphFrames' distributed graph algorithms. You can now:

- Store and query graphs interactively in Neo4j
- Scale computation to massive graphs with Spark
- Enrich your graph with computed metrics
- Build end-to-end graph analytics pipelines

For more information:
- [Neo4j Spark Connector Documentation](https://neo4j.com/docs/spark/current/)
- [GraphFrames User Guide](/04-user-guide/01-overview.md)
- [Network Motif Tutorial](/03-tutorials/02-motif-tutorial.md)

# Further Reading

- [Neo4j Graph Data Science Library](https://neo4j.com/docs/graph-data-science/current/) - Native Neo4j algorithms
- [Apache Spark GraphX](https://spark.apache.org/docs/latest/graphx-programming-guide.html) - Lower-level graph processing in Spark
- [Property Graph Model](https://neo4j.com/developer/graph-database/#property-graph) - Understanding graph data models
- [Cypher Query Language](https://neo4j.com/developer/cypher/) - Neo4j's graph query language
