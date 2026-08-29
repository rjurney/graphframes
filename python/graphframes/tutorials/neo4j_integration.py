"""Neo4j Integration Tutorial - Load Stack Exchange data into Neo4j, process with GraphFrames, and store results back."""  # noqa: E501

#
# Interactive Usage: 
#   pyspark --packages graphframes:graphframes:0.8.4-spark3.5-s_2.12,org.neo4j:neo4j-connector-apache-spark_2.12:5.3.0_for_spark_3
#
# Batch Usage:
#   spark-submit \
#   --packages graphframes:graphframes:0.8.4-spark3.5-s_2.12,org.neo4j:neo4j-connector-apache-spark_2.12:5.3.0_for_spark_3 \
#   python/graphframes/tutorials/neo4j_integration.py
#

from __future__ import annotations

from typing import Dict, List

import click
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import DataFrame, SparkSession

from graphframes import GraphFrame

# Change me if you download a different stackexchange site
STACKEXCHANGE_SITE = "stats.meta.stackexchange.com"
BASE_PATH = f"python/graphframes/tutorials/data/{STACKEXCHANGE_SITE}"

# Neo4j connection configuration - customize these for your Neo4j instance
NEO4J_URL = "neo4j://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"


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

    @staticmethod
    def get_node_properties(node_type: str) -> List[str]:
        """Get the relevant properties for a given node type.

        Parameters
        ----------
        node_type : str
            The type of node (e.g., 'Question', 'Answer', 'User')

        Returns
        -------
        List[str]
            List of property names relevant to this node type
        """
        # Define properties specific to each node type
        property_map = {
            "Question": [
                "id",
                "Id",
                "Title",
                "Body",
                "Score",
                "ViewCount",
                "OwnerUserId",
                "CreationDate",
                "Tags",
                "AcceptedAnswerId",
                "AnswerCount",
                "CommentCount",
                "ContentLicense",
            ],
            "Answer": [
                "id",
                "Id",
                "Body",
                "Score",
                "ParentId",
                "OwnerUserId",
                "CreationDate",
                "CommentCount",
                "ContentLicense",
            ],
            "User": [
                "id",
                "Id",
                "DisplayName",
                "Reputation",
                "CreationDate",
                "LastAccessDate",
                "Location",
                "AboutMe",
                "Views",
                "UpVotes",
                "DownVotes",
            ],
            "Vote": [
                "id",
                "Id",
                "PostId",
                "VoteTypeId",
                "CreationDate",
            ],
            "Badge": [
                "id",
                "Id",
                "UserId",
                "Name",
                "Date",
                "Class",
                "TagBased",
            ],
            "Tag": [
                "id",
                "Id",
                "TagName",
                "Count",
                "ExcerptPostId",
                "WikiPostId",
            ],
            "PostLinks": [
                "id",
                "Id",
                "PostId",
                "RelatedPostId",
                "LinkTypeId",
                "CreationDate",
            ],
        }
        return property_map.get(node_type, ["id"])

    @staticmethod
    def nodes_to_neo4j_format(nodes_df: DataFrame, node_type: str) -> DataFrame:
        """Convert GraphFrames nodes to Neo4j format for a specific type.

        Parameters
        ----------
        nodes_df : DataFrame
            The unified nodes DataFrame from GraphFrames
        node_type : str
            The type of nodes to extract

        Returns
        -------
        DataFrame
            DataFrame formatted for Neo4j import with only relevant properties
        """
        # Filter to specific node type
        filtered_df = nodes_df.filter(F.col("Type") == node_type)

        # Select only relevant properties for this node type
        properties = Neo4jSchemaMapper.get_node_properties(node_type)
        available_properties = [prop for prop in properties if prop in filtered_df.columns]

        return filtered_df.select(available_properties)

    @staticmethod
    def neo4j_to_graphframes_nodes(
        spark: SparkSession, node_dfs: Dict[str, DataFrame]
    ) -> DataFrame:
        """Combine Neo4j node types into a unified GraphFrames nodes DataFrame.

        Parameters
        ----------
        spark : SparkSession
            Active Spark session
        node_dfs : Dict[str, DataFrame]
            Dictionary mapping node type names to their DataFrames

        Returns
        -------
        DataFrame
            Unified nodes DataFrame compatible with GraphFrames
        """
        # Collect all unique columns across all node types
        all_cols = set()
        for df in node_dfs.values():
            all_cols.update(df.columns)
        all_column_names = sorted(list(all_cols))

        # Add missing columns to each DataFrame
        unified_dfs = []
        for node_type, df in node_dfs.items():
            # Add Type column
            df = df.withColumn("Type", F.lit(node_type))

            # Add missing columns with null values
            for col_name in all_column_names:
                if col_name not in df.columns:
                    df = df.withColumn(col_name, F.lit(None))

            unified_dfs.append(df.select(sorted(df.columns)))

        # Union all DataFrames
        result_df = unified_dfs[0]
        for df in unified_dfs[1:]:
            result_df = result_df.union(df)

        return result_df


def write_nodes_to_neo4j(spark: SparkSession, nodes_df: DataFrame) -> None:
    """Write nodes from GraphFrames to Neo4j, separating by node type.

    Parameters
    ----------
    spark : SparkSession
        Active Spark session
    nodes_df : DataFrame
        The unified nodes DataFrame from GraphFrames
    """
    click.echo("\n=== Writing Nodes to Neo4j ===\n")

    for node_type in Neo4jSchemaMapper.NODE_TYPE_LABELS.keys():
        neo4j_label = Neo4jSchemaMapper.NODE_TYPE_LABELS[node_type]
        click.echo(f"Writing {node_type} nodes with label {neo4j_label}...")

        # Convert to Neo4j format
        type_df = Neo4jSchemaMapper.nodes_to_neo4j_format(nodes_df, node_type)

        # Write to Neo4j
        (
            type_df.write.format("org.neo4j.spark.DataSource")
            .mode("Overwrite")
            .option("url", NEO4J_URL)
            .option("authentication.basic.username", NEO4J_USER)
            .option("authentication.basic.password", NEO4J_PASSWORD)
            .option("labels", f":{neo4j_label}")
            .option("node.keys", "id")
            .save()
        )

        count = type_df.count()
        click.echo(f"  ✓ Wrote {count:,} {node_type} nodes\n")


def write_edges_to_neo4j(edges_df: DataFrame) -> None:
    """Write edges from GraphFrames to Neo4j as relationships.

    Parameters
    ----------
    edges_df : DataFrame
        The edges DataFrame from GraphFrames
    """
    click.echo("\n=== Writing Relationships to Neo4j ===\n")

    for rel_type in Neo4jSchemaMapper.RELATIONSHIP_TYPES.keys():
        neo4j_rel = Neo4jSchemaMapper.RELATIONSHIP_TYPES[rel_type]
        click.echo(f"Writing {rel_type} relationships as {neo4j_rel}...")

        # Filter to specific relationship type
        type_df = edges_df.filter(F.col("relationship") == rel_type)

        # Write to Neo4j
        (
            type_df.write.format("org.neo4j.spark.DataSource")
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

        count = type_df.count()
        click.echo(f"  ✓ Wrote {count:,} {rel_type} relationships\n")


def read_nodes_from_neo4j(spark: SparkSession) -> DataFrame:
    """Read nodes from Neo4j and combine into GraphFrames format.

    Parameters
    ----------
    spark : SparkSession
        Active Spark session

    Returns
    -------
    DataFrame
        Unified nodes DataFrame compatible with GraphFrames
    """
    click.echo("\n=== Reading Nodes from Neo4j ===\n")

    node_dfs = {}
    for node_type, neo4j_label in Neo4jSchemaMapper.NODE_TYPE_LABELS.items():
        click.echo(f"Reading {node_type} nodes with label {neo4j_label}...")

        df = (
            spark.read.format("org.neo4j.spark.DataSource")
            .option("url", NEO4J_URL)
            .option("authentication.basic.username", NEO4J_USER)
            .option("authentication.basic.password", NEO4J_PASSWORD)
            .option("labels", neo4j_label)
            .load()
        )

        node_dfs[node_type] = df
        count = df.count()
        click.echo(f"  ✓ Read {count:,} {node_type} nodes\n")

    # Combine into unified format
    return Neo4jSchemaMapper.neo4j_to_graphframes_nodes(spark, node_dfs)


def read_edges_from_neo4j(spark: SparkSession) -> DataFrame:
    """Read edges from Neo4j and combine into GraphFrames format.

    Parameters
    ----------
    spark : SparkSession
        Active Spark session

    Returns
    -------
    DataFrame
        Edges DataFrame compatible with GraphFrames
    """
    click.echo("\n=== Reading Relationships from Neo4j ===\n")

    edge_dfs = []
    for rel_type, neo4j_rel in Neo4jSchemaMapper.RELATIONSHIP_TYPES.items():
        click.echo(f"Reading {rel_type} relationships as {neo4j_rel}...")

        df = (
            spark.read.format("org.neo4j.spark.DataSource")
            .option("url", NEO4J_URL)
            .option("authentication.basic.username", NEO4J_USER)
            .option("authentication.basic.password", NEO4J_PASSWORD)
            .option("relationship", neo4j_rel)
            .option("relationship.source.labels", ":Node")
            .option("relationship.target.labels", ":Node")
            .load()
        )

        # Add relationship type column and rename columns for GraphFrames
        df = (
            df.withColumn("relationship", F.lit(rel_type))
            .withColumnRenamed("source.id", "src")
            .withColumnRenamed("target.id", "dst")
            .select("src", "dst", "relationship")
        )

        edge_dfs.append(df)
        count = df.count()
        click.echo(f"  ✓ Read {count:,} {rel_type} relationships\n")

    # Union all edge DataFrames
    result_df = edge_dfs[0]
    for df in edge_dfs[1:]:
        result_df = result_df.union(df)

    return result_df


def write_pagerank_to_neo4j(nodes_with_pagerank: DataFrame) -> None:
    """Write PageRank results back to Neo4j, updating node properties.

    Parameters
    ----------
    nodes_with_pagerank : DataFrame
        Nodes DataFrame with pagerank column
    """
    click.echo("\n=== Writing PageRank Results to Neo4j ===\n")

    # Select only id and pagerank
    pagerank_df = nodes_with_pagerank.select("id", "Type", "pagerank")

    for node_type in Neo4jSchemaMapper.NODE_TYPE_LABELS.keys():
        neo4j_label = Neo4jSchemaMapper.NODE_TYPE_LABELS[node_type]
        click.echo(f"Updating PageRank for {node_type} nodes...")

        type_df = pagerank_df.filter(F.col("Type") == node_type)

        # Write pagerank back to Neo4j
        (
            type_df.write.format("org.neo4j.spark.DataSource")
            .mode("Append")
            .option("url", NEO4J_URL)
            .option("authentication.basic.username", NEO4J_USER)
            .option("authentication.basic.password", NEO4J_PASSWORD)
            .option("labels", f":{neo4j_label}")
            .option("node.keys", "id")
            .save()
        )

        count = type_df.count()
        click.echo(f"  ✓ Updated PageRank for {count:,} {node_type} nodes\n")


def main():
    """Main function to demonstrate Neo4j integration with GraphFrames."""
    # Initialize Spark Session
    click.echo("Initializing Spark Session...")
    spark: SparkSession = (
        SparkSession.builder.appName("GraphFrames Neo4j Integration")
        .config("spark.sql.caseSensitive", True)
        .getOrCreate()
    )
    spark.sparkContext.setCheckpointDir("/tmp/graphframes-checkpoints/neo4j")

    click.echo("\n" + "=" * 60)
    click.echo("GraphFrames Neo4j Integration Tutorial")
    click.echo("=" * 60)

    # Step 1: Load data from local parquet files (created by stackexchange.py)
    click.echo("\n=== Step 1: Load Stack Exchange Data ===\n")

    NODES_PATH: str = f"{BASE_PATH}/Nodes.parquet"
    EDGES_PATH: str = f"{BASE_PATH}/Edges.parquet"

    click.echo(f"Loading nodes from {NODES_PATH}...")
    nodes_df: DataFrame = spark.read.parquet(NODES_PATH)
    click.echo(f"  ✓ Loaded {nodes_df.count():,} nodes\n")

    click.echo(f"Loading edges from {EDGES_PATH}...")
    edges_df: DataFrame = spark.read.parquet(EDGES_PATH)
    click.echo(f"  ✓ Loaded {edges_df.count():,} edges\n")

    # Step 2: Write data to Neo4j
    write_nodes_to_neo4j(spark, nodes_df)
    write_edges_to_neo4j(edges_df)

    # Step 3: Read data back from Neo4j
    neo4j_nodes_df = read_nodes_from_neo4j(spark)
    neo4j_edges_df = read_edges_from_neo4j(spark)

    # Repartition for better parallelism
    neo4j_nodes_df = neo4j_nodes_df.repartition(50).checkpoint().cache()
    neo4j_edges_df = neo4j_edges_df.repartition(50).checkpoint().cache()

    # Step 4: Create GraphFrame
    click.echo("\n=== Step 4: Create GraphFrame ===\n")
    g = GraphFrame(neo4j_nodes_df, neo4j_edges_df)
    click.echo(f"GraphFrame created with {g.vertices.count():,} vertices and {g.edges.count():,} edges\n")  # noqa: E501

    # Display node type distribution
    click.echo("Node type distribution:")
    node_counts = (
        g.vertices.select("id", F.col("Type").alias("Node Type"))
        .groupBy("Node Type")
        .count()
        .orderBy(F.col("count").desc())
        .withColumn("count", F.format_number(F.col("count"), 0))
    )
    node_counts.show()

    # Step 5: Calculate PageRank
    click.echo("\n=== Step 5: Calculate PageRank ===\n")
    click.echo("Running PageRank algorithm (this may take a few minutes)...")

    pagerank_result = g.pageRank(resetProbability=0.15, maxIter=10)
    nodes_with_pagerank = pagerank_result.vertices

    click.echo("  ✓ PageRank calculation complete\n")

    # Show top nodes by PageRank
    click.echo("Top 10 nodes by PageRank:")
    (
        nodes_with_pagerank.select("id", "Type", "pagerank")
        .orderBy(F.col("pagerank").desc())
        .show(10, truncate=False)
    )

    # Show top nodes by type
    click.echo("\nTop 5 Questions by PageRank:")
    (
        nodes_with_pagerank.filter(F.col("Type") == "Question")
        .select("Id", "Title", "pagerank")
        .orderBy(F.col("pagerank").desc())
        .show(5, truncate=50)
    )

    click.echo("\nTop 5 Users by PageRank:")
    (
        nodes_with_pagerank.filter(F.col("Type") == "User")
        .select("Id", "DisplayName", "pagerank")
        .orderBy(F.col("pagerank").desc())
        .show(5, truncate=False)
    )

    # Step 6: Write PageRank results back to Neo4j
    write_pagerank_to_neo4j(nodes_with_pagerank)

    click.echo("\n" + "=" * 60)
    click.echo("Tutorial Complete!")
    click.echo("=" * 60)
    click.echo(
        "\nYou can now query Neo4j to see the PageRank scores stored as node properties."
    )
    click.echo(
        'Example Cypher query: MATCH (n:Question) RETURN n.Title, n.pagerank ORDER BY n.pagerank DESC LIMIT 10'  # noqa: E501
    )


if __name__ == "__main__":
    main()
