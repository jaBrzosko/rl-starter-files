from SPARQLWrapper import SPARQLWrapper, JSON
from kg.recommendations.baseKGRecommender import BaseKGRecommender
import uuid
import rdflib

############################################
# -----------  MOCKED CONFIG  -------------
############################################
FUSEKI_CONFIG = {
    "UPDATE_URL": "http://localhost:3030/python/update",
    "QUERY_URL": "http://localhost:3030/python/query",
    "GRAPH_URI": "http://example.org/graph"
}

class FusekiKGOptimizedRecommender(BaseKGRecommender):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.update_endpoint = SPARQLWrapper(FUSEKI_CONFIG["UPDATE_URL"])
        self.query_endpoint = SPARQLWrapper(FUSEKI_CONFIG["QUERY_URL"])
        self.base_graph_uri = FUSEKI_CONFIG["GRAPH_URI"]

        # Unique graph resolves unnecessary conflicts when previous data was not cleared
        self.recommendations_graph_uri = f"{self.base_graph_uri}/recommendations/{uuid.uuid4()}"

        self._upload_initial_kg()

    ########################################
    # Upload ontology + recommendations
    ########################################
    def _upload_initial_kg(self):
        triples = (self.ontology + self.recommendations).serialize(format='nt')
        self._send_update(f"""
        INSERT DATA {{
            GRAPH <{self.recommendations_graph_uri}> {{
                {triples}
            }}
        }}
        """)

    ########################################
    # Override base method
    ########################################
    def get_action_mask(self, map_array):
        action_mask = [1.0] * self.action_size

        map_id = f"Map_{uuid.uuid4()}"
        map_obj = f"{self.domain}{map_id}"
        tmp_graph = f"{self.base_graph_uri}/temp/{uuid.uuid4()}"

        # Build N-Triples directly as a string
        triples = []
        
        # Map triples
        triples.append(f"<{map_obj}> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <{self.domain}Map> .")
        triples.append(f"<{map_obj}> <{self.domain}mapId> \"{map_id}\" .")
        triples.append(f"<{map_obj}> <{self.domain}hasProblemType> <{self.problem_type}> .")

        # Cell triples
        for x, row in enumerate(map_array):
            for y, cell in enumerate(row):
                cell_obj = f"{self.domain}Cell_{x}_{y}_{uuid.uuid4()}"
                triples.append(f"<{map_obj}> <{self.domain}hasCell> <{cell_obj}> .")
                triples.append(f"<{cell_obj}> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <{self.domain}Cell> .")
                triples.append(f"<{cell_obj}> <{self.domain}coordinateX> \"{x}\"^^<http://www.w3.org/2001/XMLSchema#integer> .")
                triples.append(f"<{cell_obj}> <{self.domain}coordinateY> \"{y}\"^^<http://www.w3.org/2001/XMLSchema#integer> .")
                triples.append(f"<{cell_obj}> <{self.domain}channelValue> \"{','.join(map(str, [int(v) for v in cell]))}\" .")

        triples_str = "\n".join(triples)

        # Insert data
        self._send_update(f"""
        INSERT DATA {{
            GRAPH <{tmp_graph}> {{
                {triples_str}
            }}
        }}
        """)

        # Query
        sparql_query = self.query.replace("MAP_ID_REPLACE", map_id) \
            .replace("RECOMMENDATIONS_GRAPH_REPLACE", self.recommendations_graph_uri) \
            .replace("MAP_GRAPH_REPLACE", tmp_graph)

        self.query_endpoint.setQuery(sparql_query)
        self.query_endpoint.setReturnFormat(JSON)

        results = self.query_endpoint.query().convert()
        for row in results["results"]["bindings"]:
            idx = int(row["maskedActionIndex"]["value"])
            weight = float(row["maskWeight"]["value"])
            action_mask[idx] = action_mask[idx] * (1 - weight)

        # Cleanup
        self._send_update(f"DROP GRAPH <{tmp_graph}>")
        return action_mask

    ########################################
    # Helper
    ########################################
    def _send_update(self, sparql):
        self.update_endpoint.setMethod("POST")
        self.update_endpoint.setQuery(sparql)
        self.update_endpoint.query()

    ########################################
    #   ABSTRACT QUERY IMPLEMENTATION
    ########################################
    def _execute_query(self, map_kg, map_id):
        # Not used in optimized version
        pass
