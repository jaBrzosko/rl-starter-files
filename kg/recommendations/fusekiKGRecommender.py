from SPARQLWrapper import SPARQLWrapper, JSON
from kg.recommendations.baseKGRecommender import BaseKGRecommender
import uuid

FUSEKI_CONFIG = {
    "UPDATE_URL": "http://localhost:3030/python/update",
    "QUERY_URL": "http://localhost:3030/python/query",
    "GRAPH_URI": "http://example.org/graph"
}

class FusekiKGRecommender(BaseKGRecommender):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.update_endpoint = SPARQLWrapper(FUSEKI_CONFIG["UPDATE_URL"])
        self.query_endpoint = SPARQLWrapper(FUSEKI_CONFIG["QUERY_URL"])
        self.base_graph_uri = FUSEKI_CONFIG["GRAPH_URI"]

        self.recommendations_graph_uri = f"{self.base_graph_uri}/recommendations/{uuid.uuid4()}"

        self._upload_initial_kg()

    def _upload_initial_kg(self):
        triples = (self.ontology + self.recommendations).serialize(format='nt')
        self._send_update(f"""
        INSERT DATA {{
            GRAPH <{self.recommendations_graph_uri}> {{
                {triples}
            }}
        }}
        """)

    def _execute_query(self, map_kg, map_id):
        action_mask = [1.0] * self.action_size

        tmp_graph = f"{self.base_graph_uri}/temp/{uuid.uuid4()}"
        triples = map_kg.serialize(format='nt')

        self._send_update(f"""
        INSERT DATA {{
            GRAPH <{tmp_graph}> {{
                {triples}
            }}
        }}
        """)

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

        self._send_update(f"DROP GRAPH <{tmp_graph}>")
        return action_mask

    def _send_update(self, sparql):
        self.update_endpoint.setMethod("POST")
        self.update_endpoint.setQuery(sparql)
        self.update_endpoint.query()
