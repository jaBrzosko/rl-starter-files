import uuid
import rdflib
from abc import ABC, abstractmethod

############################################
# ----------- ABSTRACT BASE ---------------
############################################

class BaseKGRecommender(ABC):
    def __init__(self, ontology_file, recommendation_file, query, domain_uri, env_problem_type, action_size=7):
        self.ontology_file = ontology_file
        self.recommendation_file = recommendation_file
        self.query = query
        self.domain_uri = domain_uri
        self.env_problem_type = env_problem_type
        self.action_size = action_size

        self.domain = rdflib.Namespace(domain_uri)
        self.problem_type = rdflib.URIRef(f"{domain_uri}problemType_{env_problem_type}")

        self.ontology = self._load_graph(ontology_file)
        self.recommendations = self._load_graph(recommendation_file)

    @staticmethod
    def _load_graph(path):
        g = rdflib.Graph()
        g.parse(path, format="turtle")
        return g

    def get_action_mask(self, map_array):
        kg, map_id = self._build_map_graph(map_array)
        return self._execute_query(kg, map_id)

    ########################################
    #       SHARED DATA BUILDING
    ########################################
    def _build_map_graph(self, map_array):
        kg = rdflib.Graph()
        map_id = f"Map_{uuid.uuid4()}"
        map_obj = rdflib.URIRef(f"{self.domain}{map_id}")

        kg.add((map_obj, rdflib.RDF.type, self.domain.Map))
        kg.add((map_obj, self.domain.mapId, rdflib.Literal(map_id)))
        kg.add((map_obj, self.domain.hasProblemType, self.problem_type))

        for x, row in enumerate(map_array):
            for y, cell in enumerate(row):
                cell_obj = rdflib.URIRef(f"{self.domain}Cell_{x}_{y}_{uuid.uuid4()}")
                kg.add((map_obj, self.domain.hasCell, cell_obj))

                kg.add((cell_obj, rdflib.RDF.type, self.domain.Cell))
                kg.add((cell_obj, self.domain.coordinateX, rdflib.Literal(x)))
                kg.add((cell_obj, self.domain.coordinateY, rdflib.Literal(y)))
                kg.add((cell_obj, self.domain.channelValue, rdflib.Literal(",".join(map(str, cell)))))

        return kg, map_id

    ########################################
    #   ABSTRACT QUERY IMPLEMENTATION
    ########################################
    @abstractmethod
    def _execute_query(self, map_kg, map_id):
        pass

